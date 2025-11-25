from __future__ import annotations
from typing import TypeAlias, Tuple, Optional
import torch
from collections import defaultdict
from dataclasses import dataclass
from qiskit.circuit import QuantumCircuit

GateType: TypeAlias = Tuple[int,int]
CircSliceType: TypeAlias = Tuple[GateType, ...]



@dataclass
class Circuit:
  slice_gates: Tuple[CircSliceType, ...]
  n_qubits: int


  def __str__(self):
    return f"Circuit(slice_gates={self.slice_gates}, n_qubits={self.n_qubits})"


  @staticmethod
  def from_qasm(qasm_file: str, n_qubits: Optional[int] = None) -> Circuit:
    circuit = QuantumCircuit.from_qasm_file(qasm_file)
    if n_qubits is None:
      n_qubits = circuit.num_qubits
    gates = []
    for ins in circuit:
      qubits = tuple(circuit.find_bit(q)[0] for q in ins.qubits)
      if len(qubits) == 2:
        gates.append(qubits)
      elif len(qubits) > 2 and ins.name != 'barrier':
        raise Exception(f"Circuit contains at least one gate with more than two qubits: {qubits} {ins}")
    return Circuit.from_gate_list(gates, n_qubits)


  @staticmethod
  def from_gate_list(gates: list[tuple[int,int]], n_qubits: Optional[int] = None) -> Circuit:
    slices = defaultdict(list)
    used_qubits = defaultdict(set)
    
    def add_gate(gate_id, t):
      q_set = set(gates[gate_id])
      if used_qubits[t].intersection(q_set):
        t += 1
        slices[t].append(gate_id)
        used_qubits[t] |= q_set
      elif t == 0:
        slices[t].append(gate_id)
        used_qubits[t] |= q_set
      else:
        add_gate(gate_id, t-1)
        
    for g in range(len(gates)):
      add_gate(g, len(slices))
    
    slice_gates = tuple(tuple(gates[id] for id in slice_ids) for slice_ids in slices.values())
    if n_qubits is None:
      n_qubits = max(max(g) for g in gates)
    return Circuit(slice_gates=slice_gates, n_qubits=n_qubits)


  # Some more attribute declarations that are only computed when required (lazy
  # initialization), as these can be expensive to compute and not always needed
  @property
  def n_gates(self) -> int:
    if not hasattr(self, "n_gates_"):
      self.n_gates_ = sum(len(slice_i) for slice_i in self.slice_gates)
    return self.n_gates_
  
  @property
  def n_gates_norm(self) -> int:
    if not hasattr(self, "n_gates_norm_"):
      self.n_gates_norm_ = sum(len(slice_i) for slice_i in self.slice_gates[1:])
    return self.n_gates_norm_
    
  @property
  def alloc_steps(self) -> torch.Tensor:
    if not hasattr(self, "alloc_steps_"):
      self.alloc_steps_ = self._get_alloc_order()
    return self.alloc_steps_
  
  @property
  def alloc_slices(self) -> list[tuple[int, list[int], list[tuple[int,int]]]]:
    if not hasattr(self, "alloc_slices_"):
      self.alloc_slices_ = self._get_alloc_slices()
    return self.alloc_slices_
  
  @property
  def n_steps(self) -> int:
    if not hasattr(self, "n_steps_"):
        self.n_steps_ = self.alloc_steps.shape[0]
    return self.n_steps_
  
  @property
  def adj_matrices(self) -> torch.Tensor:
    if not hasattr(self, "adj_matrices_"):
      self.adj_matrices_ = self._get_adj_matrices()
    return self.adj_matrices_
  
  @property
  def embedding(self) -> torch.Tensor:
    if not hasattr(self, 'embedding_'):
      self.embedding_ = self._get_embedding()
    return self.embedding_


  def _get_alloc_slices(self) -> list[tuple[int, list[int], list[tuple[int,int]]]]:
    gates_per_slice = [len(s) for s in self.slice_gates]
    remaining_gates_list = [sum(gates_per_slice[i:]) for i in range(1, self.n_slices+1)]
    alloc_slices = []
    for s, rem_gates in zip(self.slice_gates, remaining_gates_list):
      paired_qubits = set(sum(s, tuple()))
      free_qubits = tuple(set(range(self.n_qubits)) - paired_qubits)
      alloc_slices.append((rem_gates, free_qubits, s))
    return alloc_slices

  def _get_adj_matrices(self) -> torch.Tensor:
    matrices = torch.zeros(size=(self.n_slices,self.n_qubits,self.n_qubits))
    for s_i, slice in enumerate(self.slice_gates):
      for (a,b) in slice:
        matrices[s_i,a,b] = matrices[s_i,b,a] = 1
    return matrices
  
  def _get_embedding(self) -> torch.Tensor:
    adj = self.adj_matrices
    embeddings = torch.empty_like(adj, dtype=torch.float)
    embeddings[-1] = 0.5 * adj[-1]
    for slice_i in range(self.n_slices-2,-1,-1):
      embeddings[slice_i] = 0.5 * (embeddings[slice_i + 1] + adj[slice_i])
    return embeddings

  def _get_alloc_order(self, order=True) -> torch.Tensor:
    ''' Get the allocation order of te qubits for a given circuit.

    Returns a tensor with the allocations to be performed. Each row contains 4
    columns: the first item indicates the slice index of the allocation; the
    second the first qubit to be allocated; the third the second qubit to be
    allocated or -1 if single qubit allocation step; and the fourth column the
    number of gates that remain to be allocated (excluding those in the first slice).
    '''
    gate_counter = 0
    n_steps = self.n_slices*self.n_qubits - self.n_gates
    allocations = torch.empty([n_steps, 4], dtype=torch.int32)
    alloc_step = 0
    embs = self.embedding
    for slice_i, slice in enumerate(self.slice_gates):
      free_qubits = set(range(self.n_qubits))
      # Order gates by interaction intensity of the two qubits in the gate
      ordered_gates = [(embs[slice_i,g0,g1], (g0,g1)) for (g0,g1) in slice]
      if order:
        ordered_gates.sort(reverse=True)
      for _, gate in ordered_gates:
        allocations[alloc_step,0] = slice_i
        allocations[alloc_step,1] = gate[0]
        allocations[alloc_step,2] = gate[1]
        allocations[alloc_step,3] = self.n_gates_norm - gate_counter
        gate_counter += 1 if slice_i != 0 else 0
        free_qubits -= set(gate) # Remove qubits in gates from set of free qubits
        alloc_step += 1
      # Order free qubits by highest interaction intensity (ignoring itself)
      ordered_fq = [(embs[slice_i,q,torch.arange(self.n_qubits) != q].max(), (q,)) for q in free_qubits]
      if order:
        ordered_fq.sort(reverse=True)
      for (_, (q,)) in ordered_fq:
        allocations[alloc_step,0] = slice_i
        allocations[alloc_step,1] = q
        allocations[alloc_step,2] = -1
        allocations[alloc_step,3] = self.n_gates_norm - gate_counter
        alloc_step += 1
    return allocations
  
  @property
  def n_slices(self) -> int:
    return len(self.slice_gates)



@dataclass
class Hardware:
  core_capacities: torch.Tensor
  core_connectivity: torch.Tensor
  # sparse_core_con: automatically set in init, has the core_connectivity matrix in sparse format
  # sparse_core_ws: weights of the sparse_core_con matrix


  def __post_init__(self):
    ''' Ensures the correctness of the data.
    '''
    assert len(self.core_capacities.shape) == 1, "Core capacities must be a vector"
    assert not torch.is_floating_point(self.core_capacities), "Core capacities must be of dtype int"
    assert all(self.core_capacities > 0), f"All core capacities should be greater than 0"
    assert len(self.core_connectivity.shape) == 2 and \
           self.core_connectivity.shape[0] == self.core_connectivity.shape[1], \
      f"Core connectivity should be a square matrix, found matrix of shape {self.core_capacities.shape}"
    assert torch.all(self.core_connectivity == self.core_connectivity.T), \
      "Core connectivity matrix should be symmetric"

  
  @property
  def n_cores(self):
    return len(self.core_capacities)
  

  @property
  def n_qubits(self):
    return sum(self.core_capacities).item()