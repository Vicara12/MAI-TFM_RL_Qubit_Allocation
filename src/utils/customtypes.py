from typing import TypeAlias, Tuple, Union
import torch
from dataclasses import dataclass


GateType: TypeAlias = Tuple[int,int]
CircSliceType: TypeAlias = Tuple[GateType, ...]



@dataclass
class Circuit:
  slice_gates: Tuple[CircSliceType, ...]
  n_qubits: int

  # Some more attribute declarations that are only computed when required (lazy
  # initialization), as these can be expensive to compute and not always needed
  @property
  def n_gates(self) -> int:
    if not hasattr(self, "n_gates_"):
      self.n_gates_ = self._get_N_gates()
    return self.n_gates_
    
  @property
  def alloc_steps(self) -> torch.Tensor:
    if not hasattr(self, "alloc_steps_"):
      self.alloc_steps_ = self._get_alloc_order()
    return self.alloc_steps_
  
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

  def _get_N_gates(self) -> int:
    return sum(len(slice_i) for slice_i in self.slice_gates)

  def _get_adj_matrices(self) -> torch.Tensor:
    matrices = torch.eye(self.n_qubits).repeat(self.n_slices,1,1)
    for s_i, slice in enumerate(self.slice_gates):
      for (a,b) in slice:
        matrices[s_i,a,b] = matrices[s_i,b,a] = 1
        matrices[s_i,a,a] = matrices[s_i,b,b] = 0
    return matrices

  def _get_alloc_order(self) -> torch.Tensor:
    ''' Get the allocation order of te qubits for a given circuit.

    Returns a tensor with the allocations to be performed. Each row contains 4
    columns: the first item indicates the slice index of the allocation; the
    second the first qubit to be allocated; the third the second qubit to be
    allocated or -1 if single qubit allocation step; and the fourth column the
    number of gates that remain to be allocated.
    '''
    gate_counter = 0
    n_steps = self.n_slices*self.n_qubits - self.n_gates
    allocations = torch.empty([n_steps, 4], dtype=torch.int32)
    alloc_step = 0
    for slice_i, slice in enumerate(self.slice_gates):
      free_qubits = set(range(self.n_qubits))
      for gate in slice:
        allocations[alloc_step,0] = slice_i
        allocations[alloc_step,1] = gate[0]
        allocations[alloc_step,2] = gate[1]
        allocations[alloc_step,3] = self.n_gates - gate_counter
        gate_counter += 1
        free_qubits -= set(gate) # Remove qubits in gates from set of free qubits
        alloc_step += 1
      for q in free_qubits:
        allocations[alloc_step,0] = slice_i
        allocations[alloc_step,1] = q
        allocations[alloc_step,2] = -1
        allocations[alloc_step,3] = self.n_gates - gate_counter
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
  def n_physical_qubits(self):
    return sum(self.core_capacities).item()