from typing import TypeAlias, Tuple, Union
import torch
from torch_geometric.utils import dense_to_sparse
from dataclasses import dataclass


GateType: TypeAlias = Tuple[int,int]
CircSliceType: TypeAlias = Tuple[GateType, ...]



@dataclass
class Circuit:
  slice_gates: Tuple[CircSliceType, ...]
  n_qubits: int
  # n_gates: total number of gates in the circuit
  # alloc_steps: refer to the function __getAllocOrder for info on this attribute
  # n_steps: len(alloc_steps)
  # adj_matrices: per-slice adjacency matrices of qubit interactions


  def __post_init__(self):
    self.n_gates = self.__get_N_gates()
    self.alloc_steps = self.__get_alloc_order()
    self.n_steps = len(self.alloc_steps)
    self.adj_matrices = self.__get_adj_matrices()


  def __get_N_gates(self) -> int:
    return sum(len(slice_i) for slice_i in self.slice_gates)
  

  def __get_adj_matrices(self) -> torch.Tensor:
    matrices = torch.eye(self.n_qubits).repeat(self.n_slices,1,1)
    for s_i, slice in enumerate(self.slice_gates):
      for (a,b) in slice:
        matrices[s_i,a,b] = matrices[s_i,b,a] = 1
        matrices[s_i,a,a] = matrices[s_i,b,b] = 0
    return matrices


  def __get_alloc_order(self) -> Tuple[Tuple[int, Union[GateType, Tuple[int]]], ...]:
    ''' Get the allocation order of te qubits for a given circuit.

    Returns a tuple with the allocations to be performed. Each tuple element is another tuple that
    contains the slice the allocation corresponds to; the qubits involved in the allocation, two
    if the qubits belong to a gate in that time slice or a single one if they don't; and the number
    of gates that remains to be allocated.
    '''
    gate_counter = 0
    allocations = []
    for slice_i, slice in enumerate(self.slice_gates):
      free_qubits = set(range(self.n_qubits))
      for gate in slice:
        allocations.append((slice_i, gate, self.n_gates - gate_counter))
        gate_counter += 1
        free_qubits -= set(gate) # Remove qubits in gates from set of free qubits
      for q in free_qubits:
        allocations.append((slice_i, (q,), self.n_gates - gate_counter))
    return tuple(allocations)
  

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
    self.sparse_core_con, self.sparse_core_ws = dense_to_sparse(self.core_connectivity.float())

  
  @property
  def n_cores(self):
    return len(self.core_capacities)
  

  @property
  def n_physical_qubits(self):
    return sum(self.core_capacities).item()