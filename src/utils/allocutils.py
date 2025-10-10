import torch
from copy import copy
from typing import Tuple, List
from utils.customtypes import Hardware, Circuit


def sol_cost(allocations: torch.Tensor, core_con: torch.Tensor) -> int:
  ''' Compute the cost of the allocation with the given core connectivity matrix.

  The cost is computed as the number of swaps and the cost per swap.

  Args:
    - allocations: matrix of shape [Q,T] where Q = number of logical qubits, T = number of time slices.
    - core_con: matrix of shape [C,C] where C = number of cores. Item (i,j) indicates the cost of
        swapping qubits at cores i and j. No self loops, diagonal must be zero.
  '''
  num_slices = allocations.shape[0]
  cost = torch.tensor(0.0)
  for i in range(num_slices-1):
    cost += core_con[allocations[i,:].flatten(), allocations[i+1,:].flatten()].sum()
  return cost.item()


def check_sanity(
  allocs: torch.Tensor,
  circuit: Circuit,
  hardware: Hardware
):
  nans = torch.isnan(allocs)
  if torch.any(nans):
    raise Exception((
      f'Found Nan(s) in allocations at (slice,qubit): '
      f'{(nans).nonzero(as_tuple=False).tolist()}'
    ))
  # Check valid core
  valid_allocs = torch.logical_and(allocs >= 0, allocs < hardware.n_cores)
  if not torch.all(valid_allocs):
    raise Exception((
      f'Some qubit(s) has been allocated to an invalid core (slice,qubit): '
      f'{(~valid_allocs).nonzero(as_tuple=False).tolist()}'
    ))
  # Check core capacities
  for i, slice in enumerate(allocs):
    valid_core_caps = (slice.bincount(minlength=hardware.n_cores) <= hardware.core_capacities)
    if not torch.all(valid_core_caps):
      raise Exception((
        f'Overflowed core capacity for slice {i} and core(s) '
        f'{(~valid_core_caps).nonzero(as_tuple=False).tolist()}'
      ))
  # Check qubits that belong to same gate are in same core
  for i, (circuit_slice, alloc_slice) in enumerate(zip(circuit.slice_gates, allocs)):
    for gate in circuit_slice:
      if not (alloc_slice[gate[0]] == alloc_slice[gate[1]]):
        return ((
          f'Qubits {gate} in slice {i} belong to the same gate but are in different cores: '
          f'{alloc_slice[gate[0]]} and {alloc_slice[gate[1]]}'
        ))


def core_allocs_to_qubit_allocs(allocations: torch.Tensor,
                            core_capacities: Tuple[int, ...]
                          ) -> torch.Tensor:
  ''' Given a core allocation of the logical qubits, returns a plausible mapping to physical qubits.

  In the allocations tensor, each logical qubit is assigned a physical core. However, for some
  purposes (such as drawing) it is also useful to have the physical qubits that the logical qubits
  map to, not only the cores. This function does not consider core topology or any other issue
  alike.

  Args:
    - allocations: matrix of shape [Q,T] where Q = number of logical qubits, T = number of time slices.
  '''
  first_pq_in_core = [0] + [sum(core_capacities[:i]).item() for i in range(1,len(core_capacities))]
  physical_qubit_allocations = torch.zeros_like(allocations)
  for t_slice_i in range(allocations.shape[0]):
    first_free_pq_in_core = copy(first_pq_in_core)
    for lq_i in range(allocations.shape[1]):
      physical_core = allocations[t_slice_i, lq_i]
      physical_qubit_allocations[t_slice_i, lq_i] = first_free_pq_in_core[physical_core]
      first_free_pq_in_core[physical_core] += 1
  return physical_qubit_allocations


def swaps_from_alloc(allocations: torch.Tensor, n_cores: int) -> List[List[Tuple[int,int]]]:
  swaps = []
  for (prev_slice, next_slice) in zip(allocations[:-1], allocations[1:]):
    # Extract inter-core communications as edges of a graph
    adj_list = [[] for _ in range(n_cores)]
    core_count = [0]*n_cores
    for (q, (prev_core, next_core)) in enumerate(zip(prev_slice, next_slice)):
      if prev_core != next_core:
        core_count[prev_core] -= 1
        core_count[next_core] += 1
        com = (q, (prev_core, next_core))
        adj_list[prev_core].append(com)
    
    # Beware this might give a false positive if #logical qubits < #physical qubits
    assert all([cc == 0 for cc in core_count]), f"Some core is not balanced: {core_count}"

    is_cycle = lambda path: (path[0][1][0] == path[-1][1][1])
    cycle_to_swaps = lambda path: [(q0, q1) for ((q0, _), (q1, _)) in zip(path[:-1],path[1:])]
    paths_intersect = lambda p0, p1: bool(set(p0).intersection(set(p1)))

    # Find all cycles in the directed graph in ascending order by length
    swaps_slice = []
    paths = [(com,) for con_from_core in adj_list for com in con_from_core]
    while paths:
      path = paths.pop(0)
      # If path is a cycle, remove all paths that share edges with this one
      if is_cycle(path):
        swaps_slice += cycle_to_swaps(path)
        paths = list(filter(lambda p: (not paths_intersect(path, p)), paths))
        # Remove all edges in the cycle form the adj_list
        for com in path:
          l = adj_list[com[1][0]]
          del l[l.index(com)]
      # If it's not a cycle add all possible paths that follow by adding edges
      else:
        final_core = path[-1][1][1]
        for com in adj_list[final_core]:
          paths.append(path + (com,))
    swaps.append(swaps_slice)
  return swaps


def count_swaps(swaps: List[List[Tuple[int,int]]]) -> int:
  return sum(len(l) for l in swaps)


def check_sanity_swap(allocations: torch.Tensor, swaps: List[List[Tuple[int,int]]]):
  for slice_i, (prev_slice, next_slice) in enumerate(zip(allocations[:-1], allocations[1:])):
    qubit_cores = prev_slice.clone()
    for (q0, q1) in swaps[slice_i]:
      qubit_cores[[q0,q1]] = qubit_cores[[q1,q0]]
    assert torch.equal(qubit_cores, next_slice), (
      f"Target and result do not coincide after swapping for slices {slice_i} - {slice_i + 1}: "
      f"{qubit_cores.tolist()} vs target {next_slice.tolist()}"
    )