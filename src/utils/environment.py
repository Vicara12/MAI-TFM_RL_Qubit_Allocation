from typing import Optional, Tuple
import torch
from .customtypes import Circuit, Hardware



class QubitAllocationEnvironment:
  """This is a flexible environment class for qubit allocation.
  It can be reset with a new circuit and hardware"""
  def __init__(self, circuit: Optional[Circuit] = None, hardware: Optional[Hardware] = None,
               validate_solution: bool = False, auto_reset: bool = True):
    self.circuit = circuit
    self.hardware = hardware
    self.validate_solution = validate_solution
    self.allocations = None
    self.current_core_caps = None
    self.current_slice_ = 0
    self.stage = 0
    self.unallocated_qubits = None
    self.current_allocation = None
    self.current_assignment = None
    if auto_reset and circuit is not None and hardware is not None:
      self.reset(circuit, hardware)
  

  def reset(self, circuit: Optional[Circuit] = None, hardware: Optional[Hardware] = None):
    if circuit is not None:
      self.circuit = circuit
    if hardware is not None:
      self.hardware = hardware
    if self.circuit is None or self.hardware is None:
      raise ValueError("Circuit and hardware must be set before resetting the environment")
    self.allocations = torch.full(
      size=(self.circuit.n_slices, self.circuit.n_qubits),
      fill_value=self.hardware.n_cores,
      dtype=int,
      device=self.hardware.core_capacities.device,
    )

    self.current_core_caps = torch.empty(
      self.hardware.n_cores + 1,
      dtype=self.hardware.core_capacities.dtype,
      device=self.hardware.core_capacities.device,
    )
    self.current_core_caps[:-1] = self.hardware.core_capacities
    self.current_core_caps[-1] = self.circuit.n_qubits
    self.current_slice_ = 0
    self.stage = 0
    self.unallocated_qubits = torch.ones(self.circuit.n_qubits, dtype=torch.bool)
    self.current_allocation = self.allocations[self.current_slice_]
  

  def allocate(self, cores: torch.Tensor) -> int:
    ''' 
    Allocates qubits each timeslice given a vector of actions, one for each qubit.
    This is the multi-agent approach. 
    Contains several asserts to ensure the validity of the solution.
    Also contains masking. 
    '''
    # Capacities are recomputed at each decoding step (successive steps within a slice!)
    # based on the current allocation
    core_fill = torch.bincount(cores, minlength=self.hardware.n_cores)
    self.current_core_caps[:-1] = self.hardware.core_capacities - core_fill
    self.current_core_caps[-1] = self.circuit.n_qubits
    
    if self.validate_solution:
      assert self.current_slice_ < self.circuit.n_slices, "Tried to allocate past the end of the circuit"
      assert all(cores >= 0) and all(cores <= self.hardware.n_cores), \
        f"Tried to allocate to core not in [0,{self.hardware.n_cores-1}] or the buffer action {self.hardware.n_cores}"
      assert len(cores) == self.circuit.n_qubits, \
        f"Expected {self.circuit.n_qubits} actions, got {len(cores)}"
      #assert not self.qubit_is_allocated[qubit], f"Tried to allocate qubit {qubit} twice"
      #assert self.hardware.core_capacities[core] > 0, f"Tried to allocate to complete core {core}"
      assert all(self.current_core_caps > 0), \
        f"Capacity violation(s) in core(s) with index(es) {torch.where(self.current_core_caps <= 0)[0]}"

    self.unallocated_qubits = cores == self.hardware.n_cores
    self.current_assignment = cores
    self._advance_stage() # The stage flag determines masking

    # If finished allocation of time slice
    if self.unallocated_qubits.sum().item() == 0:
      if self.validate_solution:
        # Check all gates have their qubits in the same core
        for gate in self.circuit.slice_gates[self.current_slice_]:
          assert self.allocations[self.current_slice_,gate[0]] == self.allocations[self.current_slice_,gate[1]], \
            (f"In time slice {self.current_slice_} allocated qubit {gate[0]} and {gate[0]} to cores "
            f"{self.allocations[self.current_slice_, gate[0]]} and "
            f"{self.allocations[self.current_slice_, gate[1]]}, but they belong to the same gate")
          
      # Store the allocation
      self.allocations[self.current_slice_] = cores
      # Compute the reward
      alloc_cost = self._get_reward(cores)
        
      self.stage = 0
      self.current_slice_ += 1

    return alloc_cost 
  

  def _advance_stage(self):
    """Multi-stage allocation. First we allocate pair qubits, then the rest. This function checks whether
    all pairs have been allocated. If so, it advances to the next stage. """
    #new_alloc_mask = (self.current_allocation != cores) & (cores != self.hardware.n_cores)  # Mask of newly allocated qubits
    new_alloc_mask = (self.current_assignment != self.hardware.n_cores)  # Mask of allocated qubits in current step 
    pair_q_indices = self.pair_indices.reshape(-1)
    if pair_q_indices.numel() == 0:
      return
    if torch.all(new_alloc_mask[pair_q_indices]):
      self.stage = 1


  def _get_reward(self, cores) -> int:
    """Reward is negative of the cost, so that the agent learns to minimize cost."""
    if self.current_slice_ == 0:
      alloc_cost = 0
    else:
      prev_core = self.allocations[self.current_slice_-1,]
      alloc_cost = self.hardware.core_connectivity[prev_core,cores].sum().item()
    return alloc_cost
  

  def get_mask(self) -> torch.Tensor:
    """Returns a tensor of shape [n_qubits, n_cores+1] with True for valid actions and False for invalid actions. 
    This can be used for action masking in the policy."""
    
    mask = torch.ones((self.circuit.n_qubits, self.hardware.n_cores+1), dtype=torch.bool)
    pair_q_indices = self.pair_indices.reshape(-1)
    is_pair = torch.zeros(self.circuit.n_qubits, dtype=torch.bool)
    if pair_q_indices.numel() > 0:
      is_pair[pair_q_indices] = True

    # First, mask out the buffer action for pairs
    mask[is_pair, self.hardware.n_cores] = False

    # Mask out all actions which are not the buffer core in single qubits if stage = 0
    if self.stage == 0:
      mask[~is_pair, :self.hardware.n_cores] = False

    # Mask out all cores that have capacity < 1 for single qubits and < 2 for pair qubits
    mask[:, self.current_core_caps < 1] = False
    pair_cap_mask = self.current_core_caps < 2
    if is_pair.any() and (pair_cap_mask).any():
      mask[is_pair.unsqueeze(1), pair_cap_mask] = False

    # Once a qubit is allocated, it cannot be allocated again: mask out all cores for that qubit except their allocation
    # Actively unmask actions that were masked for capacity reasons for allocated qubits
    mask[~self.unallocated_qubits] = False
    mask[~self.unallocated_qubits, self.current_assignment[~self.unallocated_qubits]] = True
    return mask


  @property
  def pair_indices(self) -> torch.Tensor:
    """Returns a tensor of shape [num_pairs, 2] with the indices of the qubits that belong to the same gate in the current slice. 
    This can be used for pairwise allocation."""
    # TODO: This being called every time is too much. Perhaps we should store the pair indices
    gates = self.circuit.slice_gates[self.current_slice_]
    if len(gates) == 0:
      return torch.empty((0, 2), dtype=torch.int64, device=self.current_assignment.device)
    return torch.as_tensor(gates, dtype=torch.int64, device=self.current_assignment.device)
  
  @property
  def current_slice(self) -> int:
    return self.current_slice_

  @property
  def prev_slice_allocations(self) -> torch.Tensor:
    assert self.current_slice_ > 0, "No previous slice"
    return self.allocations[self.current_slice_-1,:].squeeze()
  
  @property
  def finished(self) -> bool:
    return self.current_slice_ == self.circuit.n_slices
  
  @property
  def qubit_allocations(self) -> torch.Tensor:
    assert self.finished, "Tried to get incomplete allocation list"
    return self.allocations
  

class MAQubitAllocationEnvironment(QubitAllocationEnvironment):
  """This wrapper environment class contains functions to map qubit-level to 
  agent-level representations and vice versa, for the multi-agent approach."""
  def __init__(self, circuit: Optional[Circuit] = None, hardware: Optional[Hardware] = None,
               validate_solution: bool = False, auto_reset: bool = True):
    super().__init__(circuit, hardware, validate_solution, auto_reset)
    self._agent_mapping_slice = 0 # This is just a flag indicating when to cache the agent mapping


  def map_qubit_to_agent(self, tensor: torch.Tensor, reducer: str = 'mean') -> Tuple[torch.Tensor, int, int]:
    """Map qubit-dim -> padded agent-dim (mean/any/all over members)"""
    q_to_agent, num_agents, max_agents = self.agent_mapping
    q_dim = next(i for i, s in enumerate(tensor.shape) if s == self.circuit.n_qubits)
    idx = q_to_agent.view([1]*q_dim + [-1] + [1]*(tensor.ndim-q_dim-1))
    idx = idx.expand(*tensor.shape[:q_dim], -1, *tensor.shape[q_dim+1:])
    out_shape = list(tensor.shape)
    out_shape[q_dim] = max_agents

    if reducer in ('any', 'all'):
      # Use integer accumulation; scatter_add_ does not support bool destination.
      out = torch.zeros(out_shape, device=tensor.device, dtype=torch.int)
      out.scatter_add_(q_dim, idx, tensor.to(torch.int))
      counts = torch.bincount(q_to_agent, minlength=max_agents).to(tensor.device).clamp_min(1)
      shape = [1]*tensor.ndim
      shape[q_dim] = -1
      if reducer == 'any':
        out = out > 0
      else:  # all
        out = out == counts.view(shape)
      return out, num_agents, max_agents

    out = torch.zeros(out_shape, device=tensor.device, dtype=tensor.dtype)
    out.scatter_add_(q_dim, idx, tensor)
    if reducer == 'mean':
      counts = torch.bincount(q_to_agent, minlength=max_agents).to(tensor.device).clamp_min(1)
      shape = [1]*tensor.ndim
      shape[q_dim] = -1
      out = out / counts.view(shape)
    return out, num_agents, max_agents
  
  
  def map_agent_to_qubit(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Map agent-dim -> qubit-dim (broadcast)"""
    q_to_agent, num_agents, max_agents = self.agent_mapping
    a_dim = next(i for i, s in enumerate(tensor.shape) if s == max_agents)
    idx = q_to_agent.view([1]*a_dim + [-1] + [1]*(tensor.ndim-a_dim-1))
    idx = idx.expand(*tensor.shape[:a_dim], -1, *tensor.shape[a_dim+1:])
    return torch.gather(tensor, a_dim, idx), num_agents, max_agents
  
  
  def _agent_mapping(self) -> tuple[torch.Tensor, int, int]:
    """Returns (q_to_agent, num_agents, max_agents) for current slice."""
    
    n_q = self.circuit.n_qubits
    # Upper bound for padding; use n_q to cover the no-pair case cleanly.
    max_agents = n_q
    pairs = self.pair_indices  # [P,2] or [0,2]

    if pairs.numel() == 0:
      return torch.arange(n_q, dtype=torch.long), n_q, max_agents

    q_to_agent = torch.full((n_q,), -1, dtype=torch.long)
    p = pairs.shape[0]
    # assign pair agents
    q_to_agent.scatter_(0, pairs.reshape(-1), torch.arange(p, dtype=torch.long).repeat_interleave(2))

    # assign single agents
    singles = (q_to_agent == -1).nonzero(as_tuple=False).flatten()
    start = p
    q_to_agent.scatter_(0, singles, torch.arange(start, start + singles.numel(), dtype=torch.long))
    num_agents = start + singles.numel()
    return q_to_agent, num_agents, max_agents


  @property
  def agent_mapping(self) -> tuple[torch.Tensor, int, int]:
    if not hasattr(self, '_agent_mapping_cache') or self._agent_mapping_slice != self.current_slice_:
      self._agent_mapping_cache = self._agent_mapping()
      self._agent_mapping_slice = self.current_slice_
    return self._agent_mapping_cache



ENV_REGISTRY = {
  'qa': QubitAllocationEnvironment,
  'maqa': MAQubitAllocationEnvironment,
} 

# ############################# TESTING #############################

# import torch
# from src.sampler.randomcircuit import RandomCircuit


# def make_dummy_circuit(n_qubits: int, n_slices: int):
#   sampler = RandomCircuit(num_lq=n_qubits, num_slices=n_slices)
#   return sampler.sample()

# def make_dummy_hw(n_qubits: int, n_cores: int):
#   # Example: evenly distribute capacity, fully connected except self.
#   cap = torch.tensor([max(1, n_qubits // n_cores)] * n_cores, dtype=torch.int)
#   con = torch.ones((n_cores, n_cores), dtype=torch.float) - torch.eye(n_cores)
#   return Hardware(core_capacities=cap, core_connectivity=con)

# def main():
#   circuit = make_dummy_circuit(n_qubits=6, n_slices=3)
#   hardware = make_dummy_hw(n_qubits=6, n_cores=3)

#   env = MAQubitAllocationEnvironment(circuit, hardware)
#   print("After init:", env.current_slice, env.finished)

#   for t in range(circuit.n_slices):
#     cores = torch.randint(low=0, high=hardware.n_cores, size=(circuit.n_qubits,))
#     cost = env.allocate(cores)
#     print(f"Slice {t} cost={cost} current_slice={env.current_slice} finished={env.finished}")

#   print("Final allocations:", env.qubit_allocations)

#   new_circuit = make_dummy_circuit(n_qubits=4, n_slices=2)
#   new_hw = make_dummy_hw(n_qubits=4, n_cores=2)
#   env.reset(circuit=new_circuit, hardware=new_hw)
#   print("After reset:", env.current_slice, env.finished)

# if __name__ == "__main__":
#   main()