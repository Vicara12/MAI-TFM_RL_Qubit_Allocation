from typing import Tuple, List
import torch



class EmbedModel(torch.nn.Module):
  def __init__(self, layer_sizes: List[int]):
    super().__init__()
    self.layers = []
    for (prev_sz, next_sz) in zip(layer_sizes[:-1], layer_sizes[1:]):
      self.layers.append(torch.nn.Linear(prev_sz, next_sz))
      self.layers.append(torch.nn.ReLU())
    self.layers.append(torch.nn.LayerNorm(layer_sizes[-1]))
    self.layers = torch.nn.Sequential(*self.layers)

  def forward(self, x):
    return self.layers(x)



class PredictionModel(torch.nn.Module):
  ''' For each qubit and time slice, output core allocation probability density and value of state.

  This model is used to determine the probabilities of allocating a qubit to each core in an
  specific time slice of the circuit and predicting the normalized value function of the current
  (input) state.

  The normalization of the value function is given by $V_norm = C/N_g$, where C is the cost of
  allocating what remains of the circuit (including the current allocation) and N_g is the number of
  two qubit gates in the remaining part of the circuit.

  Args:
    - n_qubits: Number of physical qubits among all cores.
    - n_cores: Number of cores.
    - number_emb_size: size of the embedding used to encode numbers (core capacity and swap cost).
    - glimpse_size: size of the glimpse (core embedding and output of the MHA).
    - alloc_ctx_emb_size: size of the allocation context embedding (which will be fed into MHA).
    - n_heads: number of heads used in the MHA mixing of core embeddings and allocation context.
  '''

  def __init__(
      self,
      layers: List[int],
  ):
    super().__init__()
    layers = [8] + layers
    self.ff_key = EmbedModel(layers)
    self.ff_query = EmbedModel(layers)
    self.output_logits_ = False


  def output_logits(self, value: bool):
    self.output_logits_ = value

  
  def _get_qs(self, qubits: torch.Tensor, C: int, Q: int, device: torch.device) -> torch.Tensor:
    B = qubits.shape[0]
    double_qubits = (qubits[:,1] != -1)
    qubit_matrix = torch.zeros((B,Q), dtype=torch.float, device=device) # [B,Q]
    # Encode qubits as one hot
    qubit_matrix[torch.arange(B),qubits[:,0]] = 1
    qubit_matrix[double_qubits, qubits[double_qubits,1]] = 1
    qubit_matrix = qubit_matrix.unsqueeze(1).expand(-1,C,-1) # [B,C,Q]
    return qubit_matrix
  

  def _get_prev_cores_one_hot(self, prev_core_allocs: torch.Tensor, C: int) -> torch.Tensor:
    # Detect batch items that do not have previous core (all items are set to n_cores)
    has_prev_core = (prev_core_allocs != C).any(dim=-1)
    # Transform prev_core_allocs of size [B,Q] with items in the range 0:C to a one hot version of
    # size [B,C,Q] with a 1 in position (b,c,q) if in prev_core_allocs[b,q] == c, 0 otherwise
    prev_c_allocs_sparse = torch.nn.functional.one_hot( # [B,Q,C+1]
      prev_core_allocs,
      num_classes=C+1,
    ).permute(0,2,1) # [B,C+1,Q]
    # Discard core C (no previous core assigned) and fill cores without previous assignment with 1
    prev_c_allocs_sparse = prev_c_allocs_sparse[:,:-1,:]
    prev_c_allocs_sparse[~has_prev_core,:,:] = 1
    return prev_c_allocs_sparse


  def _get_curr_cores_one_hot(self, current_core_allocs: torch.Tensor, C: int) -> torch.Tensor:
    curr_c_allocs_sparse = torch.nn.functional.one_hot( # [B,Q,C+1]
      current_core_allocs,
      num_classes=C+1,
    ).permute(0,2,1) # [B,C+1,Q]
    curr_c_allocs_sparse = curr_c_allocs_sparse[:,:-1,:] # [B,C,Q]
    return curr_c_allocs_sparse
  

  def _get_core_caps(
    self,
    core_capacities: torch.Tensor,
    Q: int,
    device: torch.device
  ) -> torch.Tensor:
    (B,C) = core_capacities.shape
    core_caps = 1/(core_capacities + 1)
    core_caps = core_caps.unsqueeze(-1).expand(-1,-1,Q) # [B,C,Q]
    return core_caps


  def _get_core_cost(
    self,
    qubits: torch.Tensor,
    prev_core_allocs: torch.Tensor,
    core_capacities: torch.Tensor,
    core_connectivity: torch.Tensor,
    Q: int,
    C: int,
  ) -> torch.Tensor:
    has_prev_core = (prev_core_allocs != C).any(dim=-1)
    swap_cost = torch.zeros_like(core_capacities, dtype=torch.float) # [B,C]
    prev_cores = prev_core_allocs[has_prev_core,qubits[has_prev_core,0]] # [B]
    swap_cost[has_prev_core] = core_connectivity[prev_cores] # [B,C]
    # For double qubit allocs, compute the cost of allocation of the second
    double_qubits = (qubits[:,1] != -1) & has_prev_core
    prev_cores = prev_core_allocs[double_qubits,qubits[double_qubits,1].flatten()]
    swap_cost[double_qubits] += core_connectivity[prev_cores,:]
    swap_cost = 1/(swap_cost + 1)
    swap_cost = swap_cost.unsqueeze(-1).expand(-1,-1,Q) # [B,C,Q]
    return swap_cost


  def _get_core_attraction(
    self,
    C: int,
    circuit_embs: torch.Tensor,
    prev_core: torch.Tensor,
  ):
    # Expand prev_core for comparison: (B, 1, Q)
    # Compare against all possible core indices (C,):
    core_ids = torch.arange(C, device=prev_core.device).view(1, C, 1)  # [1,C,1]
    prev_core_expanded = prev_core.unsqueeze(1)                        # [B,1,Q]
    # Create mask: True where prev_core[b, q'] == c
    mask = (prev_core_expanded == core_ids)                            # [B,C,Q]
    # Use mask to sum over qb dimension, expand circuit_embs for broadcasting: (B, 1, Q, Q)
    weighted = circuit_embs.unsqueeze(1) * mask.unsqueeze(-2)          # [B,C,Q,Q]
    affinities = weighted.sum(dim=-1)                                  # [B, C, Q]
    # Normalize
    orig_shape = affinities.shape
    affinities = affinities.reshape(prev_core.shape[0], -1)
    maximums = torch.max(affinities, dim=-1).values
    mask = (maximums != 0)
    affinities[mask] /= maximums[mask].unsqueeze(-1)
    return affinities.reshape(orig_shape)


  def _format_circuit_emb(
    self,
    circuit_emb: torch.Tensor,
    qubits: torch.Tensor,
    C: int
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    (B,Q,Q) = circuit_emb.shape
    ce_q0 = circuit_emb[torch.arange(B), qubits[:,0]] # [B,C,Q]
    ce_q1 = torch.zeros_like(ce_q0)                   # [B,C,Q]
    double_qubits = (qubits[:,1] != -1)
    if double_qubits.any():
      ce_q1[double_qubits] = circuit_emb[double_qubits, qubits[double_qubits,1]]
    ce_q0 = ce_q0.unsqueeze(1).expand(-1,C,-1)
    ce_q1 = ce_q1.unsqueeze(1).expand(-1,C,-1)
    return ce_q0, ce_q1


  def _format_input(
    self,
    qubits: torch.Tensor,
    prev_core_allocs: torch.Tensor,
    current_core_allocs: torch.Tensor,
    core_capacities: torch.Tensor,
    core_connectivity: torch.Tensor,
    circuit_emb: torch.Tensor,
  ) -> torch.Tensor:
    (B,C) = core_capacities.shape
    (B,Q) = current_core_allocs.shape
    device = qubits.device
    qubit_matrix = self._get_qs(qubits, C, Q, device)
    prev_core = self._get_prev_cores_one_hot(prev_core_allocs, C)
    curr_core = self._get_curr_cores_one_hot(current_core_allocs, C)
    core_caps = self._get_core_caps(core_capacities, Q, device)
    core_cost = self._get_core_cost(
      qubits, prev_core_allocs, core_capacities, core_connectivity, Q, C)
    core_attraction = self._get_core_attraction(C, circuit_emb, prev_core_allocs)
    ce_q0, ce_q1 = self._format_circuit_emb(circuit_emb, qubits, C)
    return torch.cat([
      qubit_matrix.unsqueeze(-1),
      prev_core.unsqueeze(-1),
      curr_core.unsqueeze(-1),
      core_caps.unsqueeze(-1),
      core_cost.unsqueeze(-1),
      core_attraction.unsqueeze(-1),
      ce_q0.unsqueeze(-1),
      ce_q1.unsqueeze(-1),
    ], dim=-1) # [B,C,Q,8]


  def _extract_qubit_inputs(self, idx: torch.Tensor, inputs: torch.Tensor, C: int) -> torch.Tensor:
    # I'm really sorry for this function, the tensor manipulation is extremely toxic but it's what
    # it takes to do this
    input_size = inputs.shape[-1]
    ix_per_core = idx.unsqueeze(-1).expand(-1,C)
    idx_expanded = ix_per_core.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, input_size) # [B, C, 1, 8]
    result = torch.gather(inputs, dim=2, index=idx_expanded.type(torch.long))  # [B, C, 1, 8]
    return result.squeeze(2)  # [B, C, 8]


  def _get_embeddings(self, inputs, qubits) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (B,C,Q,_) = inputs.shape

    key_embs = self.ff_key(
      inputs.reshape(-1,8) # [B*C*Q,8]
    ).reshape(B,C,Q,-1) # [B,C,Q,H]

    q0_inputs = self._extract_qubit_inputs(qubits[:,0], inputs, C) # [B,C,8]
    q0_embs = self.ff_query(
      q0_inputs.reshape(-1,8) # [B*C,8]
    ).reshape(B,C,-1) # [B,C,H]

    double_q = (qubits[:,1] != -1)
    b = double_q.sum()
    q1_embs = torch.zeros_like(q0_embs)
    if b != 0:
      q1_inputs = self._extract_qubit_inputs(qubits[double_q,1], inputs[double_q], C) # [b,C,8]
      q1_embs[double_q] = self.ff_key(
        q1_inputs.reshape(-1,8) # [b*C,8]
      ).reshape(b,C,-1) # [b,C,H]

    return key_embs, q0_embs, q1_embs
  
  def _project(
    self,
    key_embs: torch.Tensor,
    q0_embs: torch.Tensor,
    q1_embs: torch.Tensor
  ) -> torch.Tensor:
    (B,C,Q,H) = key_embs.shape
    projs_q0 = torch.bmm(key_embs.reshape(B*C,Q,H), q0_embs.reshape(B*C,H,1)).reshape(B,C,Q) # [B,C,Q]
    projs_q1 = torch.bmm(key_embs.reshape(B*C,Q,H), q1_embs.reshape(B*C,H,1)).reshape(B,C,Q) # [B,C,Q]
    return (projs_q0 + projs_q1) / torch.sqrt(torch.tensor(H))


  def forward(
      self,
      qubits: torch.Tensor,
      prev_core_allocs: torch.Tensor,
      current_core_allocs: torch.Tensor,
      core_capacities: torch.Tensor,
      core_connectivity: torch.Tensor,
      circuit_emb: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ''' Get the per-core allocation probability and normalized value function for the current state.

    Args:
      - qubits [B,2]: For a batch of size B, tensor with the qubit(s) to allocate. If the allocation
        corresponds to a gate, then the two positions should be filled with the qubits of the gate.
        For a single qubit allocation the second position should contain a -1.
      - prev_core_allocs [B,Q]: For a batch of size B, contains a vector of
        size Q where the position i contains the core that the ith logical qubit was allocated to it
        in the previous time slice. If there is no previous core allocation (first slice) then the
        corresponding qubit should have core value -1.
      - current_core_allocs [B,Q]: Equivalent to prev_core_allocs, but contains the allocations
        performed in the current time slice. Positions with -1 indicate the given qubit has not been
        allocated yet.
      - core_capacities [B,C]: For a batch of size B, contains a vector of size C with the number of
        qubits that can still be allocated in each core.
      - core_connectivity [C,C]: A symmetric matrix where item (i,j) indicates the cost of swapping
        a qubit from core i to core j or vice versa. Assumed to be the same for all elements in the
        batch.
      - circuit_emb [B,Q,Q]: For a batch of size B, contains a QxQ matrix which corresponds to the
        circuit embedding from the current slice until the end of the circuit.

    Returns:
      - [B,C]: For a batch of size B, a vector where each element corresponds to the probability of
        allocating the given qubit(s) to that core.
      - [B]: For a batch of size B, a scalar that corresponds to the expected normalized value cost
        of the allocating the input state.
    '''
    inputs = self._format_input(
      qubits, prev_core_allocs, current_core_allocs, core_capacities, core_connectivity, circuit_emb)
    key_embs, q0_embs, q1_embs = self._get_embeddings(inputs, qubits)
    projs = self._project(key_embs, q0_embs, q1_embs) # [B,C,Q]
    logits = projs.sum(dim=-1) # [B,C]
    vals = torch.tensor([[1.3] for _ in range(qubits.shape[0])], device=qubits.device) # Placeholder
    log_probs = torch.log_softmax(logits, dim=-1) # [B,C]
    if self.output_logits_:
      return logits, vals, log_probs
    probs = torch.softmax(logits, dim=-1) # [B,C]
    return probs, vals, log_probs
