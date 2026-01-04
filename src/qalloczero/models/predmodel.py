from typing import Tuple
import torch



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
    embed_size: int,
    num_heads: int,
    num_layers: int,
  ):
    super().__init__()
    self.h = 10 # Size of the feature space (number of features)
    self.ff_up = torch.nn.Sequential(
     torch.nn.Linear(self.h, embed_size//2),
     torch.nn.ReLU(),
     torch.nn.Linear(embed_size//2, embed_size),
     torch.nn.ReLU(),
     torch.nn.LayerNorm(embed_size),
    )
    self.ff_proj_q = torch.nn.Sequential(
     torch.nn.Linear(2*self.h, embed_size//2),
     torch.nn.ReLU(),
     torch.nn.Linear(embed_size//2, embed_size),
     torch.nn.LayerNorm(embed_size),
    )
    self.ff_comb = torch.nn.Sequential(
     torch.nn.Linear(2*embed_size, embed_size),
     torch.nn.ReLU(),
     torch.nn.Linear(embed_size, embed_size),
     torch.nn.LayerNorm(embed_size),
    )
    encoder_layer = torch.nn.TransformerEncoderLayer(
      d_model=embed_size,
      nhead=num_heads,
      dim_feedforward=embed_size,
      batch_first=True
    )
    self.key_transf = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.output_logits_ = False
  

  def set_dropout(self, p: float):
    for module in self.modules():
        # 1. Update standard Dropout layers (in Sequential & Transformers)
        if isinstance(module, torch.nn.Dropout):
            module.p = p
        # 2. Update MultiheadAttention layers (attribute based)
        if isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = p


  def show_params(self):
    total_params = sum(p.numel() for p in self.parameters())
    total_train_par = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print(f"* {total_params} params, {total_train_par} trainable ({100*total_train_par/total_params}%)")
    for name, param in self.named_parameters():
      print(f"Name: {name}{' (trainable)' if param.requires_grad else ''}, Shape: {param.shape}, {param.numel()} elements.")


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
  

  def _get_core_caps(
    self,
    core_capacities: torch.Tensor,
    Q: int,
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
  ) -> torch.Tensor:
    has_prev_core = (prev_core_allocs != 0).any(dim=-1).any(dim=-1)
    swap_cost = torch.zeros_like(core_capacities, dtype=torch.float) # [B,C]
    prev_core_num = torch.argmax(prev_core_allocs, dim=1)
    prev_cores = prev_core_num[has_prev_core,qubits[has_prev_core,0]] # [B]
    swap_cost[has_prev_core] = core_connectivity[prev_cores] # [B,C]
    # For double qubit allocs, compute the cost of allocation of the second
    double_qubits = (qubits[:,1] != -1) & has_prev_core
    prev_cores = prev_core_num[double_qubits,qubits[double_qubits,1].flatten()]
    swap_cost[double_qubits] += core_connectivity[prev_cores,:]
    swap_cost = 1/(swap_cost + 1)
    swap_cost = swap_cost.unsqueeze(-1).expand(-1,-1,Q) # [B,C,Q]
    return swap_cost


  def _get_core_attraction(
    self,
    circuit_embs: torch.Tensor,
    prev_core: torch.Tensor,
  ):
    return torch.bmm(prev_core, circuit_embs)


  def _format_circuit_data(
    self,
    circuit_data: torch.Tensor,
    qubits: torch.Tensor,
    C: int
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    (B,Q,Q) = circuit_data.shape
    info_q0 = circuit_data[torch.arange(B), qubits[:,0]] # [B,C,Q]
    info_q1 = torch.zeros_like(info_q0)                   # [B,C,Q]
    double_qubits = (qubits[:,1] != -1)
    if double_qubits.any():
      info_q1[double_qubits] = circuit_data[double_qubits, qubits[double_qubits,1]]
    info_q0 = info_q0.unsqueeze(1).expand(-1,C,-1)
    info_q1 = info_q1.unsqueeze(1).expand(-1,C,-1)
    return info_q0, info_q1


  def _format_input(
    self,
    qubits: torch.Tensor,
    prev_core_allocs: torch.Tensor,
    current_core_allocs: torch.Tensor,
    core_capacities: torch.Tensor,
    core_connectivity: torch.Tensor,
    circuit_emb: torch.Tensor,
    next_interactions: torch.Tensor,
  ) -> torch.Tensor:
    (B,C) = core_capacities.shape
    (B,C,Q) = current_core_allocs.shape
    device = qubits.device
    qubit_matrix = self._get_qs(qubits, C, Q, device)
    core_caps = self._get_core_caps(core_capacities, Q)
    core_cost = self._get_core_cost(qubits, prev_core_allocs, core_capacities, core_connectivity, Q)
    core_attraction = self._get_core_attraction(circuit_emb, prev_core_allocs)
    ce_q0, ce_q1 = self._format_circuit_data(circuit_emb, qubits, C)
    ni_q0, ni_q1 = self._format_circuit_data(next_interactions, qubits, C)
    return torch.cat([
      qubit_matrix.unsqueeze(-1),
      prev_core_allocs.unsqueeze(-1),
      current_core_allocs.unsqueeze(-1),
      core_caps.unsqueeze(-1),
      core_cost.unsqueeze(-1),
      core_attraction.unsqueeze(-1),
      ce_q0.unsqueeze(-1),
      ce_q1.unsqueeze(-1),
      ni_q0.unsqueeze(-1),
      ni_q1.unsqueeze(-1),
    ], dim=-1) # [B,C,Q,h]


  def _extract_qubit_inputs(self, idx: torch.Tensor, inputs: torch.Tensor, C: int) -> torch.Tensor:
    # I'm really sorry for this function, the tensor manipulation is extremely toxic but it's what
    # it takes to do this
    input_size = inputs.shape[-1]
    ix_per_core = idx.unsqueeze(-1).expand(-1,C)
    idx_expanded = ix_per_core.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, input_size) # [B, C, 1, h]
    result = torch.gather(inputs, dim=2, index=idx_expanded.type(torch.long))  # [B, C, 1, h]
    return result.squeeze(2)  # [B, C, h]


  def _get_embeddings(self, inputs, qubits) -> Tuple[torch.Tensor, torch.Tensor]:
    (B,C,Q,_) = inputs.shape

    key_embs = self.ff_up(
      inputs.reshape(B*C*Q,self.h) # [B*C*Q,h]
    ).reshape(B,C,Q,-1) # [B,C,Q,H]

    q0_inputs = self._extract_qubit_inputs(qubits[:,0], inputs, C) # [B,C,h]

    double_q = (qubits[:,1] != -1)
    b = double_q.sum()
    q1_inputs = torch.zeros_like(q0_inputs)
    if b != 0:
      q1_inputs[double_q] = self._extract_qubit_inputs(qubits[double_q,1], inputs[double_q], C) # [b,C,h]
    q_inputs = torch.cat([q0_inputs, q1_inputs], dim=-1)
    proj_embs = self.ff_proj_q(
      q_inputs.reshape(B*C,2*self.h) # [B*C,h]
    ).reshape(B,C,-1) # [B,C,H]
    return key_embs, proj_embs
  

  def _project(
    self,
    key_embs: torch.Tensor,
    proj_embs: torch.Tensor,
  ) -> torch.Tensor:
    (B,C,Q,H) = key_embs.shape
    mixed_key_embs = self.key_transf(key_embs.reshape(B*C,Q,H))
    joined_key_embs = self.ff_comb(torch.cat([mixed_key_embs.reshape(B*C*Q,H), key_embs.reshape(B*C*Q,H)], dim=-1))
    compatibilities = torch.bmm(
      joined_key_embs.reshape(B*C,Q,H),
      proj_embs.reshape(B*C,H,1)
    ).reshape(B,C,Q) / torch.sqrt(torch.tensor(H))
    return compatibilities.mean(dim=-1) # [B,C]


  def forward(
      self,
      qubits: torch.Tensor,
      prev_core_allocs: torch.Tensor,
      current_core_allocs: torch.Tensor,
      core_capacities: torch.Tensor,
      core_connectivity: torch.Tensor,
      circuit_emb: torch.Tensor,
      next_interactions: torch.Tensor,
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
      qubits,
      prev_core_allocs,
      current_core_allocs,
      core_capacities,
      core_connectivity,
      circuit_emb,
      next_interactions,
    )
    key_embs, proj_embs = self._get_embeddings(inputs, qubits)
    logits = self._project(key_embs, proj_embs) # [B,C]
    vals = torch.tensor([[0.4] for _ in range(qubits.shape[0])], device=qubits.device) # Placeholder
    log_probs = torch.log_softmax(logits, dim=-1) # [B,C]
    if self.output_logits_:
      return logits, vals, log_probs
    probs = torch.softmax(logits, dim=-1) # [B,C]
    return probs, vals, log_probs
