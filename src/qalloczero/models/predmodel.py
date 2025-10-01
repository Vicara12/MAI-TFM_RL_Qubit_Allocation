from typing import Tuple, Optional, Union
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
      n_qubits: int,
      n_cores: int,
      number_emb_size: int,
      n_heads: int,
  ):
    super().__init__()
    self.n_qubits = n_qubits
    self.n_cores = n_cores
    self.n_emb_size = number_emb_size

    # Used to convert numbers to vector embeddings. We use this instead of proper embeddings in
    # order to make it more flexible (no fixed number of bins)
    self.number_encoder = torch.nn.Sequential(
      torch.nn.Linear(1,number_emb_size//2),
      torch.nn.ReLU(),
      torch.nn.Linear(number_emb_size//2, number_emb_size),
      torch.nn.ReLU()
    )

    # The information of a core includes a one hot vector of the qubits it stored in the previous
    # time slice, as well as those stored in the current time slice, and two number embeddings, one
    # for the allocation cost and another for the core capacity
    self.qubit_allocation_joiner = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1),
      torch.nn.ReLU(),
      torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1),
    )
    self.core_encoder = torch.nn.Sequential(
      torch.nn.Linear(n_qubits + 2*number_emb_size, n_qubits),
      torch.nn.ReLU(),
      torch.nn.Linear(n_qubits, n_qubits),
      torch.nn.ReLU(),
    )

    # Encode circuit context into an embedding vector
    self.context_joiner = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1),
      torch.nn.ReLU(),
      torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1),
      torch.nn.ReLU(),
      torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1),
      torch.nn.ReLU(),
    )
    self.context_mha = torch.nn.MultiheadAttention(
      embed_dim=n_qubits,
      num_heads=n_heads,
      batch_first=True,
    )
    self.context_ff = torch.nn.Sequential(
      torch.nn.Linear(n_qubits, n_qubits),
      torch.nn.ReLU(),
      torch.nn.Linear(n_qubits, n_qubits),
      torch.nn.ReLU(),
    )

    # We use MHA to mix core information and allocation context information into a glimpse, which is
    # then projected back into the core embeddings to get the logits of allocation to each core
    self.mha = torch.nn.MultiheadAttention(
      embed_dim=n_qubits,
      num_heads=n_heads,
      batch_first=True,
    )

    self.value_network = torch.nn.Sequential(
      torch.nn.Linear(2*n_qubits, n_qubits),
      torch.nn.ReLU(),
      torch.nn.Linear(n_qubits, n_qubits),
      torch.nn.ReLU(),
      torch.nn.Linear(n_qubits, n_qubits),
      torch.nn.ReLU(),
      torch.nn.Linear(n_qubits, 1),
    )

  
  def _encode_cores(
      self,
      qubits: torch.Tensor,
      prev_core_allocs: torch.Tensor,
      current_core_allocs: torch.Tensor,
      core_capacities: torch.Tensor,
      core_connectivity: torch.Tensor
  ) -> torch.Tensor:
    ''' Computes the core embeddings (G) from a batch of cores.

    Arguments:
      Refer to the forward method.
    
    Returns:
      - [B,C,glimpse_size]: For each element in the batch of size B, a core embedding for each of
        the C cores of length [glimpse_size].
    '''
    # We will write to these two, so we make a copy to not modify the originals
    (B,C) = core_capacities.shape
    (_,Q) = prev_core_allocs.shape
    # Compute the cost of allocating the first qubit of the pair to each core
    has_prev_core = (prev_core_allocs != self.n_cores).any(dim=-1) # Check rows that only contain -1
    swap_cost = torch.zeros_like(core_capacities, dtype=torch.float) # [B,C]
    prev_cores = prev_core_allocs[has_prev_core,qubits[has_prev_core,0]] # [B]
    swap_cost[has_prev_core] = core_connectivity[prev_cores] # [B,C]
    # For double qubit allocs, compute the cost of allocation of the second
    double_qubits = (qubits[:,1] != -1) & has_prev_core
    prev_cores = prev_core_allocs[double_qubits,qubits[double_qubits,1].flatten()] # [b]
    swap_cost[double_qubits] += core_connectivity[prev_cores,:] # [b,C]
    # Encode scalars (allocation cost and core capacities) into embedding vectors
    # We will organize core info as [prev_core_allocs, current_core_allocs, swap_cost_emb, core_caps_emb],
    # so we put the two numerical values side to side so that we can then flatten the number vector
    # and compute it in batch, then we reshape it back into [swap_cost_emb, core_caps_emb]
    numerical_info_row = torch.cat([swap_cost, core_capacities], dim=-1).reshape(2*B*C,1)
    number_embeddings = self.number_encoder(numerical_info_row).reshape(B,C,2*self.n_emb_size)
    # Transform prev_core_allocs of size [B,Q] with items in the range 0:C-1 to a one hot version of
    # size [B,C,Q] with a 1 in position (b,c,q) if in prev_core_allocs[b,q] == c, 0 otherwise
    prev_c_allocs_sparse = torch.nn.functional.one_hot( # [B,Q,C+1]
      prev_core_allocs,
      num_classes=C+1,
    ).permute(0,2,1) # [B,C+1,Q]
    # Discard core C (-1) and fill cores without previous assignment with 1
    prev_c_allocs_sparse = prev_c_allocs_sparse[:,:-1,:]
    prev_c_allocs_sparse[~has_prev_core,:,:] = 1
    # Set qubit not allocated (-1) to a new core which we will remove later
    curr_c_allocs_sparse = torch.nn.functional.one_hot( # [B,Q,C+1]
      current_core_allocs,
      num_classes=C+1,
    ).permute(0,2,1) # [B,C+1,Q]
    curr_c_allocs_sparse = curr_c_allocs_sparse[:,:-1,:] # [B,C,Q]
    joined_alloc_info = self.qubit_allocation_joiner(
      torch.cat([prev_c_allocs_sparse.float().unsqueeze(1), curr_c_allocs_sparse.float().unsqueeze(1)], dim=1)
    ).squeeze(1)
    core_info = torch.cat([joined_alloc_info, number_embeddings], dim=-1
    ).reshape(B*C, Q + 2*self.n_emb_size) # Flatten into matrix to compute all in batch
    core_embs = self.core_encoder(core_info).reshape(B,C,Q) # [B,C,Q]
    return core_embs
    

  def _encode_circuit_context(
      self,
      circuit_emb: torch.Tensor,
      slice_adj_mat: torch.Tensor,
      qubits: torch.Tensor
  ) -> torch.Tensor:
    ''' Get an embedding that contains the information about the allocation context.

    Args:
      Refer to the forward method.
    
    Returns:
      - [B,alloc_ctx_emb_size]: For each item in the batch of size B, an embedding of length
        [alloc_ctx_emb_size] with the information of the rest of the circuit, the current slice
        and qubits to be allocated.
    '''
    (B,Q,_) = circuit_emb.shape
    double_qubits = (qubits[:,1] != -1)

    qubit_matrix = torch.zeros((B,Q), dtype=torch.float, device=slice_adj_mat.device)
    # Encode qubits as one hot
    qubit_matrix[torch.arange(B),qubits[:,0]] = 1
    qubit_matrix[double_qubits, qubits[double_qubits,1]] = 1

    joined_context = self.context_joiner(
      torch.cat([circuit_emb.unsqueeze(1), slice_adj_mat.unsqueeze(1)], dim=1)
    ).squeeze(1)
    alloc_ctx_emb, _ = self.context_mha(
      query=qubit_matrix.unsqueeze(1),
      key=joined_context,
      value=joined_context
    )
    alloc_ctx_emb = self.context_ff(alloc_ctx_emb)
    return alloc_ctx_emb


  def forward(
      self,
      qubits: torch.Tensor,
      prev_core_allocs: torch.Tensor,
      current_core_allocs: torch.Tensor,
      core_capacities: torch.Tensor,
      core_connectivity: torch.Tensor,
      circuit_emb: torch.Tensor,
      slice_adj_mat: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
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
      - slice_adj_mat [B,Q,Q]: For a batch of size B, contains a QxQ matrix with the one hot encoded
        adjacency matrix of the current slice being allocated.

    Returns:
      - [B,C]: For a batch of size B, a vector where each element corresponds to the probability of
        allocating the given qubit(s) to that core.
      - [B]: For a batch of size B, a scalar that corresponds to the expected normalized value cost
        of the allocating the input state.
    '''
    core_embs = self._encode_cores(
      qubits=qubits,
      prev_core_allocs=prev_core_allocs,
      current_core_allocs=current_core_allocs,
      core_capacities=core_capacities,
      core_connectivity=core_connectivity,
    )
    alloc_ctx_embs = self._encode_circuit_context(
      circuit_emb=circuit_emb,
      slice_adj_mat=slice_adj_mat,
      qubits=qubits,
    )
    glimpses, attn_weights = self.mha(
      query=alloc_ctx_embs,
      key=core_embs,
      value=core_embs,
      need_weights=True,
    )
    # This will never be false, but is needed to tell jitter that the tensor is not optional
    assert attn_weights is not None, "Attention weights returned None"
    value = self.value_network(torch.cat([glimpses, alloc_ctx_embs], dim=-1).squeeze(1))
    return attn_weights.squeeze(1), value