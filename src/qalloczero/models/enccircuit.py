from typing import Dict, Tuple
import torch



class CircuitEncoder(torch.nn.Module):
  ''' Handles the codification of circuit slices via an Encoder-Decoder transformer.
  '''

  # Recurrently used values are cached for efficiency
  POS_EMBEDDINGS: Dict[Tuple[int, int], torch.Tensor] = {}
  TRANSFORMER_MASKS: Dict[int, torch.Tensor] = {}

  @staticmethod
  def comp_pos_emb(T: int, d_model: int) -> torch.Tensor:
    position = torch.arange(T).unsqueeze(1)  # [T, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe = torch.zeros(T, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    # Scale positional encoding to prevent it from overwhelming the one-hot input
    return pe*0.5  # [T, d_model]
  

  @staticmethod
  def get_pos_emb(T: int, d_model: int) -> torch.Tensor:
    request = (T, d_model)
    if request not in CircuitEncoder.POS_EMBEDDINGS.keys():
      CircuitEncoder.POS_EMBEDDINGS[request] = CircuitEncoder.comp_pos_emb(*request)
    return CircuitEncoder.POS_EMBEDDINGS[request]
  

  @staticmethod
  def get_transformer_mask(size: int) -> torch.Tensor:
    if size not in CircuitEncoder.TRANSFORMER_MASKS.keys():
      mask = torch.nn.Transformer.generate_square_subsequent_mask(size)
      CircuitEncoder.TRANSFORMER_MASKS[size] = mask
    return CircuitEncoder.TRANSFORMER_MASKS[size]
  

  def __init__(self, n_qubits: int, n_heads: int, n_layers: int):
    super().__init__()
    self.n_qubits = n_qubits
    self.transformer = torch.nn.Transformer(
      d_model=n_qubits**2,
      nhead=n_heads,
      num_encoder_layers=n_layers,
      num_decoder_layers=n_layers,
      batch_first=True
    )


  def encode_slice(self, adjacency_matrices: torch.Tensor, later_embeddings: torch.Tensor) -> torch.Tensor:
    ''' Get the circuit embedding for slice s_i.

    Arguments:
      - adjacency_matrices [B, S, Q*Q]: for a batch of size B, S flattened adjacency matrices of
        size [Q*Q] from s_i until the end of the circuit [s_i, s_{i+1}, s_{i+2}, ..., s_{n-1}].
      - later_embeddings [B, S, Q*Q]: embeddings produced in the S-1 previous encoding steps,
        followed by the identity matrix (as the "bootstrap embedding" for the sequence).
    Returns:
      - [B,Q*Q]: flattened circuit embedding of size [Q*Q] for the B elements in the batch.
    '''
    device = adjacency_matrices.device
    # We will flatten adjacency matrix into a vector to make it easier to handle in the code
    (B, S, Q2) = adjacency_matrices.shape
    pos_emb = CircuitEncoder.get_pos_emb(S, Q2).to(device).expand(B,S,Q2)
    in_seq = pos_emb + adjacency_matrices
    out_seq = pos_emb + later_embeddings
    mask = CircuitEncoder.get_transformer_mask(S).to(device)
    return self.transformer(in_seq, out_seq, tgt_mask=mask)[:,-1,:]
  

  def forward(self, adjacency_matrices: torch.Tensor) -> torch.Tensor:
    ''' Get circuit embedding for each slice.

    Arguments:
      - adjacency_matrices [B, N, Q, Q]: for a batch of size B, N adjacency matrices of size [Q,Q]
        corresponding to the N time slices in each circuit [s_0, s_1, ..., s_{n-1}].
    
    Returns:
      - [B, N, Q, Q]: a circuit embedding of size [Q, Q] for each of the N times slices per circuit
        for all circuits in the batch of size B.
    '''
    device = adjacency_matrices.device
    (B, N, Q, Q) = adjacency_matrices.shape
    adjacency_matrices = adjacency_matrices.reshape(B,N,Q*Q)
    # Add initial decoded slice to be identity (all gates only interact with themselves)
    last_slice = torch.eye(Q, device=device).expand(B,1,-1,-1).reshape(B,1,-1)
    circuit_embs = torch.concat([torch.empty_like(adjacency_matrices), last_slice], dim=1)
    # Iterate slices backwards (from the end of the circuit towards the beginning)
    for s_i in range(N-1,-1,-1):
      circuit_embs[:,s_i,:] = self.encode_slice(
        adjacency_matrices=adjacency_matrices[:,s_i:,:],
        later_embeddings=circuit_embs[:,(s_i+1):,:],
      )
    # Ignore last embedding, as it was only using for bootstrapping regression
    return circuit_embs[:,:-1,:].reshape(B,N,Q,Q)