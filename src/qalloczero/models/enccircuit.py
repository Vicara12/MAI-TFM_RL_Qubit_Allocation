from typing import Dict, Tuple
import torch



class CircuitEncoder(torch.nn.Module):
  ''' Handles the codification of circuit slices via an Encoder-Decoder transformer.
  '''
  

  def __init__(self, n_qubits: int, n_heads: int, n_layers: int):
    super().__init__()

  def forward(self, adjacency_matrices: torch.Tensor) -> torch.Tensor:
    ''' Get circuit embedding for each slice.

    Arguments:
      - adjacency_matrices [B, N, Q, Q]: for a batch of size B, N adjacency matrices of size [Q,Q]
        corresponding to the N time slices in each circuit [s_0, s_1, ..., s_{n-1}].
    
    Returns:
      - [B, N, Q, Q]: a circuit embedding of size [Q, Q] for each of the N times slices per circuit
        for all circuits in the batch of size B.
    '''
    circuit_embs = torch.empty_like(adjacency_matrices, dtype=torch.float)
    circuit_embs[:,-1] = 0.5 * adjacency_matrices[:,-1]
    n_slices = adjacency_matrices.shape[1]
    for slice_i in range(n_slices-2,-1,-1):
      circuit_embs[:,slice_i] = 0.5 * (circuit_embs[:,slice_i + 1] + adjacency_matrices[:,slice_i])
    return circuit_embs