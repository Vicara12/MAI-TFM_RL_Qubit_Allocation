import torch
import torch.nn as nn
from .communication import QubitCommEmbedding
from .core_embds import CoreFeatureEncoder, CoreSnapshotEncoder
#from .slice_embds import SliceEmbedding, TemporalGlobalContext, pool_slices, get_context
from .qubit_embds import QubitEmbedding, TemporalGlobalContext, get_context_tokens


from models.nn.transformer import TransformerBlock as CommunicationLayer


class QubitAllocInitEmbedding(nn.Module):
    """Initial embedding for Qubit Allocation env.

    Produces:
      - Agent (qubit) embeddings (slice-aware)
      - Core embeddings (snapshot + features)
      - Temporal/global tokens
    """

    def __init__(
        self,
        num_qubits: int,
        num_cores: int,
        core_size: int,
        embed_dim: int,
        adjacency_matrix: torch.Tensor,
        distance_matrix: torch.Tensor,
        pool_type: str = "mean",
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Agent side
        self.qubit_embeddings = QubitEmbedding(embed_dim, num_qubits, dropout=dropout)
        self.temporal_context = TemporalGlobalContext(
            hidden_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            pool_type=pool_type,
        )

        # Core side
        self.snapshot_enc = CoreSnapshotEncoder(
            num_qubits=num_qubits,
            num_cores=num_cores,
            core_size=core_size,
            hidden_dim=embed_dim,
            adjacency_matrix=adjacency_matrix,
        )
        self.feature_enc = CoreFeatureEncoder(
            core_size=core_size,
            distance_matrix=distance_matrix,
            embed_dim=embed_dim,
        )

    def forward(self, td, slice_idx: int):
        """
        Args:
            td: TensorDict with keys ['slices', 'last_assignment', 'current_core_capacity']
            slice_idx: int, index of current slice
        Returns:
            agent_embeds: [B, Q, d]
            core_embeds:  [B, C, d]
            slice_token:  [B, d]
            global_token: [B, d]
        """
        # Agent embeddings
        agent_slices = self.slice_embedding(td)  # [B, T, Q, d]
        agent_embeds = agent_slices[:, slice_idx]  # current slice [B, Q, d]

        # Temporal/global context
        slice_token, global_token = self.temporal_context(agent_slices, slice_idx)

        #Core embeddings
        snapshot_emb = self.snapshot_enc(td["last_assignment"])  # [B, C, d]
        feature_emb = self.feature_enc(td)                       # [B, C, d]
        core_embeds = snapshot_emb + feature_emb                 # [B, C, d]

        return agent_embeds, core_embeds, slice_token, global_token
    
    
# class QubitAllocInitEmbedding(nn.Module):
#     """Initial embedding for Qubit Allocation env.

#     Produces:
#       - Agent (qubit) embeddings (slice-aware)
#       - Core embeddings (snapshot + features)
#       - Temporal/global tokens
#     """

#     def __init__(
#         self,
#         num_qubits: int,
#         num_cores: int,
#         core_size: int,
#         embed_dim: int,
#         adjacency_matrix: torch.Tensor,
#         distance_matrix: torch.Tensor,
#         pool_type: str = "mean",
#         num_layers: int = 2,
#         num_heads: int = 4,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         # Agent side
#         self.slice_embedding = SliceEmbedding(embed_dim, num_qubits, dropout=dropout)
#         self.temporal_context = TemporalGlobalContext(
#             hidden_dim=embed_dim,
#             num_layers=num_layers,
#             num_heads=num_heads,
#             dropout=dropout,
#             pool_type=pool_type,
#         )

#         # Core side
#         self.snapshot_enc = CoreSnapshotEncoder(
#             num_qubits=num_qubits,
#             num_cores=num_cores,
#             core_size=core_size,
#             hidden_dim=embed_dim,
#             adjacency_matrix=adjacency_matrix,
#         )
#         self.feature_enc = CoreFeatureEncoder(
#             core_size=core_size,
#             distance_matrix=distance_matrix,
#             embed_dim=embed_dim,
#         )

#     def forward(self, td, slice_idx: int):
#         """
#         Args:
#             td: TensorDict with keys ['slices', 'last_assignment', 'current_core_capacity']
#             slice_idx: int, index of current slice
#         Returns:
#             agent_embeds: [B, Q, d]
#             core_embeds:  [B, C, d]
#             slice_token:  [B, d]
#             global_token: [B, d]
#         """
#         # --- Agent embeddings
#         agent_slices = self.slice_embedding(td)  # [B, T, Q, d]
#         agent_embeds = agent_slices[:, slice_idx]  # current slice [B, Q, d]

#         # Temporal/global context
#         slice_token, global_token = self.temporal_context(agent_slices, slice_idx)

#         # --- Core embeddings
#         snapshot_emb = self.snapshot_enc(td["last_assignment"])  # [B, C, d]
#         feature_emb = self.feature_enc(td)                       # [B, C, d]
#         core_embeds = snapshot_emb + feature_emb                 # [B, C, d]

#         return agent_embeds, core_embeds, slice_token, global_token
