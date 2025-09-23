import torch
import torch.nn as nn

from models.nn.transformer import (
    Normalization,
    TransformerBlock as CommunicationLayer,
)

class QubitCommEmbedding(nn.Module):
    #TODO: might lack some arguments
    """Communication layer for Qubit Allocation
    Args:
        embed_dim: int, hidden dimension
        use_communication: bool, enable comm layers
        num_communication_layers: int, number of Transformer layers
        **communication_kwargs: passed to TransformerBlock
    """

    def __init__(
        self,
        embed_dim,
        use_communication: bool = True,
        use_final_norm: bool = False,
        num_communication_layers: int = 1,
        **communication_kwargs,
    ):
        super().__init__()
        if use_communication:
            self.communication_layers = nn.Sequential(
                *(
                    CommunicationLayer(
                        embed_dim=embed_dim,
                        **communication_kwargs,
                    )
                    for _ in range(num_communication_layers)
                )
            )
        else:
            self.communication_layers = nn.Identity()

        self.norm = (
            Normalization(embed_dim, communication_kwargs.get("normalization", "rms"))
            if use_final_norm
            else None
        )

    def forward(self, agent_embeddings, slice_token, global_token):
        """
        Args:
            agent_embeddings: [B, Q, d]
            slice_token: [B, d]
            global_token: [B, d]
        Returns:
            h_comm: [B, Q, d] (agent embeddings after comm)
        """
        B, Q, d = agent_embeddings.shape
        context_tokens = torch.stack([slice_token, global_token], dim=1)  # [B, 2, d]
        tokens = torch.cat([agent_embeddings, context_tokens], dim=1)     # [B, Q+2, d]

        h = self.communication_layers(tokens)  # [B, Q+2, d]
        h_agents = h[:, :Q]  # discard context outputs, return agent embds only

        if self.norm is not None:
            h_agents = self.norm(h_agents)
        return h_agents