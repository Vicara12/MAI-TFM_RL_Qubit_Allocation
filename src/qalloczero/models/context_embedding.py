from typing import Tuple
import torch
import torch.nn as nn

from .encoder import CoreFeatureEncoder, DynamicAgentGrouper
from .nn.transformer import (
    TransformerBlock as CommunicationLayer, 
    Normalization
)

class QAContextEmbedding(nn.Module):
    """Context embeddings.

    Produces:
      - Agent (qubit) embeddings (slice-aware)
      - Core embeddings (snapshot + features)
      - Temporal/global tokens
    """
    def __init__(
        self,
        embed_dim: int,
        max_qubits: int,
        use_communication: bool = True,
        num_communication_layers: int = 1,
        use_final_norm: bool = False,
        distance_matrix: torch.Tensor = None,
        **communication_layer_kwargs,
    ):
        super().__init__()

        # Agents embeddings (both global and contextual) are precomputed in initial embedding
        # so we've already called those, now we just need to retrieve the index we need
        layer_kwargs = dict(communication_layer_kwargs)

        # self.grouper = DynamicAgentGrouper(
        #     max_qubits=max_qubits, 
        #     embed_dim=embed_dim, 
        #     distance_matrix=distance_matrix
        # )

        self.core_feature_enc = CoreFeatureEncoder(
            embed_dim=embed_dim,
        )
        
        self.use_communication = use_communication
        if self.use_communication:
            self.q_layers = nn.Sequential(
                *(
                    CommunicationLayer(embed_dim=embed_dim, **layer_kwargs)
                    for _ in range(num_communication_layers)
                )
            )
            self.c_layers = nn.Sequential(
                *(
                    CommunicationLayer(embed_dim=embed_dim, **layer_kwargs)
                    for _ in range(num_communication_layers)
                )
            )
        else:
            self.q_layers = nn.Identity()
            self.c_layers = nn.Identity()

        self.norm = (
            Normalization(embed_dim, layer_kwargs.get("normalization", "rms"))
            if use_final_norm
            else None
        )

        self.project_global = nn.Linear(2*embed_dim, embed_dim)

    def _agent_state_embedding(self, embeddings, global_embeddings=None, **kwargs):
        # Global embedding might not add new information
        #if embeddings.dim() == 4:
            #slice_idx = td["current_slice"]
            #agent_slice_embeds = gather_by_index(embeddings, slice_idx, dim=1)  # [B, Q, d]
        #else:
        agent_slice_embeds = embeddings  # already [B, Q, d]

        if global_embeddings is not None:
            agent_embds = torch.cat([agent_slice_embeds, global_embeddings], dim=-1)
            return self.project_global(agent_embds)

        return agent_slice_embeds
    
    def _agent_global_embedding(self, embeddings, **kwargs):
        raise NotImplementedError #TODO: See if we can avoid using this

    def forward(
            self, 
            slice_embds: torch.Tensor,
            prev_core_allocs: torch.Tensor,
            current_core_allocs: torch.Tensor,
            core_capacities: torch.Tensor,
            core_size: torch.Tensor,
            core_connectivity: torch.Tensor = None,
            core_mask: torch.Tensor = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Gather the agent embeddings
        agent_embeds = self._agent_state_embedding(slice_embds)  # [B, Q, d] -> [B, Agents, C, d]

        # Agent embeddings
        grouped_agent_emds, agent_mask, agent_demands, final_action_mask = self.grouper(agent_embeds)  # [B, Q, d] -> [B, Agents, C, d]

        # Gather core embeddings (capacity)
        core_embeds = self.core_feature_enc(core_capacities, core_size=core_size, core_mask=core_mask)  # [B, C, d]

        agent_embeds = grouped_agent_emds + core_embeds.unsqueeze(1)  # [B, Agents, C, d]

        if self.use_communication:
            if agent_embeds.dim() == 4:
                B, Agents, C, d = agent_embeds.shape

                # [B, Agents, C, d] -> [B, C, Agents, d] -> [B*C, Agents, d]
                q_view = agent_embeds.permute(0, 2, 1, 3).reshape(B * C, Agents, d)
                # we must mask the padding agents so they don't participate!
                q_mask = agent_mask.repeat_interleave(C, dim=0) # [B, Agents] -> [B*C, Agents]
                padding_mask = ~q_mask
                for layer in self.q_layers:
                    q_view = layer(q_view, mask=padding_mask)
                # [B, C, Agents, d] -> [B, Agents, C, d]
                agent_embeds = q_view.view(B, C, Agents, d).permute(0, 2, 1, 3)

                # [B, Agents, C, d] -> [B*Agents, C, d]
                c_view = agent_embeds.reshape(B * Agents, C, d)
                c_view = self.c_layers(c_view)
                agent_embeds = c_view.view(B, Agents, C, d)

        if self.norm is not None:
            if agent_embeds.dim() == 4:
                B, Agents, C, d = agent_embeds.shape
                norm_view = agent_embeds.view(B, Agents * C, d)
                norm_view = self.norm(norm_view)
                agent_embeds = norm_view.view(B, Agents, C, d)
            else:
                agent_embeds = self.norm(agent_embeds)

        return agent_embeds, agent_mask, agent_demands, final_action_mask