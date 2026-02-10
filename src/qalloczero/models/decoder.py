from typing import Optional, Tuple

import torch
import torch.nn as nn


class QADecoder(nn.Module):
    """Minimal decoder for qubit-to-core scoring.

    Expects precomputed agent embeddings and an action mask. It appends a learnable
    "buffer" embedding (no-core placement) and scores every agent-core option with
    a small MLP. Invalid actions are masked to ``-inf`` so callers can safely apply
    ``softmax``/``log_softmax`` without extra checks.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )
        self.buffer = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(
        self,
        agent_embeds: torch.Tensor,
        action_mask: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            agent_embeds: [B, Agents, C, H] agent-core features.
            action_mask: [B, Agents, C+1] boolean mask of valid actions (last column = buffer).
            agent_mask: [B, Agents] optional mask indicating which agents are real (False for padding).

        Returns:
            logits: [B, Agents, C+1] masked logits (``-inf`` for invalid actions).
            final_mask: [B, Agents, C+1] mask actually used for masking/logits.
        """
        if agent_embeds.dim() != 4:
            raise ValueError("agent_embeds must have shape [B, Agents, C, H]")

        B, Agents, C, H = agent_embeds.shape

        # Append learnable buffer embedding so each agent always has a fallback option.
        buffer_emb = self.buffer.view(1, 1, 1, H).expand(B, Agents, -1, -1)
        features = torch.cat((agent_embeds, buffer_emb), dim=2)  # [B, Agents, C+1, H]
        
        #TODO: I think it's safe to remove this check
        if action_mask.shape[-1] == C:
            # Tolerate missing buffer mask by assuming buffer is always allowed.
            buffer_mask = torch.ones(B, Agents, 1, dtype=action_mask.dtype, device=action_mask.device)
            final_mask = torch.cat((action_mask, buffer_mask), dim=-1)
        else:
            final_mask = action_mask.clone()

        #TODO: Check this. Does it actually do something?
        if agent_mask is not None:
            final_mask[..., -1] |= (~agent_mask)

        logits = self.score_head(features).squeeze(-1)
        logits = logits.masked_fill(~final_mask, float("-inf"))

        return logits, final_mask
        raw_embeddings = cached.node_embeddings if isinstance(cached, PrecomputedCache) else cached

