from typing import Tuple, Union

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from env_embeddings import env_init_embedding
from nn.transformer import Normalization, TransformerBlock


class PARCOEncoder(nn.Module):
    def __init__(
        self,
        env_name: str = "hcvrp",
        num_heads: int = 8,
        embed_dim: int = 128,
        num_layers: int = 3,
        normalization: str = "instance",
        use_final_norm: bool = False,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = {},
        norm_after: bool = False,
        use_pos_token: bool = False,
        trainable_pos_token: bool = True,
        **transformer_kwargs,
    ):
        super(PARCOEncoder, self).__init__()

        self.env_name = env_name
        init_embedding_kwargs["embed_dim"] = embed_dim
        self.init_embedding = (
            init_embedding
            if init_embedding is not None
            else env_init_embedding(self.env_name, init_embedding_kwargs)
        )

        self.layers = nn.Sequential(
            *(
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    normalization=normalization,
                    norm_after=norm_after,
                    **transformer_kwargs,
                )
                for _ in range(num_layers)
            )
        )

        if use_pos_token and trainable_pos_token:
            self.pos_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        elif use_pos_token:
            self.pos_token = torch.zeros(1, 1, embed_dim)
        else:
            self.pos_token = None
        self.use_pos_token = use_pos_token
        self.norm = Normalization(embed_dim, normalization) if use_final_norm else None

    def forward(
        self, td: TensorDict, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:
        # Transfer to embedding space
        init_h = self.init_embedding(td)  # [B, N, H]

        if self.use_pos_token:
            # Add a POS (pause-of-sequence) action to the embeddings
            # [B, N, H] -> [B, N+1, H]
            pos_token = self.pos_token.expand(init_h.size(0), 1, -1).to(td.device)
            init_h = torch.cat([init_h, pos_token], dim=1)

        # Process embedding
        h = init_h
        for layer in self.layers:
            h = layer(h, mask)

        # https://github.com/meta-llama/llama/blob/8fac8befd776bc03242fe7bc2236cdb41b6c609c/llama/model.py#L493
        if self.norm is not None:
            h = self.norm(h)

        # Return latent representation and initial embedding
        # [B, N, H]
        return h, init_h


