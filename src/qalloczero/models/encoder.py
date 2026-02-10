from typing import Optional

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from .nn.positional_encoder import PositionalEncoder
from .nn.transformer import TransformerBlock


_ORTHOGONAL_IDS_CACHE = {}



class QubitEmbedding(nn.Module):
    """
    Embeds each qubit (agent) in each slice using a GNN.
    Output: [batch, num_slices, num_qubits, hidden_dim]
    """
    def __init__(self, embed_size, max_qubits, dropout=0.0, use_learnable_ids=False, use_temp_transformer=True):
        super().__init__()
        self.embed_size = embed_size
        self.max_qubits = max_qubits
        self.use_learnable_ids = use_learnable_ids 
        self.use_temp_transformer = use_temp_transformer
        
        if use_learnable_ids:
            # Original learnable embeddings
            self.qubit_ids = nn.Embedding(self.max_qubits+1, self.embed_size, padding_idx=self.max_qubits)
        else:
            self.register_buffer("qubit_ids", self._create_ids(max_qubits, embed_size))
            
        self.slice_encoding = gnn.GCNConv(self.embed_size, self.embed_size, add_self_loops=False)
        self.positional_encoding = PositionalEncoder(self.embed_size)
    
    def _create_ids(self, max_qubits, embed_size):
        cache_key = (max_qubits, embed_size)
        
        if cache_key not in _ORTHOGONAL_IDS_CACHE:
            if max_qubits > embed_size:
                raise ValueError(f"Cannot create {max_qubits} orthogonal vectors in {embed_size}D space")
            #TODO: Relax this constraint in future? (for more qubits)
            
            generator = torch.Generator()
            generator.manual_seed(42)  # Fixed seed!!
            
            random_matrix = torch.randn(embed_size, max_qubits, generator=generator)
            q, _ = torch.linalg.qr(random_matrix, mode='reduced')
            
            ids = q[:, :max_qubits].t()
            ids = ids / ids.norm(dim=1, keepdim=True)
            
            # We're caching the tensor so as to ensure consistency across all instances
            _ORTHOGONAL_IDS_CACHE[cache_key] = ids
        
        return _ORTHOGONAL_IDS_CACHE[cache_key].clone()
    
    def forward(self, adj_matrices: torch.Tensor) -> torch.Tensor:
        """Compute qubit embeddings for a (potentially padded) circuit.

        Args:
            adj_matrices: [B, S, Q, Q] adjacency tensor. Q can be <= max_qubits.
        """
        batch_size = adj_matrices.shape[0]
        num_slices = adj_matrices.shape[1]
        num_qubits = adj_matrices.shape[-1]

        if num_qubits > self.max_qubits:
            raise ValueError(
                f"adjacency has {num_qubits} qubits but max_qubits={self.max_qubits}."
            )

        # This ensures that qubits from different batches and slices don't overlap in the graph
        # edges[[2,3]] gives source and target qubit indices for each edge
        # edges[0] gives the batch index for each edge, edges[1] the slice index.
        # edges[0] * num_slices * self.max_qubits + edges[1] * self.max_qubits shifts
        # the qubit indices so that each batch and slice get a unique range of node indices in the graph
        
        if self.use_learnable_ids:
            ids = self.qubit_ids.weight[:num_qubits]  # [Q, D]
        else:
            # Use cached orthogonal IDs
            ids = self.qubit_ids[:num_qubits]  # [Q, D]

        # Expand ids for batch*slices then flatten to node list
        nodes = ids.unsqueeze(0).expand(batch_size * num_slices, -1, -1).reshape(
            batch_size * num_slices * num_qubits, -1
        )

        # Only when we use the temporal transformer do we need to compute GNN embeddings in this step
        # GNN embeddings for lookahead method are computed in that module
        if self.use_temp_transformer:
            # TODO: If we use the temporal transformer + FastTdDataset, we need to convert the dense slices back to sparse here
            # But I'm leaving it as it is for now
            if adj_matrices.layout == torch.sparse_coo:
                edges = adj_matrices._indices()  # [4, num_edges] (batch, slice, src, dst)
            else:
                # Support dense tensors by extracting the non-zero connections as edges
                edges = torch.nonzero(adj_matrices, as_tuple=False).t()  # [4, num_edges]

            edges = edges.to(nodes.device)
            if edges.numel() == 0 or edges.size(1) == 0:
                out = nodes  # No edges -> fall back to base embeddings
            else:
                edge_index = edges[[2, 3]] + edges[0] * num_slices * num_qubits + edges[1] * num_qubits

                # GNN forward
                out = self.slice_encoding(nodes, edge_index)  # [B*S*Q, D]
            out = out.view(batch_size, num_slices, num_qubits, -1)
            # Positional encoding (per slice)
            if self.positional_encoding is not None:
                out = self.positional_encoding(out)
            return out
        
        return nodes.view(batch_size, num_slices, num_qubits, -1) # [batch, num_slices, num_qubits, hidden_dim] #TODO: Not ideal
    


class QubitContextTransformer(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, num_heads=4, dropout=0.1, normalization="instance", use_reverse_causal=False):
        super().__init__()
        self.use_reverse_causal = use_reverse_causal
        # Note: an error is thrown if we use both masks and causal
        # But here we don't need masks anyway

        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                normalization=normalization,
                bias=True,
                causal=use_reverse_causal,
            ) for _ in range(num_layers)
        ])

    def forward(self, qubit_embeds):
        # qubit_embeds: [batch, num_slices, num_qubits, hidden_dim]
        B, T, Q, d = qubit_embeds.shape
        x = qubit_embeds.permute(0, 2, 1, 3)  # [batch, num_qubits, num_slices, hidden_dim]
        x = x.reshape(B*Q, T, d)  
        
        # For reverse causal, flip time dimension
        if self.use_reverse_causal:
            x = torch.flip(x, dims=[1])  
        
        for layer in self.layers:
            x = layer(x, mask=None)
        
        # Flip back if we used reverse causal!
        if self.use_reverse_causal:
            x = torch.flip(x, dims=[1])  
            
        return x.view(B, Q, T, d).permute(0, 2, 1, 3)  # [batch, num_slices, num_qubits, hidden_dim] 



class QubitContextLookahead(nn.Module):
    """Alternative to QubitContextTransformer using lookahead weights and a GNN.
    """

    def __init__(self, num_qubits, embed_dim, num_layers: int = 2, lookahead_weight: float = 0.5):
        super().__init__()
        self.embed_dim = embed_dim
        # num_qubits acts as an upper bound; actual slices may have fewer qubits
        self.max_qubits = num_qubits
        self.num_layers = num_layers
        self.lookahead_weight = lookahead_weight
        self.init_qubit_embeddings = nn.Embedding(self.max_qubits+1, self.embed_dim, padding_idx=self.max_qubits)
        self.gnn_layers = nn.ModuleList([
            gnn.DenseGCNConv(self.embed_dim, self.embed_dim) for _ in range(self.num_layers)
        ])

    def _get_lookahead_weights(self, slices) -> torch.Tensor:
        """Compute lookahead weights for each slice.

        Accepts either a tensor [B, T, Q, Q] or a dict with key 'slices'.
        """

        if isinstance(slices, dict):
            slices = slices['slices'] # [B, T, Q, Q] sparse tensor

        # We convert to dense for simplicity (advanced indexing is tricky with COO)
        if slices.layout == torch.sparse_coo:
            slices = slices.to_dense().float()
        
        B, T, Q, _ = slices.shape

        weights = torch.zeros_like(slices, dtype=torch.float, device=slices.device)
        # Initialize with last slice
        weights[:, -1] = self.lookahead_weight * slices[:, -1]
        
        # Backward pass through time slices
        for slice_idx in range(T-2, -1, -1):
            weights[:, slice_idx] = self.lookahead_weight * (
                weights[:, slice_idx + 1] + slices[:, slice_idx]
            )

        return weights
    
    def forward(self, agent_embeds, adj_matrices: torch.Tensor) -> torch.Tensor: 
        B = adj_matrices.shape[0]
        T = adj_matrices.shape[1]
        num_qubits = adj_matrices.shape[-1]

        if num_qubits > self.max_qubits:
            raise ValueError(f"adjacency has {num_qubits} qubits but max_qubits={self.max_qubits}")

        weights = self._get_lookahead_weights(adj_matrices)  # [B, S, Q, Q] (dense)
        # Note that the weight matrices will be denser than the slice matrices.
        # So we can pass them directly to the GNN. 

        #base_nodes = self.init_qubit_embeddings.weight[:-1] # [Q, D]
        #base_nodes = agent_embeds
        # Now tile to [B, T, Q, D]
        #nodes = base_nodes.to(device).unsqueeze(0).unsqueeze(0).expand(B, T, self.num_qubits, self.embed_dim)

        x = agent_embeds.view(B * T, num_qubits, self.embed_dim)  # [B*T, Q, D]
        w = weights.view(B * T, num_qubits, num_qubits)  # [B*T, Q, Q]

        for layer in self.gnn_layers:
            x = layer(x, w)

        out = x  # [B*T*Q, D]
        out = out.view(B, T, num_qubits, self.embed_dim) # [B, T, Q, D]

        return out



class QAInitEmbedding(nn.Module):
    """Initial embedding for Qubit Allocation env.

    Produces:
      - Agent (qubit) embeddings (slice-aware)
      - Temporal/global tokens
    """

    def __init__(
        self,
        num_qubits: int = None,
        embed_dim: int = 0,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        normalization: str = "instance",
        use_temp_transformer: bool = True,
        lookahead_weight: float = 0.5,
        use_learnable_qubit_ids: bool = False,
        max_qubits: int = None,
    ):
        super().__init__()
        self.max_qubits = max_qubits if max_qubits is not None else num_qubits
        if self.max_qubits is None:
            raise ValueError("QAInitEmbedding requires max_qubits (or num_qubits as alias).")
        # Agent side
        self.qubit_embds = QubitEmbedding(
            embed_dim, 
            self.max_qubits, 
            dropout=dropout, 
            use_learnable_ids=use_learnable_qubit_ids,
            use_temp_transformer=use_temp_transformer
            )
        if use_temp_transformer:
            self.temporal_qubit_embds = QubitContextTransformer(
                hidden_dim=embed_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout, 
                normalization=normalization, use_reverse_causal=True
            )

        else:
            self.temporal_qubit_embds = QubitContextLookahead(
                self.max_qubits, embed_dim, num_layers=num_layers, lookahead_weight=lookahead_weight
            )
            self.global_qubit_embds = None

    def forward(self, adj_matrices: torch.Tensor):
        """
        Args:
            adj_matrices: Tensor with shape [B, S, Q, Q]
            slice_idx: int, index of current slice
        Returns:
            agent_embeds: [B, Q, d]
            core_embeds:  [B, C, d]
            slice_token:  [B, d]
            global_token: [B, d]
        """
        # Slice GNN embeddings
        agent_gnn_embeds = self.qubit_embds(adj_matrices)  # [B, S, Q, d]

        # Temporal agent embeddings
        agent_slice_embds = self.temporal_qubit_embds(agent_gnn_embeds)  # [B, S, Q, d]
        
        return agent_slice_embds
    


class CoreFeatureEncoder(nn.Module):
    """Encode dynamic per-core features into embeddings.

    Args:
        core_size: max capacity per core
        distance_matrix: [num_cores, num_cores] distances
        embed_dim: output embedding dimension
        linear_bias: use bias in linear layers
    """

    def __init__(self,  embed_dim, linear_bias=False):
        super().__init__()

        self.capacity_proj = nn.Linear(1, embed_dim, bias=linear_bias)
        #self.dist_proj = nn.Linear(distance_matrix.size(-1), embed_dim, bias=linear_bias)

    def forward(self, core_capacities: torch.Tensor, core_size: torch.tensor):
        """
        Args:
            td: TensorDict containing
                - 'current_core_capacity': [B, C]
                - 'last_assignment': [B, Q]
        Returns:
            core_feats: [B, C, d]
        """
        # capacities are given without the batch size
        # core_capacities has shape [C+1] where the last entry is the buffer
        # we remove it here. 
        cap = core_capacities[:-1].unsqueeze(-1) / core_size.unsqueeze(-1)  # normalize [0,1]
        cap_emb = self.capacity_proj(cap)  # [B,C,d]

        # distance-based features
        # Here, we just use static per-core distances as embedding
        #dist_emb = self.dist_proj(self.distance_matrix.unsqueeze(0).expand(td.batch_size[0], -1, -1))

        return cap_emb.unsqueeze(0)  # [B, C, d]
    


class AgentBinder(nn.Module):
    """
    Performs binding  of qubit identity and interaction costs.

    Motivation:
        In a pair-agent setup, simply pooling qubit embeddings and distance embeddings separately
        introduces a commutativity ambiguity. The model sees the total pair features and the 
        total distance cost, but loses the specific attribution of which qubit pays which cost.
        
        Example: 
        - Case A: hub qubit (high value) at dist 0 + leaf qubit (low value) at dist 2.
        - Case B: hub qubit at dist 2 + leaf qubit at dist 0.
        
        Both sum to the same total distance (2), but case A is better strategically.

    Mechanism:
        This module fuses the individual qubit embedding [d] with its specific distance embedding [d]
        via a non-linear MLP before any pooling occurs. 
    """

    def __init__(self, embed_dim):
        super().__init__()
        # 2*d -> d for efficiency (i will try try larger later)
        self.proj_in = nn.Linear(2 * embed_dim, embed_dim)
        self.act = nn.GELU() # or nn.ReLU(), TODO: ablate
        self.proj_out = nn.Linear(embed_dim, embed_dim)
        # add residual so the model still knows who the qubit is
        self.resid_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, qubit_emb, dist_emb):
        """
        qubit_emb: [B, Q, C, d] (Expanded)
        dist_emb:  [B, Q, C, d] (Projected)
        """
        # [B, Q, C, 2d]
        combined = torch.cat([qubit_emb, dist_emb], dim=-1)
        
        # [B, Q, C, d]
        fused = self.proj_in(combined)
        fused = self.act(fused)
        fused = self.proj_out(fused)
        
        return fused + (self.resid_scale * qubit_emb)
    

    
class DynamicAgentGrouper(nn.Module):
    """
    Computes distances for all qubits to all cores.
    Binds qubit identity + distance using AgentBinder.
    Groups qubits into pairs and singletons.
    Returns a padded tensor of agents
        """
    def __init__(self, max_qubits: int, embed_dim: int, distance_matrix: Optional[torch.Tensor]):
        super().__init__()

        # project scalar distance to embedding vector
        self.dist_proj = nn.Linear(1, embed_dim, bias=False)

        self.binder = AgentBinder(embed_dim)

        # distance matrix is optional; fall back to the one passed at runtime
        if distance_matrix is not None:
            self.register_buffer("distance_matrix", distance_matrix)
        else:
            self.distance_matrix = None

    def _get_dist(self, prev_core_allocs, core_connectivity: torch.Tensor):
        """
        Computes [B, Q, C] distance matrix based on current assignments
        """ # [B, Q]
        B, Q = prev_core_allocs.shape

        num_cores = core_connectivity.size(0)

        is_buffer = prev_core_allocs >= num_cores
        safe_assignments = prev_core_allocs.clamp(0, num_cores - 1)
        
        # TODO: Review this safe assignment logic
        # [B*Q] -> select rows -> [B*Q, C]
        assigned_rows = safe_assignments.long().view(-1)
        qubit_core_dist = core_connectivity.index_select(0, assigned_rows)
        
        # Reshape to [B, Q, C]
        qubit_core_dist = qubit_core_dist.view(B, Q, num_cores)

        # [B, Q, 1] to [B, Q, C]
        qubit_core_dist = torch.where(
            is_buffer.unsqueeze(-1), 
            torch.zeros_like(qubit_core_dist), 
            qubit_core_dist
        ) # if qubit is in the buffer, replace its row of distances with zeros
        
        return qubit_core_dist
    

    @staticmethod
    def _batch_positions(batch_idx, counts):
        if batch_idx.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=batch_idx.device)
        counts = counts.to(batch_idx.device)
        offsets = torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)
        return torch.arange(batch_idx.size(0), device=batch_idx.device) - offsets
    

    def forward(
        self,
        qubit_embeds: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
        prev_core_allocs: Optional[torch.Tensor] = None,
        current_core_allocs: Optional[torch.Tensor] = None,
        core_connectivity: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ):
        """
        Simplified, batch-1 friendly grouper.

        Args:
            qubit_embeds: [B, Q, d] or [Q, d] (B defaults to 1)
            adjacency: [Q, Q] or [B, Q, Q] for the current slice
            prev_core_allocs: [B, Q] or [Q] with current core indices; values >= num_cores = buffer
            current_core_allocs: same shape as prev_core_allocs; used to filter already placed qubits
            core_connectivity: [C, C] distance matrix (falls back to the buffer if provided in __init__)
            action_mask: [B, Q, C+1] boolean action mask per qubit (optional)
        Returns:
            agent_embeds: [B, Agents, C, d]
            agent_mask: [B, Agents]
            agent_demands: [B, Agents] (2 for pairs, 1 for singles)
            final_action_mask: [B, Agents, C+1]
        """
        # TODO: Do we really need padding? Batch size is always 1

        # Normalize shapes to always include batch dim (B=1 by default)
        if qubit_embeds.dim() == 2:
            qubit_embeds = qubit_embeds.unsqueeze(0)
        B, Q, D = qubit_embeds.shape
        device = qubit_embeds.device

        # core_connectivity = core_connectivity if core_connectivity is not None else self.distance_matrix
        # if core_connectivity is None:
        #     raise ValueError("core_connectivity must be provided (either in __init__ or at call time).")
        num_cores = core_connectivity.size(0)

        # all unassigned -> buffer index = num_cores TODO: Maybe not necessary
        # if prev_core_allocs is None:
        #     prev_core_allocs = torch.full((B, Q), num_cores, device=device, dtype=torch.long)
        # elif prev_core_allocs.dim() == 1:
        #     prev_core_allocs = prev_core_allocs.unsqueeze(0)

        # if current_core_allocs is None:
        #     current_core_allocs = torch.full((B, Q), num_cores, device=device, dtype=torch.long)
        # elif current_core_allocs.dim() == 1:
        #     current_core_allocs = current_core_allocs.unsqueeze(0)

        # Distances
        # qubits in buffer have distance 0 to all cores
        qubit_core_dist = self._get_dist(prev_core_allocs, core_connectivity)  # [B, Q, C]
        dist_embeds = self.dist_proj(qubit_core_dist.unsqueeze(-1))
        q_expanded = qubit_embeds.unsqueeze(2).expand(-1, -1, num_cores, -1)
        bound_embeds = self.binder(q_expanded, dist_embeds)  # [B, Q, C, d]

        # handle action mask
        if action_mask is None:
            env_mask = torch.ones(B, Q, num_cores + 1, dtype=torch.bool, device=device)
        else:
            env_mask = action_mask if action_mask.dim() == 3 else action_mask.unsqueeze(0)
            # TODO:why was this line added in the first place?
            #bound_embeds = bound_embeds.masked_fill(~env_mask.unsqueeze(-1), 0)

        if adj_matrix is None:
            adj_matrix = torch.zeros(B, Q, Q, device=device, dtype=bound_embeds.dtype)
        elif adj_matrix.dim() == 2:
            adj_matrix = adj_matrix.unsqueeze(0)

        # Build pairs
        pair_mask = torch.triu(adj_matrix, diagonal=1) > 0
        b_idx, q1, q2 = torch.where(pair_mask)

        # keep only pairs whose first qubit is unassigned 
        unassigned = current_core_allocs[b_idx, q1] == num_cores
        b_idx_p, q1, q2 = b_idx[unassigned], q1[unassigned], q2[unassigned]
        # TODO: here we sum the embeddings of the two qubits; we could also concatenate and reproject them
        pair_agents = bound_embeds[b_idx_p, q1] + bound_embeds[b_idx_p, q2]

        # singletons
        is_used = torch.zeros((B, Q), dtype=torch.bool, device=device)
        is_used[b_idx_p, q1] = True
        is_used[b_idx_p, q2] = True
        b_idx_s, q_s = torch.where(~is_used)
        single_agents = bound_embeds[b_idx_s, q_s]

        # action mask: 
        pair_action_mask = env_mask[b_idx_p, q1] if pair_agents.numel() > 0 else bound_embeds.new_zeros((0, num_cores + 1), dtype=torch.bool)
        single_action_mask = env_mask[b_idx_s, q_s] if single_agents.numel() > 0 else bound_embeds.new_zeros((0, num_cores + 1), dtype=torch.bool)

        # counts per batch (B=1 in current use, but we keep general padding)
        pair_counts = torch.bincount(b_idx_p, minlength=B).to(device)
        single_counts = torch.bincount(b_idx_s, minlength=B).to(device)
        agent_counts = pair_counts + single_counts
        max_agents = int(agent_counts.max().item()) if agent_counts.numel() > 0 else 0

        agent_embeds = bound_embeds.new_zeros(B, max_agents, num_cores, D)
        agent_demands = bound_embeds.new_zeros(B, max_agents)
        final_action_mask = torch.zeros(B, max_agents, num_cores + 1, dtype=torch.bool, device=device)

        if max_agents == 0:
            agent_mask = agent_demands > 0
            return agent_embeds, agent_mask, agent_demands, final_action_mask

        if pair_agents.numel() > 0:
            pair_pos = self._batch_positions(b_idx_p, pair_counts)
            agent_embeds[b_idx_p, pair_pos] = pair_agents
            agent_demands[b_idx_p, pair_pos] = 2.0
            final_action_mask[b_idx_p, pair_pos] = pair_action_mask

        if single_agents.numel() > 0:
            single_pos = self._batch_positions(b_idx_s, single_counts)
            final_pos = single_pos + pair_counts.to(single_pos.device)[b_idx_s]
            agent_embeds[b_idx_s, final_pos] = single_agents
            agent_demands[b_idx_s, final_pos] = 1.0
            final_action_mask[b_idx_s, final_pos] = single_action_mask

        # TODO: Is this really needed?
        agent_mask = agent_demands > 0 

        return agent_embeds, agent_mask, agent_demands, final_action_mask
    

