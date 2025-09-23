import torch
import torch.nn as nn
import torch_geometric.nn as gnn

class CoreSnapshotEncoder(nn.Module):
    """Encode cores given current/past qubit assignments and core adjacency.

    Args:
        num_qubits: total number of qubits
        num_cores: number of cores
        core_size: maximum qubits per core
        hidden_dim: embedding dimension
        adjacency_matrix: [num_cores, num_cores] torch.Tensor (0/1)
    """

    def __init__(self, num_qubits, num_cores, core_size, hidden_dim, adjacency_matrix):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_cores = num_cores
        self.core_size = core_size
        self.hidden_dim = hidden_dim

        # Graph adjacency (register buffers so they move with .to(device))
        self.register_buffer('adjacency_matrix', adjacency_matrix)
        self.register_buffer('edges', adjacency_matrix.nonzero(as_tuple=False).t())

        # Embedding for qubits
        self.qubit_embedding = nn.Embedding(num_qubits + 1, hidden_dim, padding_idx=num_qubits)

        # Simple GCN over core graph
        self.gnn = gnn.GCNConv(hidden_dim, hidden_dim, add_self_loops=False)

    def forward(self, last_assignment):
        """
        Args:
            last_assignment: [B, num_qubits] tensor with core index for each qubit
        Returns:
            core_embeddings: [B, num_cores, hidden_dim]
        """
        B = last_assignment.size(0)
        device = last_assignment.device

        # Build (B, num_cores, core_size) array of qubits in each core
        last_qubit_assignment = torch.full(
            (B, self.num_cores + 1, self.core_size), self.num_qubits, device=device
        )
        num_qubits_in_core = torch.zeros((B, self.num_cores + 1), dtype=torch.int64, device=device)

        bi = torch.arange(B, device=device)
        for q in range(self.num_qubits):
            c = last_assignment[:, q]  # [B]
            idx = num_qubits_in_core[bi, c]  # slot index
            last_qubit_assignment[bi, c, idx] = q
            num_qubits_in_core[bi, c] += 1
            num_qubits_in_core[:, -1] = 0  # padding bucket unused

        last_qubit_assignment = last_qubit_assignment[:, :-1, :]  # drop padding core

        # Embed and pool qubits per core
        qubit_embs = self.qubit_embedding(last_qubit_assignment)  # [B, C, core_size, d]
        core_embs = qubit_embs.amax(dim=2)                       # [B, C, d]

        # Flatten batch for GCN
        x = core_embs.reshape(B * self.num_cores, self.hidden_dim)
        edge_index = self.edges.repeat(1, B)
        batch_offsets = torch.arange(B, device=device).repeat_interleave(self.edges.size(1)) * self.num_cores
        edge_index = edge_index + batch_offsets.unsqueeze(0)

        x = self.gnn(x, edge_index)  # [B*C, d]
        return x.view(B, self.num_cores, self.hidden_dim)

class CoreFeatureEncoder(nn.Module):
    """Encode dynamic per-core features (capacity, distances) into embeddings.

    Args:
        core_size: max capacity per core
        distance_matrix: [num_cores, num_cores] distances
        embed_dim: output embedding dimension
        linear_bias: use bias in linear layers
    """

    def __init__(self, core_size, distance_matrix, embed_dim, linear_bias=False):
        super().__init__()
        self.core_size = core_size
        self.register_buffer('distance_matrix', distance_matrix)

        self.capacity_proj = nn.Linear(1, embed_dim, bias=linear_bias)
        self.dist_proj = nn.Linear(distance_matrix.size(-1), embed_dim, bias=linear_bias)

    def forward(self, td):
        """
        Args:
            td: TensorDict containing
                - 'current_core_capacity': [B, C]
                - 'last_assignment': [B, Q]
        Returns:
            core_feats: [B, C, d]
        """
        cap = td['current_core_capacity'].unsqueeze(-1) / self.core_size  # normalize [0,1]
        cap_emb = self.capacity_proj(cap)  # [B,C,d]

        # Optional: distance-based features (e.g. mean distance to occupied cores)
        # Here, we just use static per-core distances as embedding
        dist_emb = self.dist_proj(self.distance_matrix.unsqueeze(0).expand(td.batch_size[0], -1, -1))

        return cap_emb + dist_emb  # [B, C, d]
