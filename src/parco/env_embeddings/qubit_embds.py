import math
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from models.nn.positional_encoder import PositionalEncoder

    
def get_context_tokens(context, slice_idx):
    current_qubit_tokens = context[:, slice_idx]  # [batch, num_qubits, hidden_dim]
    global_context_tokens = context.mean(dim=1)   # [batch, num_qubits, hidden_dim]
    return current_qubit_tokens, global_context_tokens

class QubitEmbedding(nn.Module):
    """
    Embeds each qubit (agent) in each slice using a GNN.
    Output: [batch, num_slices, num_qubits, hidden_dim]
    """
    def __init__(self, hidden_dim, num_qubits, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_qubits = num_qubits
        self.init_qubit_embeddings = nn.Embedding(self.num_qubits+1, self.hidden_dim, padding_idx=self.num_qubits)
        self.slice_encoding = gnn.GCNConv(self.hidden_dim, self.hidden_dim, add_self_loops=False)
        self.positional_encoding = PositionalEncoder(self.hidden_dim)

    def forward(self, td):
        batch_size = td['slices'].shape[0]
        num_slices = td['slices'].shape[1]

        # Prepare edge_index for PyG
        edges = td['slices']._indices()
        edge_index = edges[[2,3]] + edges[0] * num_slices * self.num_qubits + edges[1] * self.num_qubits

        # Node features: repeat qubit embeddings for each batch and slice
        nodes = self.init_qubit_embeddings.weight[:-1].repeat(batch_size * num_slices, 1)  # [B*S*Q, D]

        # GNN forward
        out = self.slice_encoding(nodes, edge_index)  # [B*S*Q, D]
        out = out.view(batch_size, num_slices, self.num_qubits, -1)

        # Optionally add positional encoding (per slice)
        out = self.positional_encoding(out)

        return out  # [batch, num_slices, num_qubits, hidden_dim]
    

class QubitContextTransformer(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, qubit_embeds):
        # qubit_embeds: [batch, num_slices, num_qubits, hidden_dim]
        B, T, Q, d = qubit_embeds.shape
        x = qubit_embeds.permute(0,2,1,3)  # [batch, num_qubits, num_slices, hidden_dim]
        x = x.reshape(B*Q, T, d)  # [batch*num_qubits, num_slices, hidden_dim]
        h = self.encoder(x)  # [batch*num_qubits, num_slices, hidden_dim]
        return h.view(B, Q, T, d).permute(0, 2, 1, 3)  # [batch, num_slices, num_qubits, hidden_dim] 

class TemporalGlobalContext(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, num_heads=4, dropout=0.1, pool_type="mean"):
        super().__init__()
        self.pool_type = pool_type
        self.qubit_transformer = QubitContextTransformer(hidden_dim, num_layers, num_heads, dropout)

    def forward(self, agent_embeds, slice_idx):
        # agent_embeds: [batch, num_slices, num_qubits, hidden_dim]
        agent_context = self.qubit_transformer(agent_embeds)  # [batch, num_qubits, num_slices, hidden_dim]
        current_slice_tokens, global_context_tokens = get_context_tokens(agent_context, slice_idx)  # [batch, num_qubits, hidden_dim]
        return current_slice_tokens, global_context_tokens  # both [batch, hidden_dim]

#TODO: Add relative positional encoding