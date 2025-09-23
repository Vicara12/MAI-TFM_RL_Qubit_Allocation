import math
import torch
import torch.nn as nn
import torch_geometric.nn as gnn

def pool_slices(agent_embeds, pool_type="mean"):
    # agent_embeds: [batch, num_slices, num_qubits, hidden_dim]
    if pool_type == "mean":
        return agent_embeds.mean(dim=2)  # [batch, num_slices, hidden_dim]
    elif pool_type == "max":
        return agent_embeds.max(dim=2).values  # [batch, num_slices, hidden_dim]
    else:
        raise NotImplementedError("Only mean/max pooling supported for now.")
    
def get_context_tokens(slice_context, slice_idx):
    # slice_context: [batch, num_slices, hidden_dim]
    # slice_idx: int or [batch] (current slice index)
    current_slice_token = slice_context[:, slice_idx]  # [batch, hidden_dim]
    global_context_token = slice_context.mean(dim=1)   # [batch, hidden_dim]
    return current_slice_token, global_context_token

class SliceEmbedding(nn.Module):
    """
    Embeds each qubit (agent) in each slice using a GNN.
    Output: [batch, num_slices, num_qubits, hidden_dim]
    """
    def __init__(self, hidden_dim, num_qubits, dropout:=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_qubits = num_qubits
        self.qubit_embeddings = nn.Embedding(self.num_qubits+1, self.hidden_dim, padding_idx=self.num_qubits)
        self.slice_encoding = gnn.GCNConv(self.hidden_dim, self.hidden_dim, add_self_loops=False)
        self.positional_encoding = PositionalEncoding(self.hidden_dim, dropout=dropout)

    def forward(self, td):
        batch_size = td['slices'].shape[0]
        num_slices = td['slices'].shape[1]

        # Prepare edge_index for PyG
        edges = td['slices']._indices()
        edge_index = edges[[2,3]] + edges[0] * num_slices * self.num_qubits + edges[1] * self.num_qubits

        # Node features: repeat qubit embeddings for each batch and slice
        nodes = self.qubit_embeddings.weight[:-1].repeat(batch_size * num_slices, 1)  # [B*S*Q, D]

        # GNN forward
        out = self.slice_encoding(nodes, edge_index)  # [B*S*Q, D]

        # Reshape to [batch, num_slices, num_qubits, hidden_dim]
        out = out.view(batch_size, num_slices, self.num_qubits, -1)

        # Optionally add positional encoding (per slice)
        out = self.positional_encoding(out)

        return out  # [batch, num_slices, num_qubits, hidden_dim]
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, ..., d_model]
        x = x + self.pe[:x.size(1)].unsqueeze(0).unsqueeze(2)
        return self.dropout(x)
    

class SliceContextTransformer(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, slice_embeds):
        # slice_embeds: [batch, num_slices, hidden_dim]
        return self.encoder(slice_embeds)  # [batch, num_slices, hidden_dim]
    

class TemporalGlobalContext(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, num_heads=4, dropout=0.1, pool_type="mean"):
        super().__init__()
        self.pool_type = pool_type
        self.slice_transformer = SliceContextTransformer(hidden_dim, num_layers, num_heads, dropout)

    def forward(self, agent_embeds, slice_idx):
        # agent_embeds: [batch, num_slices, num_qubits, hidden_dim]
        slice_embeds = pool_slices(agent_embeds, self.pool_type)  # [batch, num_slices, hidden_dim]
        slice_context = self.slice_transformer(slice_embeds)      # [batch, num_slices, hidden_dim]
        current_slice_token, global_context_token = get_context_tokens(slice_context, slice_idx)
        return current_slice_token, global_context_token  # both [batch, hidden_dim]
    
#TODO: Add relative positional encoding 
