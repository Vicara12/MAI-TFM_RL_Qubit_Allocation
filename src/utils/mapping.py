import torch
from typing import Tuple


def map_qubit_to_agent(
    tensor: torch.Tensor,
    q_to_agent: torch.Tensor,
    max_agents: int,
    reducer: str = "mean",
    concat_max_members: int = 2,
) -> Tuple[torch.Tensor, int, int]:
    """Map a qubit dimension to a padded agent dimension.

    Args:
        tensor: input tensor containing a dimension of size ``n_qubits``.
        q_to_agent: long tensor mapping each qubit index to an agent index.
        max_agents: padded agent dimension length.
        reducer: one of {"mean", "any", "all"} controlling how to aggregate over qubits in the same agent.

    Returns:
        (mapped, num_agents, max_agents)
    """
    num_agents = int(q_to_agent.max().item() + 1) if q_to_agent.numel() > 0 else 0
    q_dim = next(i for i, s in enumerate(tensor.shape) if s == q_to_agent.numel())
    idx = q_to_agent.view([1]*q_dim + [-1] + [1]*(tensor.ndim - q_dim - 1))
    idx = idx.expand(*tensor.shape[:q_dim], -1, *tensor.shape[q_dim+1:])
    out_shape = list(tensor.shape)
    out_shape[q_dim] = max_agents

    if reducer in ("any", "all"):
        accum = torch.zeros(out_shape, device=tensor.device, dtype=torch.int)
        accum.scatter_add_(q_dim, idx, tensor.to(torch.int))
        counts = torch.bincount(q_to_agent, minlength=max_agents).to(tensor.device).clamp_min(1)
        shape = [1]*tensor.ndim
        shape[q_dim] = -1
        if reducer == "any":
            out = accum > 0
        else:
            out = accum == counts.view(shape)
        return out, num_agents, max_agents

    if reducer == "concat":
        # Concatenate up to 2 embeddings per agent along the last dim, vectorized.
        if concat_max_members != 2:
            raise ValueError("concat reducer is specialized for concat_max_members=2")
        if q_dim != 1:
            raise ValueError("concat reducer currently expects qubit dimension at position 1")
        if tensor.ndim < 3:
            raise ValueError("concat reducer expects tensor with at least 3 dims (B, Q, ..., H)")

        B, Q, feat_dim = tensor.shape[0], tensor.shape[1], tensor.shape[-1]
        tail_shape = tensor.shape[2:-1]  # e.g., cores
        tail_size = int(torch.tensor(tail_shape).prod().item()) if tail_shape else 1

        flat = tensor.reshape(B, Q, tail_size, feat_dim)  # [B, Q, T, H]

        # Build per-agent list of qubit indices, take smallest 2 per agent.
        idx_mat = torch.full((max_agents, Q), Q, device=tensor.device, dtype=torch.long)
        arange_q = torch.arange(Q, device=tensor.device)
        idx_mat[q_to_agent, arange_q] = arange_q
        picked_vals, picked = torch.topk(idx_mat, k=min(2, Q), dim=1, largest=False)  # [A, 2]
        picked_mask = picked < Q

        gather_idx = picked.view(1, max_agents, 2, 1, 1).expand(B, -1, -1, tail_size, feat_dim)
        flat_exp = flat.unsqueeze(1).expand(-1, max_agents, -1, -1, -1)
        gathered = torch.gather(flat_exp, 2, gather_idx)  # [B, A, 2, T, H]
        gathered = gathered * picked_mask.view(1, max_agents, 2, 1, 1)

        # [B, A, T, 2, H] -> [B, A, T, 2H]
        gathered = gathered.permute(0, 1, 3, 2, 4).reshape(B, max_agents, *tail_shape, 2 * feat_dim)
        return gathered, num_agents, max_agents

    out = torch.zeros(out_shape, device=tensor.device, dtype=tensor.dtype)
    out.scatter_add_(q_dim, idx, tensor)
    if reducer == "mean":
        counts = torch.bincount(q_to_agent, minlength=max_agents).to(tensor.device).clamp_min(1)
        shape = [1]*tensor.ndim
        shape[q_dim] = -1
        out = out / counts.view(shape)
    return out, num_agents, max_agents


def map_agent_to_qubit(
    tensor: torch.Tensor, 
    q_to_agent: torch.Tensor, 
    max_agents: int
) -> Tuple[torch.Tensor, int, int]:
    """Broadcast agent-dim tensor back to qubit dimension using ``q_to_agent``."""
    num_agents = int(q_to_agent.max().item() + 1) if q_to_agent.numel() > 0 else 0
    a_dim = next(i for i, s in enumerate(tensor.shape) if s == max_agents)
    idx = q_to_agent.view([1]*a_dim + [-1] + [1]*(tensor.ndim - a_dim - 1))
    idx = idx.expand(*tensor.shape[:a_dim], -1, *tensor.shape[a_dim+1:])
    return torch.gather(tensor, a_dim, idx), num_agents, max_agents
