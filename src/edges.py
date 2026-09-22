import torch


def coarsen(adj: torch.Tensor, assign: torch.Tensor, cluster_num: int):
    """Cell-averaged propagation matrix: W[j, r] = mean over t in cell j of sum over u in cell r of adj[t, u].
    Feeding the cell-r representative of (adj X) with weight W[j, r] reproduces the cell-j mean of adj (adj X)."""
    n = len(assign)
    Pi = torch.zeros(n, cluster_num, dtype=adj.dtype, device=adj.device)
    Pi[torch.arange(n, device=adj.device), assign] = 1.0
    AP = torch.sparse.mm(adj, Pi)
    W = torch.zeros(cluster_num, cluster_num, dtype=adj.dtype, device=adj.device).index_add_(0, assign, AP)
    counts = torch.bincount(assign, minlength=cluster_num).clamp(min=1).to(adj.dtype)
    return W / counts.unsqueeze(1)


def to_graph(W: torch.Tensor):
    """Dense propagation matrix W (target j, source r) -> (edge_index [source; target], edge_attr)."""
    idx = W.nonzero().t()
    return idx.flip(0).contiguous(), W[idx[0], idx[1]]
