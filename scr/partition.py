import faiss
import torch

EPS = 1e-12

def kmeans_init(X: torch.Tensor, k: int):
    X_np = X.detach().cpu().numpy().astype('float32')
    kmeans = faiss.Kmeans(X_np.shape[1], k, gpu=False)
    kmeans.cp.min_points_per_centroid = 1
    kmeans.train(X_np)
    _, assign = kmeans.index.search(X_np, 1)
    return torch.from_numpy(assign.flatten()).long().to(X.device)

def geometric_medians(X: torch.Tensor, assign: torch.Tensor, k: int, iters=30):
    counts = torch.bincount(assign, minlength=k).clamp(min=1).to(X.dtype)
    centers = torch.zeros(k, X.shape[1], dtype=X.dtype, device=X.device).index_add_(0, assign, X) / counts.unsqueeze(1)
    for _ in range(iters):
        w = 1.0 / (X - centers[assign]).norm(dim=1).clamp(min=EPS)
        num = torch.zeros_like(centers).index_add_(0, assign, w.unsqueeze(1) * X)
        den = torch.zeros(k, dtype=X.dtype, device=X.device).index_add_(0, assign, w).clamp(min=EPS)
        centers = num / den.unsqueeze(1)
    return centers

def cell_means(P: torch.Tensor, assign: torch.Tensor, k: int):
    counts = torch.bincount(assign, minlength=k).clamp(min=1).to(P.dtype)
    return torch.zeros(k, P.shape[1], dtype=P.dtype, device=P.device).index_add_(0, assign, P) / counts.unsqueeze(1)

def partition(X: torch.Tensor, P: torch.Tensor, k: int, kl_weight=0.5, iters=20):
    X, P = X.double(), P.double().clamp(min=EPS)
    assign = kmeans_init(X, k)
    entropy = (P * P.log()).sum(1, keepdim=True)

    centers = geometric_medians(X, assign, k)
    scale = (X - centers[assign]).norm(dim=1).mean()
    for _ in range(iters):
        labels = cell_means(P, assign, k).clamp(min=EPS)
        costs = []
        for X_chunk, P_chunk, H_chunk in zip(X.split(8192), P.split(8192), entropy.split(8192)):
            dist = torch.cdist(X_chunk, centers) / scale
            kl = H_chunk - P_chunk @ labels.log().T
            costs.append(dist + kl_weight * kl)
        new_assign = torch.cat(costs).argmin(1)
        moved = int((new_assign != assign).sum())
        assign = new_assign
        centers = geometric_medians(X, assign, k)
        if moved == 0:
            break

    keep = torch.bincount(assign, minlength=k) > 0
    remap = torch.cumsum(keep, 0) - 1
    assign = remap[assign]
    x_cond = centers[keep]
    y_cond = cell_means(P, assign, int(keep.sum()))
    return x_cond, y_cond
