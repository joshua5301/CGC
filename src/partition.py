import faiss
import torch

EPS = 1e-12

def kmeans_init(X: torch.Tensor, cluster_num: int):
    X_np = X.detach().cpu().numpy().astype('float32')
    kmeans = faiss.Kmeans(X_np.shape[1], cluster_num, gpu=False)
    kmeans.cp.min_points_per_centroid = 1
    kmeans.train(X_np)
    _, assign = kmeans.index.search(X_np, 1)
    return torch.from_numpy(assign.flatten()).long().to(X.device)

def geometric_medians(X: torch.Tensor, assign: torch.Tensor, cluster_num: int, iters=30):
    counts = torch.bincount(assign, minlength=cluster_num).clamp(min=1).to(X.dtype)
    centers = torch.zeros(cluster_num, X.shape[1], dtype=X.dtype, device=X.device).index_add_(0, assign, X) / counts.unsqueeze(1)
    for _ in range(iters):
        w = 1.0 / (X - centers[assign]).norm(dim=1).clamp(min=EPS)
        num = torch.zeros_like(centers).index_add_(0, assign, w.unsqueeze(1) * X)
        den = torch.zeros(cluster_num, dtype=X.dtype, device=X.device).index_add_(0, assign, w).clamp(min=EPS)
        centers = num / den.unsqueeze(1)
    return centers

def cell_means(y_pred: torch.Tensor, assign: torch.Tensor, cluster_num: int):
    counts = torch.bincount(assign, minlength=cluster_num).clamp(min=1).to(y_pred.dtype)
    return torch.zeros(cluster_num, y_pred.shape[1], dtype=y_pred.dtype, device=y_pred.device).index_add_(0, assign, y_pred) / counts.unsqueeze(1)

def partition(X: torch.Tensor, y_pred: torch.Tensor, cluster_num: int, kl_weight=0.5, iters=100, return_state=False):
    X, y_pred = X.double(), y_pred.double().clamp(min=EPS)
    assign = kmeans_init(X, cluster_num)
    entropy = (y_pred * y_pred.log()).sum(1, keepdim=True)

    global_center = geometric_medians(X, torch.zeros(len(X), dtype=torch.long, device=X.device), 1)
    dist_scale = (X - global_center).norm(dim=1).mean()
    global_label = y_pred.mean(0)
    kl_scale = (entropy.squeeze(1) - y_pred @ global_label.log()).mean()
    centers = geometric_medians(X, assign, cluster_num)
    for _ in range(iters):
        labels = cell_means(y_pred, assign, cluster_num).clamp(min=EPS)
        costs = []
        for X_chunk, y_pred_chunk, ent_chunk in zip(X.split(8192), y_pred.split(8192), entropy.split(8192)):
            dist = torch.cdist(X_chunk, centers) / dist_scale
            kl = (ent_chunk - y_pred_chunk @ labels.log().T) / kl_scale
            costs.append(dist + kl_weight * kl)
        new_assign = torch.cat(costs).argmin(dim=1)
        moved = int((new_assign != assign).sum())
        assign = new_assign
        centers = geometric_medians(X, assign, cluster_num)
        if moved == 0:
            break

    keep = torch.bincount(assign, minlength=cluster_num) > 0
    remap = torch.cumsum(keep, 0) - 1
    assign = remap[assign]
    X_cond = centers[keep]
    y_cond = cell_means(y_pred, assign, int(keep.sum()))
    if return_state:
        return X_cond, y_cond, assign, moved == 0
    return X_cond, y_cond
