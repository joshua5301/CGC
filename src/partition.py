import faiss
import torch
from sklearn.cluster import kmeans_plusplus

EPS = 1e-12
PAIRED_INITS = ('uniform_kmeans', 'uniform_kmedians', 'distance_kmeans', 'distance_kmedians')


@torch.no_grad()
def seed_centers(X, cluster_num, seed, distance_sampling):
    generator = torch.Generator(device=X.device).manual_seed(seed)
    if not distance_sampling:
        return torch.randperm(len(X), generator=generator, device=X.device)[:cluster_num]
    selected = torch.zeros(len(X), dtype=torch.bool, device=X.device)
    nearest = X.new_full((len(X),), torch.inf)
    indices = []
    for _ in range(cluster_num):
        weights = nearest.masked_fill(selected, 0) if indices else (~selected).to(X.dtype)
        if float(weights.sum()) == 0:
            weights = (~selected).to(X.dtype)
        index = int(torch.multinomial(weights, 1, generator=generator))
        indices.append(index)
        selected[index] = True
        nearest = torch.minimum(nearest, (X - X[index]).norm(dim=1))
    return torch.tensor(indices, device=X.device)


@torch.no_grad()
def paired_feature_init(X, cluster_num, seed, init, steps=100):
    indices = seed_centers(X, cluster_num, seed, init.startswith('distance_'))
    centers, previous = X[indices].clone(), None
    median = init.endswith('kmedians')
    for step in range(steps):
        distances, assignments = [], []
        for chunk in X.split(2048):
            distance, assignment = torch.cdist(chunk, centers).min(1)
            distances.append(distance)
            assignments.append(assignment)
        assignment, residual = torch.cat(assignments), torch.cat(distances)
        counts = torch.bincount(assignment, minlength=cluster_num)
        for empty in torch.nonzero(counts == 0).flatten().tolist():
            node = int(residual.masked_fill(counts[assignment] <= 1, -torch.inf).argmax())
            counts[assignment[node]] -= 1
            assignment[node] = empty
            counts[empty] += 1
        if previous is not None and torch.equal(previous, assignment):
            return assignment, indices, dict(feature_steps=step + 1, feature_converged=True)
        means = cell_means(X, assignment, cluster_num)
        if median:
            proposed = geometric_medians(X, assignment, cluster_num)
            old_cost = X.new_zeros(cluster_num).index_add_(0, assignment, (X - centers[assignment]).norm(dim=1))
            new_cost = X.new_zeros(cluster_num).index_add_(0, assignment, (X - proposed[assignment]).norm(dim=1))
            mean_cost = X.new_zeros(cluster_num).index_add_(0, assignment, (X - means[assignment]).norm(dim=1))
            use_mean = mean_cost < new_cost
            proposed[use_mean] = means[use_mean]
            new_cost = torch.minimum(new_cost, mean_cost)
            centers = torch.where((new_cost <= old_cost)[:, None], proposed, centers)
        else:
            centers = means
        previous = assignment.clone()
    return assignment, indices, dict(feature_steps=steps, feature_converged=False)

def kmeans_init(X: torch.Tensor, cluster_num: int, seed=1234, plus_plus=False, initial_centers=None):
    X_np = X.detach().cpu().numpy().astype('float32')
    kmeans = faiss.Kmeans(X_np.shape[1], cluster_num, gpu=False)
    kmeans.cp.min_points_per_centroid = 1
    kmeans.cp.seed = seed
    if initial_centers is not None:
        kmeans.train(X_np, init_centroids=initial_centers.detach().cpu().numpy().astype('float32'))
    elif plus_plus:
        centers, _ = kmeans_plusplus(X_np, cluster_num, random_state=seed, n_local_trials=1)
        kmeans.train(X_np, init_centroids=centers)
    else:
        kmeans.train(X_np)
    _, assign = kmeans.index.search(X_np, 1)
    return torch.from_numpy(assign.flatten()).long().to(X.device)


@torch.no_grad()
def greedy_init(X, Q, cluster_num, kl_weight, dist_scale, kl_scale, block_size=256):
    n = len(X)
    entropy, log_q = (Q * Q.log()).sum(1, keepdim=True), Q.log()

    def costs(start, end):
        distance = torch.cdist(X, X[start:end]) / dist_scale
        divergence = (entropy - Q @ log_q[start:end].T).clamp_min(0) / kl_scale
        result = distance + kl_weight * divergence
        rows = torch.arange(start, end, device=X.device)
        result[rows, rows - start] = 0
        return result

    cached = costs(0, n) if n * n <= 16_000_000 else None
    nearest = X.new_full((n,), torch.inf)
    assignment = torch.zeros(n, dtype=torch.long, device=X.device)
    selected = torch.zeros(n, dtype=torch.bool, device=X.device)
    anchors = []
    for k in range(cluster_num):
        best_value, best_node = float('inf'), None
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            cost = cached[:, start:end] if cached is not None else costs(start, end)
            values = torch.minimum(nearest[:, None], cost).sum(0)
            values[selected[start:end]] = torch.inf
            value, index = values.min(0)
            if float(value) < best_value:
                best_value, best_node = float(value), start + int(index)
        cost = cached[:, best_node] if cached is not None else costs(best_node, best_node + 1)[:, 0]
        assignment[cost < nearest] = k
        nearest = torch.minimum(nearest, cost)
        selected[best_node] = True
        anchors.append(best_node)
    assignment[torch.tensor(anchors, device=X.device)] = torch.arange(cluster_num, device=X.device)
    return assignment

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


@torch.no_grad()
def partition_cost(X, Q, assignment, centers, labels, dist_scale, kl_scale, kl_weight, label_weights=None):
    feature = float((X - centers[assignment]).norm(dim=1).mean() / dist_scale)
    divergence = (Q * (Q.clamp_min(EPS).log() - labels[assignment].clamp_min(EPS).log())).sum(1)
    kl = float((divergence if label_weights is None else divergence * label_weights).mean() / kl_scale)
    return dict(feature=feature, kl=kl, weighted_kl=kl_weight * kl, J=feature + kl_weight * kl)


def partition(X: torch.Tensor, y_pred: torch.Tensor, cluster_num: int, kl_weight=0.5, iters=100, return_state=False, seed=1234,
              init='kmeans', init_block_size=256, return_diagnostics=False, feature_steps=100, label_weights=None, return_initial_state=False,
              initial_assignment=None, initial_node_ids=None):
    if initial_assignment is not None and initial_node_ids is not None:
        raise ValueError('Specify only one initial state')
    if init not in ('kmeans', 'kmeans++', 'greedy') + PAIRED_INITS or not 1 <= cluster_num <= len(X) or init_block_size < 1 or kl_weight < 0 or feature_steps < 1:
        raise ValueError('Invalid GRIP initialization or clustering settings')
    X, y_pred = X.double(), y_pred.double().clamp(min=EPS)
    if label_weights is not None:
        label_weights = label_weights.to(device=X.device, dtype=X.dtype)
        if label_weights.shape != (len(X),) or not bool(torch.isfinite(label_weights).all()) or bool((label_weights <= 0).any()):
            raise ValueError('Require one positive finite label weight per node')
        if init == 'greedy':
            raise ValueError('Weighted GRIP requires a feature-only initializer')
        label_weights = label_weights / label_weights.mean()

    def label_means(ids, count):
        if label_weights is None:
            return cell_means(y_pred, ids, count)
        mass = X.new_zeros(count).index_add_(0, ids, label_weights)
        sums = y_pred.new_zeros(count, y_pred.shape[1]).index_add_(0, ids, label_weights[:, None] * y_pred)
        return sums / mass.clamp_min(EPS)[:, None]

    entropy = (y_pred * y_pred.log()).sum(1, keepdim=True)

    global_center = geometric_medians(X, torch.zeros(len(X), dtype=torch.long, device=X.device), 1)
    dist_scale = (X - global_center).norm(dim=1).mean().clamp_min(EPS)
    global_label = y_pred.mean(0)
    kl_scale = (entropy.squeeze(1) - y_pred @ global_label.log()).mean().clamp_min(EPS)
    init_info = {}
    if initial_node_ids is not None:
        ids = initial_node_ids.to(device=X.device, dtype=torch.long)
        if ids.shape != (cluster_num,) or len(ids.unique()) != cluster_num or int(ids.min()) < 0 or int(ids.max()) >= len(X):
            raise ValueError('Require K distinct initial node IDs')
        parts = []
        for start in range(0, len(X), 8192):
            stop = start + 8192
            dist = torch.cdist(X[start:stop], X[ids]) / dist_scale
            kl = (entropy[start:stop] - y_pred[start:stop] @ y_pred[ids].log().T) / kl_scale
            if label_weights is not None:
                kl = kl * label_weights[start:stop, None]
            parts.append((dist + kl_weight * kl).argmin(1))
        assign = torch.cat(parts)
        assign[ids] = torch.arange(cluster_num, device=X.device)
        init_info['initial_centroid_ids'] = ids.cpu()
    elif initial_assignment is not None:
        assign = initial_assignment.to(device=X.device, dtype=torch.long).clone()
        if assign.shape != (len(X),) or int(assign.min()) < 0 or int(assign.max()) >= cluster_num:
            raise ValueError('Invalid initial assignment')
    elif init in PAIRED_INITS:
        assign, indices, init_info = paired_feature_init(X, cluster_num, seed, init, feature_steps)
        init_info['initial_centroid_ids'] = indices.cpu()
    else:
        assign = (kmeans_init(X, cluster_num, seed=seed, plus_plus=init == 'kmeans++') if init != 'greedy' else
                  greedy_init(X, y_pred, cluster_num, kl_weight, dist_scale, kl_scale, init_block_size))
    centers = geometric_medians(X, assign, cluster_num)
    if return_initial_state:
        init_info.update(initial_assignment=assign.cpu().clone(),
                         initial_x=centers.float().cpu().clone(),
                         initial_y=label_means(assign, cluster_num).float().cpu().clone())
    if return_diagnostics:
        means = cell_means(X, assign, cluster_num)
        initial_sse = float((X - means[assign]).square().sum(1).mean())
        initial = partition_cost(X, y_pred, assign, centers,
                                 label_means(assign, cluster_num), dist_scale, kl_scale, kl_weight, label_weights)
    for _ in range(iters):
        labels = label_means(assign, cluster_num).clamp(min=EPS)
        costs = []
        for X_chunk, y_pred_chunk, ent_chunk in zip(X.split(8192), y_pred.split(8192), entropy.split(8192)):
            dist = torch.cdist(X_chunk, centers) / dist_scale
            kl = (ent_chunk - y_pred_chunk @ labels.log().T) / kl_scale
            start = sum(len(v) for v in costs)
            if label_weights is not None:
                kl = kl * label_weights[start:start + len(kl), None]
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
    y_cond = label_means(assign, int(keep.sum()))
    if return_diagnostics:
        final = partition_cost(X, y_pred, assign, X_cond, y_cond, dist_scale, kl_scale, kl_weight, label_weights)
        return dict(x=X_cond.float().cpu(), y=y_cond.float().cpu(),
                    counts=torch.bincount(assign).cpu(), assignment=assign.cpu(),
                    nodes=len(X_cond), initial_sse=initial_sse,
                    **{f'initial_{k}': v for k, v in initial.items()},
                    **{f'final_{k}': v for k, v in final.items()},
                    dist_scale=float(dist_scale), kl_scale=float(kl_scale),
                    converged=moved == 0, **init_info)
    if return_state:
        return X_cond, y_cond, assign, moved == 0
    return X_cond, y_cond
