import torch

EPS = 1e-12


@torch.no_grad()
def label_scale(X: torch.Tensor, labels: torch.Tensor, alpha=0.05, pairs=4096, ridge=1e-3, steps=300, seed=0):
    if not 0 <= alpha < 1 or pairs < 1 or ridge <= 0 or steps < 1:
        raise ValueError('invalid label metric parameters')
    if alpha == 0 or len(X) < 2:
        return torch.ones(X.shape[1], dtype=torch.float64, device=X.device)
    X, labels = X.detach().double(), labels.detach().double().clamp(min=EPS)
    labels = labels / labels.sum(1, keepdim=True)
    generator = torch.Generator(device=X.device).manual_seed(seed)
    u = torch.randint(len(X), (pairs,), device=X.device, generator=generator)
    v = torch.randint(len(X) - 1, (pairs,), device=X.device, generator=generator)
    v = v + (v >= u)
    Q = (X[u] - X[v]).square()
    p, q = labels[u], labels[v]
    mid = (p + q) / 2
    target = ((p * (p.log() - mid.log())).sum(1) + (q * (q.log() - mid.log())).sum(1)) / 2
    target = target.clamp(min=0)
    q_scale, y_scale = Q.square().mean().sqrt(), target.square().mean().sqrt()
    if q_scale <= EPS or y_scale <= EPS:
        return torch.ones(X.shape[1], dtype=X.dtype, device=X.device)
    Q /= q_scale
    target /= y_scale
    rate = 1 / (2 * Q.square().sum() / pairs + 2 * ridge)
    weights = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
    point, momentum = weights.clone(), 1.
    for _ in range(steps):
        gradient = 2 * (Q.T @ (Q @ point - target)) / pairs + 2 * ridge * point
        updated = (point - rate * gradient).clamp(min=0)
        next_momentum = (1 + (1 + 4 * momentum ** 2) ** .5) / 2
        point = updated + (momentum - 1) / next_momentum * (updated - weights)
        weights, momentum = updated, next_momentum
    if not torch.isfinite(weights).all():
        raise RuntimeError('nonfinite label metric weights')
    if weights.mean() <= EPS:
        return torch.ones_like(weights)
    return ((1 - alpha) + alpha * weights / weights.mean()).sqrt()
