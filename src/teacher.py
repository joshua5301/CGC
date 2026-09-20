import math
import torch
import torch.nn.functional as F

EPS = 1e-12
def get_kernel_values(A: torch.Tensor, B: torch.Tensor, kernel_kind: str):
    d = B.shape[1]
    if kernel_kind == 'erf':
        bandwidth = (B * B).sum(1).mean() / d
        S = (A @ B.T) / (d * bandwidth)
        a = (A * A).sum(1, keepdim=True) / (d * bandwidth)
        b = (B * B).sum(1).unsqueeze(0) / (d * bandwidth)
        r = 2 * S / torch.sqrt((1 + 2 * a) * (1 + 2 * b))
        return (2 / math.pi) * torch.asin(r.clamp(-1 + EPS, 1 - EPS))
    if kernel_kind.startswith('relu'):
        layers = int(kernel_kind[4:] or 1)
        bandwidth = (B * B).sum(1).mean() / d
        na = A.norm(dim=1, keepdim=True).clamp(min=EPS)
        nb = B.norm(dim=1).unsqueeze(0).clamp(min=EPS)
        cos = ((A @ B.T) / (na * nb)).clamp(-1 + EPS, 1 - EPS)
        for _ in range(layers):
            th = torch.acos(cos)
            cos = ((torch.sin(th) + (math.pi - th) * torch.cos(th)) / math.pi).clamp(-1 + EPS, 1 - EPS)
        return (na * nb) / (d * bandwidth) * cos
    if kernel_kind == 'rbf':
        d2 = (A * A).sum(1, keepdim=True) + (B * B).sum(1).unsqueeze(0) - 2 * (A @ B.T)
        bandwidth = (B * B).sum(1).mean()
        return torch.exp(-d2.clamp(min=0) / bandwidth)
    if kernel_kind == 'linear':
        return A @ B.T
    raise ValueError(f'unknown teacher kernel {kernel_kind}')

def fit_logistic(X_kernel_train: torch.Tensor, y_train: torch.Tensor, gamma: float, steps=1000):
    X_kernel_train, y_train = X_kernel_train.double(), y_train.double()
    n, dim = X_kernel_train.shape
    W = torch.zeros(dim, y_train.shape[1], dtype=X_kernel_train.dtype, device=X_kernel_train.device).requires_grad_(True)
    opt = torch.optim.LBFGS([W], max_iter=steps, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X_kernel_train @ W, y_train) + gamma / n * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach()

def get_kernel_features(X: torch.Tensor, kernel_kind='erf', basis_num=3000):
    X = X.double()
    B = X if basis_num >= len(X) else X[torch.randperm(len(X))[:basis_num]]

    K_BB = get_kernel_values(B, B, kernel_kind)
    K_BB = (K_BB + K_BB.T) / 2
    eye = torch.eye(len(B), dtype=B.dtype, device=B.device)
    L = torch.linalg.cholesky(K_BB + 1e-8 * K_BB.diagonal().mean() * eye)
    T = torch.linalg.solve_triangular(L, eye, upper=False).T

    feats = []
    for chunk in X.split(8192):
        feats.append(get_kernel_values(chunk, B, kernel_kind) @ T)
    return torch.cat(feats)

def get_teacher_labels(X: torch.Tensor, train_mask: torch.Tensor, y: torch.Tensor,
                       kernel_kind='erf', gamma=0.1, temp=1.0, basis_num=3000):
    X_kernel = get_kernel_features(X, kernel_kind, basis_num)
    y_train = F.one_hot(y[train_mask], int(y.max()) + 1).to(X_kernel.dtype)
    W = fit_logistic(X_kernel[train_mask], y_train, gamma)
    logits = X_kernel @ W
    return F.softmax(logits / temp, dim=1)
