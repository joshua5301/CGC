import numpy as np
import torch
import torch.nn.functional as F


def _ainv(ctx, B):
    return torch.linalg.solve(ctx['A'], B)


def _sq_norm_A(ctx, D):
    return (D * (ctx['A'] @ D)).sum()


def _objective(ctx, Y):
    return (((ctx['M'] @ Y - ctx['Y_L']) ** 2).sum() + ctx['gamma'] * (Y ** 2).sum()).item()


def solve_labels(H_L, Hp, Y_L, beta, gamma):
    H_L, Hp, Y_L = H_L.double(), Hp.double(), Y_L.double()
    n_p = Hp.shape[0]
    eye = torch.eye(n_p, dtype=Hp.dtype, device=Hp.device)

    Kss = Hp @ Hp.T
    R = torch.linalg.inv(Kss + beta * (Kss.diagonal().sum() / n_p) * eye)
    M = (H_L @ Hp.T) @ R

    MtM = M.T @ M
    g = gamma * MtM.diagonal().sum() / n_p
    A = MtM + g * eye

    ctx = {'A': A, 'M': M, 'Y_L': Y_L, 'gamma': g,
           'rank': int(torch.linalg.matrix_rank(Hp))}
    return torch.linalg.solve(A, M.T @ Y_L), ctx


def balance(ctx, Y, prior):
    u = _ainv(ctx, torch.ones(Y.shape[0], 1, dtype=Y.dtype, device=Y.device))
    target = prior.to(Y) * Y.shape[0]
    return Y + u @ ((target - Y.sum(0)) / u.sum()).unsqueeze(0)


def row_normalize(Y):
    return Y + (1 - Y.sum(1, keepdim=True)) / Y.shape[1]


def project_simplex(Y):
    c = Y.shape[1]
    v, _ = torch.sort(Y, dim=1, descending=True)
    cs = v.cumsum(1) - 1
    k = torch.arange(1, c + 1, device=Y.device, dtype=Y.dtype)
    rho = (v - cs / k > 0).to(Y.dtype).cumsum(1).argmax(1, keepdim=True)
    theta = cs.gather(1, rho) / (rho + 1).to(Y.dtype)
    return (Y - theta).clamp(min=0)


def constrain(ctx, Y, prior, mode):
    if mode == 'none':
        return Y, float('nan')
    Yc = balance(ctx, Y, prior)
    if mode in ('row', 'simplex'):
        Yc = row_normalize(Yc)
    if mode == 'simplex':
        Yc = project_simplex(Yc)
    return Yc, (_sq_norm_A(ctx, Yc - Y) / abs(_objective(ctx, Y))).item()


def select_pool(args, data, H):
    if args.h_pool == 'train':
        mask = data.train_mask
    elif args.h_pool == 'all':
        mask = torch.ones(len(H), dtype=torch.bool, device=H.device)
    else:
        held = getattr(data, 'val_mask', None)
        if held is None:
            mask = torch.ones(len(H), dtype=torch.bool, device=H.device)
        else:
            mask = data.train_mask | ~(held | data.test_mask)
    return H[mask], data.y[mask], data.train_mask[mask]


def _cluster_means(H, assign, k):
    dtype, device = H.dtype, H.device
    assign = assign.to(device)
    acc = torch.zeros(k, H.shape[1], dtype=dtype, device=device).index_add_(0, assign, H)
    cnt = torch.zeros(k, dtype=dtype, device=device).index_add_(
        0, assign, torch.ones(len(H), dtype=dtype, device=device))
    keep = cnt > 0
    remap = torch.full((k,), -1, dtype=torch.long, device=device)
    remap[keep] = torch.arange(int(keep.sum()), device=device)
    return acc[keep] / cnt[keep].unsqueeze(1), remap[assign]


def _assign(H, k, method):
    from scr.utils import clustering_fast
    labels = clustering_fast(H.cpu().numpy().astype('float32'), int(k), method)
    return torch.from_numpy(np.ascontiguousarray(labels)).long()


def generate_landmarks(args, H_pool, y_pool):
    n_p = int(args.budget)
    if args.landmark == 'random':
        idx = torch.randperm(len(H_pool), device=H_pool.device)[:n_p]
        return H_pool[idx], None
    if args.landmark == 'class_kmeans':
        assign = torch.zeros(len(H_pool), dtype=torch.long)
        row = 0
        for cls in range(args.num_class):
            m = (y_pool == cls)
            assign[m.cpu()] = _assign(H_pool[m], args.budget_cla[cls], args.clustering) + row
            row += int(args.budget_cla[cls])
        return _cluster_means(H_pool, assign, row)
    method = 'nocluster' if args.landmark == 'random_split' else args.clustering
    return _cluster_means(H_pool, _assign(H_pool, n_p, method), n_p)


def onehot_labels(args, Hp, H_L, y_L, assign, y_pool, tr):
    votes = torch.zeros(Hp.shape[0], args.num_class, device=Hp.device)
    if assign is not None:
        a = assign.to(Hp.device)[tr]
        votes.index_put_((a, y_pool[tr]), torch.ones(len(a), device=Hp.device), accumulate=True)
    centroid = torch.stack([H_L[y_L == k].mean(0) for k in range(args.num_class)])
    fallback = (F.normalize(Hp, dim=1) @ F.normalize(centroid, dim=1).T).argmax(1)
    empty = votes.sum(1) == 0
    return torch.where(empty, fallback, votes.argmax(1)).long()


def solve_labels_logistic(H_L, Hp, Y_L, beta, gamma, steps=200):
    H_L, Hp, Y_L = H_L.double(), Hp.double(), Y_L.double()
    m, n_p = H_L.shape[0], Hp.shape[0]
    eye = torch.eye(n_p, dtype=Hp.dtype, device=Hp.device)

    Kss = Hp @ Hp.T
    R = torch.linalg.inv(Kss + beta * (Kss.diagonal().sum() / n_p) * eye)
    M = (H_L @ Hp.T) @ R
    g = gamma * (M * M).sum() / (m * n_p)
    y = Y_L.argmax(1)

    Y = torch.zeros(n_p, Y_L.shape[1], dtype=Hp.dtype, device=Hp.device, requires_grad=True)
    opt = torch.optim.LBFGS([Y], max_iter=steps, history_size=20, tolerance_grad=1e-10,
                            tolerance_change=1e-14, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(M @ Y, y) + 0.5 * g * (Y ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.enable_grad():
        loss = closure()
    ctx = {'loss': loss.item(), 'gnorm': Y.grad.norm().item(),
           'rank': int(torch.linalg.matrix_rank(Hp)), 'gamma': g.item()}
    return F.softmax(Y.detach(), dim=1), ctx
