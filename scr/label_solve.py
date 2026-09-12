import math

import numpy as np
import torch
import torch.nn.functional as F


def _kernel(A, B, kind, d, bw=None):
    if kind == 'linear':
        return A @ B.T
    if kind == 'erf':
        S = (A @ B.T) / (d * bw)
        a = (A * A).sum(1, keepdim=True) / (d * bw)
        b = (B * B).sum(1).unsqueeze(0) / (d * bw)
        r = 2 * S / torch.sqrt((1 + 2 * a) * (1 + 2 * b))
        return (2 / math.pi) * torch.asin(r.clamp(-1 + 1e-12, 1 - 1e-12))
    if kind.startswith('arccos') or kind.startswith('relu'):
        # NNGP kernel of an L-hidden-layer ReLU MLP (Cho & Saul order-1 arc-cosine, composed L times).
        # 'arccos' / 'relu1' = 1 layer, 'relu2', 'relu3', ... = deeper. Only angles matter after layer 1.
        L = 1 if kind == 'arccos' else int(kind[4:] or 1)
        na = A.norm(dim=1, keepdim=True).clamp(min=1e-12)
        nb = B.norm(dim=1).unsqueeze(0).clamp(min=1e-12)
        cos = ((A @ B.T) / (na * nb)).clamp(-1 + 1e-12, 1 - 1e-12)
        scale = (na * nb) / d
        for _ in range(L):
            th = torch.acos(cos)
            cos = (torch.sin(th) + (math.pi - th) * torch.cos(th)) / math.pi   # normalised: diag -> 1
            cos = cos.clamp(-1 + 1e-12, 1 - 1e-12)
        return scale * cos
    if kind == 'rbf':
        D2 = ((A * A).sum(1, keepdim=True) + (B * B).sum(1).unsqueeze(0) - 2 * (A @ B.T))
        return torch.exp(-D2.clamp(min=0) / (2 * bw))
    raise ValueError(kind)


def _design(H_L, Hp, beta, n_p, kind='linear'):
    d = H_L.shape[1]
    bw = None
    if kind == 'erf':
        bw = ((Hp * Hp).sum(1).mean() / d).clamp(min=1e-12)
    if kind == 'rbf':
        D2 = ((Hp * Hp).sum(1, keepdim=True) + (Hp * Hp).sum(1).unsqueeze(0) - 2 * (Hp @ Hp.T))
        bw = D2.clamp(min=0).flatten().median().clamp(min=1e-12)
    if kind == 'linear' and beta <= 0:
        return H_L @ torch.linalg.pinv(Hp), int(torch.linalg.matrix_rank(Hp))
    Kts = _kernel(H_L, Hp, kind, d, bw)
    Kss = _kernel(Hp, Hp, kind, d, bw)
    rank = max(int(torch.linalg.matrix_rank(Kss)), 1)
    if beta <= 0:
        return Kts @ torch.linalg.pinv(Kss), rank
    eye = torch.eye(n_p, dtype=Kss.dtype, device=Kss.device)
    return Kts @ torch.linalg.inv(Kss + beta * (Kss.diagonal().sum() / rank) * eye), rank


def _ainv(ctx, B):
    return torch.linalg.solve(ctx['A'], B)


def _sq_norm_A(ctx, D):
    return (D * (ctx['A'] @ D)).sum()


def _objective(ctx, Y):
    return (((ctx['M'] @ Y - ctx['Y_L']) ** 2).sum() + ctx['gamma'] * (Y ** 2).sum()).item()


def solve_labels(H_L, Hp, Y_L, beta, gamma, kind='linear'):
    H_L, Hp, Y_L = H_L.double(), Hp.double(), Y_L.double()
    n_p = Hp.shape[0]
    eye = torch.eye(n_p, dtype=Hp.dtype, device=Hp.device)

    M, rank = _design(H_L, Hp, beta, n_p, kind)

    MtM = M.T @ M
    g = gamma * MtM.diagonal().sum() / n_p
    A = MtM + g * eye

    ctx = {'A': A, 'M': M, 'Y_L': Y_L, 'gamma': g,
           'rank': rank}
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


def pool_mask(args, data, n, device):
    if args.h_pool == 'train':
        return data.train_mask
    if args.h_pool == 'all':
        return torch.ones(n, dtype=torch.bool, device=device)
    held = getattr(data, 'val_mask', None)
    if held is None:
        return torch.ones(n, dtype=torch.bool, device=device)
    return data.train_mask | ~(held | data.test_mask)


def coarsen_adj(edge_index, edge_weight, mask, assign, k):
    dev = assign.device
    n2l = torch.full((mask.shape[0],), -1, dtype=torch.long, device=dev)
    n2l[mask.to(dev)] = assign
    r, c = n2l[edge_index[0].to(dev)], n2l[edge_index[1].to(dev)]
    ok = (r >= 0) & (c >= 0) & (r != c)
    w = torch.ones(int(ok.sum()), device=dev) if edge_weight is None else edge_weight.to(dev)[ok]
    A = torch.zeros(k, k, device=dev).index_put_((r[ok], c[ok]), w, accumulate=True)
    A = (A + A.T) / 2
    A.fill_diagonal_(0)
    return A


def _norm_adj(A, eye):
    mx = A + eye
    r = mx.sum(1).clamp(min=1e-12).pow(-0.5)
    return r.unsqueeze(1) * mx * r.unsqueeze(0)


def commute_adj(h_d, K=2, steps=300, lr=0.05, l1=0.0, init=None):
    B = h_d[0].double()
    tgt = [t.double() for t in h_d[1:1 + K]]
    n = B.shape[0]
    eye = torch.eye(n, dtype=B.dtype, device=B.device)
    A = (torch.zeros(n, n, dtype=B.dtype, device=B.device) if init is None
         else init.double().clone())
    A = A.clamp(min=0)
    A.fill_diagonal_(0)
    A.requires_grad_(True)
    opt = torch.optim.Adam([A], lr=lr)
    scale = sum(float(t.pow(2).sum()) for t in tgt)
    for _ in range(steps):
        opt.zero_grad()
        S, z, loss = _norm_adj(A, eye), B, 0.0
        for t in tgt:
            z = S @ z
            loss = loss + (z - t).pow(2).sum()
        (loss / scale + l1 * A.abs().sum() / max(n * n, 1)).backward()
        opt.step()
        with torch.no_grad():
            A.data = ((A.data + A.data.T) / 2).clamp(min=0)
            A.data.fill_diagonal_(0)
    return A.detach()


def commutation_residual(a_norm, h_d):
    out, z = [], h_d[0]
    for k in range(1, len(h_d)):
        z = a_norm @ z
        out.append((z - h_d[k]).norm().item() / h_d[k].norm().clamp(min=1e-30).item())
    return out


def select_pool(args, data, H, extra=()):
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
    return H[mask], data.y[mask], data.train_mask[mask], [E[mask] for E in extra]


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


def _whiten(H, alpha, rtol=1e-7):
    if alpha <= 0:
        return H
    m, d = H.shape
    Hc = (H - H.mean(0, keepdim=True)).double()
    S, Vh = torch.linalg.svd(Hc, full_matrices=False)[1:]
    r = min(m - 1, d)
    keep = torch.zeros_like(S, dtype=torch.bool)
    keep[:r] = S[:r] > rtol * S[0]
    T = (Vh[keep].T * (S[keep] / max(m - 1, 1) ** 0.5).pow(-alpha)).to(H.dtype)
    return (H - H.mean(0, keepdim=True)) @ T


def _assign(H, k, method):
    from scr.utils import clustering_fast
    labels = clustering_fast(H.cpu().numpy().astype('float32'), int(k), method)
    return torch.from_numpy(np.ascontiguousarray(labels)).long()


def multiscale_feats(depths, mode, hp_w=1.0, K=None):
    """Clustering-only feature space built from the propagation depths [X, AX, A^2X, ...].
    last: A^K X | concat: all depths | multi: all depths + high-pass (I-A)X, (I-A)^2 X.
    Every block is rescaled to unit mean row norm so no scale dominates k-means."""
    r = lambda A: A / A.norm(dim=1).mean().clamp_min(1e-12)
    if mode == 'last':
        return depths[-1]
    blocks = [r(D) for D in depths]
    if mode == 'multi':
        X, AX = depths[0], depths[1]
        blocks.append(hp_w * r(X - AX))
        if len(depths) > 2:
            blocks.append(hp_w * r(X - 2 * AX + depths[2]))
    return torch.cat(blocks, dim=1)


def posterior_feats(H, P, lam):
    r = lambda A: A / A.norm(dim=1).mean().clamp_min(1e-12)
    return torch.cat([r(H), lam * r(P.to(H.dtype))], dim=1)


def herding_select(H, k):
    """Class-agnostic herding: greedily pick nodes whose running mean tracks the pool mean."""
    Hd = H.double()
    mu = Hd.mean(0)
    run = torch.zeros_like(mu)
    taken = torch.zeros(len(Hd), dtype=torch.bool, device=H.device)
    picked = []
    for t in range(int(k)):
        d = (((run + Hd) / (t + 1) - mu) ** 2).sum(1)
        d[taken] = float('inf')
        i = int(d.argmin())
        taken[i] = True
        run = run + Hd[i]
        picked.append(i)
    return torch.tensor(picked, dtype=torch.long, device=H.device)


def kcenter_select(H, k):
    """Greedy farthest-point (k-center) selection, seeded at the node nearest the mean."""
    Hd = H.double()
    first = int(((Hd - Hd.mean(0)) ** 2).sum(1).argmin())
    picked = [first]
    dmin = ((Hd - Hd[first]) ** 2).sum(1)
    for _ in range(int(k) - 1):
        i = int(dmin.argmax())
        picked.append(i)
        dmin = torch.minimum(dmin, ((Hd - Hd[i]) ** 2).sum(1))
    return torch.tensor(picked, dtype=torch.long, device=H.device)


def _pool_edges(edge_index, mask, n_pool):
    """Edges of the induced subgraph on the pool, relabelled to pool indices (numpy)."""
    dev = mask.device
    n2l = torch.full((mask.shape[0],), -1, dtype=torch.long, device=dev)
    n2l[mask] = torch.arange(n_pool, device=dev)
    r, c = n2l[edge_index[0].to(dev)], n2l[edge_index[1].to(dev)]
    ok = (r >= 0) & (c >= 0) & (r != c)
    return r[ok].cpu().numpy(), c[ok].cpu().numpy()


def metis_partition(edge_index, mask, n_pool, k):
    """Structure-only cells from METIS (pymetis, or torch_sparse's partition as fallback)."""
    import scipy.sparse as sp
    r, c = _pool_edges(edge_index, mask, n_pool)
    A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n_pool, n_pool)).tocsr()
    A = ((A + A.T) > 0).astype(np.int64).tocsr()
    try:
        import pymetis
        _, part = pymetis.part_graph(int(k), xadj=A.indptr.astype(np.int64),
                                     adjncy=A.indices.astype(np.int64))
        return torch.tensor(part, dtype=torch.long)
    except ImportError:
        pass
    try:
        import torch_sparse  # noqa: F401  registers torch.ops.torch_sparse.partition
        rowptr = torch.from_numpy(A.indptr.astype(np.int64))
        col = torch.from_numpy(A.indices.astype(np.int64))
        return torch.ops.torch_sparse.partition(rowptr, col, None, int(k), False)
    except (ImportError, RuntimeError, AttributeError):
        raise SystemExit('--landmark metis needs pymetis (pip install pymetis) '
                         'or torch_sparse built with METIS')


def vng_partition(edge_index, mask, n_pool, k, K=10):
    """Loukas variation-neighbourhoods coarsening (needs `pip install graph-coarsening pygsp`)."""
    try:
        from graph_coarsening import coarsen
        from pygsp import graphs
    except ImportError:
        raise SystemExit('--landmark vng needs: pip install pygsp '
                         'git+https://github.com/loukasa/graph-coarsening')
    import scipy
    import scipy.sparse as sp
    for name in ('newaxis', 'sqrt', 'floor', 'ceil', 'zeros', 'ones', 'array', 'inf'):
        if not hasattr(scipy, name):  # old numpy aliases the library still uses
            setattr(scipy, name, getattr(np, name))
    r, c = _pool_edges(edge_index, mask, n_pool)
    W = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n_pool, n_pool))
    W = ((W + W.T) > 0).astype(np.float64).tocsr()
    G = graphs.Graph(W)
    C, _, _, _ = coarsen(G, K=K, r=1.0 - float(k) / n_pool,
                         method='variation_neighborhoods', max_levels=30)
    coo = C.tocoo()
    assign = np.full(n_pool, -1, dtype=np.int64)
    assign[coo.col] = coo.row
    if (assign < 0).any():
        raise SystemExit('vng: some nodes were not assigned to a supernode')
    return torch.tensor(assign, dtype=torch.long)



def generate_landmarks(args, H_pool, y_pool, extra=(), Hc=None, graph=None):
    n_p = int(args.budget)
    if args.landmark in ('random', 'herding', 'kcenter'):
        Hs = H_pool if Hc is None else Hc
        if args.landmark == 'herding':
            idx = herding_select(Hs, n_p)
        elif args.landmark == 'kcenter':
            idx = kcenter_select(Hs, n_p)
        else:
            idx = torch.randperm(len(H_pool), device=H_pool.device)[:n_p]
        Hw = _whiten(H_pool if Hc is None else Hc, getattr(args, 'whiten', 0.0))
        assign = torch.cdist(Hw, Hw[idx]).argmin(1)
        assign[idx] = torch.arange(len(idx), device=assign.device)
        return H_pool[idx], assign, [E[idx] for E in extra]
    Hw = _whiten(H_pool if Hc is None else Hc, getattr(args, 'whiten', 0.0))
    if args.landmark == 'class_kmeans':
        assign, k = torch.zeros(len(H_pool), dtype=torch.long), 0
        for cls in range(args.num_class):
            m = (y_pool == cls)
            assign[m.cpu()] = _assign(Hw[m], args.budget_cla[cls], args.clustering) + k
            k += int(args.budget_cla[cls])
    elif args.landmark in ('metis', 'vng'):
        if graph is None:
            raise SystemExit(f'--landmark {args.landmark} needs the pool graph')
        ei, mask = graph
        part = metis_partition if args.landmark == 'metis' else vng_partition
        assign = part(ei, mask, len(H_pool), n_p)
        k = int(assign.max()) + 1
        print(f'landmark[{args.landmark}]: {k} structural cells (budget {n_p})')
    else:
        method = 'nocluster' if args.landmark == 'random_split' else args.clustering
        assign, k = _assign(Hw, n_p, method), n_p
    h, remap = _cluster_means(H_pool, assign, k)
    return h, remap, [_cluster_means(E, assign, k)[0] for E in extra]


def refine_landmarks(h, h_d, H_pool, W, feat_mode, n_p, mode):
    p = F.softmax(label_feats(feat_mode, h_d).double() @ W, dim=1).clamp_min(1e-12)
    ent = -(p * p.log()).sum(1)
    keep = ent.argsort(descending=(mode == 'hard'))[:n_p].sort()[0]
    h = h[keep]
    return (h, torch.cdist(H_pool, h).argmin(1), [E[keep] for E in h_d],
            ent[keep].mean().item(), ent.mean().item())


def label_feats(mode, mats):
    if mode == 'first':
        return mats[0]
    if mode == 'last':
        return mats[-1]
    if mode == 'mean':
        return sum(mats) / len(mats)
    return torch.cat(mats, dim=1)


def onehot_labels(args, Hp, H_L, y_L, assign, y_pool, tr):
    votes = torch.zeros(Hp.shape[0], args.num_class, device=Hp.device)
    if assign is not None:
        a = assign.to(Hp.device)[tr]
        votes.index_put_((a, y_pool[tr]), torch.ones(len(a), device=Hp.device), accumulate=True)
    centroid = torch.stack([H_L[y_L == k].mean(0) for k in range(args.num_class)])
    fallback = (F.normalize(Hp, dim=1) @ F.normalize(centroid, dim=1).T).argmax(1)
    empty = votes.sum(1) == 0
    return torch.where(empty, fallback, votes.argmax(1)).long()


def _fit_logistic(M, y, g, n_p, c, steps, prior=None, init=None):
    ref = 0.0 if prior is None else prior
    Y = (init.clone() if init is not None else
         (torch.zeros(n_p, c, dtype=M.dtype, device=M.device) if prior is None
          else prior.clone())).requires_grad_(True)
    opt = torch.optim.LBFGS([Y], max_iter=steps, history_size=20, tolerance_grad=1e-10,
                            tolerance_change=1e-14, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(M @ Y, y) + 0.5 * g * ((Y - ref) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.enable_grad():
        loss = closure()
    return Y.detach(), loss.item(), Y.grad.norm().item()


def cluster_prior(assign, tr, y_pool, n_p, c, dtype, device, eps=1e-3):
    cnt = torch.zeros(n_p, c, dtype=dtype, device=device)
    if assign is not None:
        a = assign.to(device)[tr]
        cnt.index_put_((a, y_pool[tr].to(device)),
                       torch.ones(int(tr.sum()), dtype=dtype, device=device), accumulate=True)
    glob = F.one_hot(y_pool[tr], c).to(dtype).mean(0).to(device)
    tot = cnt.sum(1, keepdim=True)
    p = torch.where(tot > 0, cnt / tot.clamp(min=1.0), glob.expand(n_p, c))
    p = ((p + eps) / (1 + c * eps)).log()
    return p - p.mean(1, keepdim=True)


def solve_labels_logistic(H_L, Hp, Y_L, beta, gamma, steps=200, prior=None, kind='linear',
                          target_maxp=0.0, iters=12, pool=None, assign=None, n_cl=0,
                          sel=None):
    H_L, Hp, Y_L = H_L.double(), Hp.double(), Y_L.double()
    m, n_p, c = H_L.shape[0], Hp.shape[0], Y_L.shape[1]
    n_cl = n_cl or n_p
    M, rank = _design(H_L, Hp, beta, n_p, kind)
    ref = (M * M).sum() / (m * n_p)
    Mp = None if pool is None else _design(pool.double(), Hp, beta, n_p, kind)[0]

    def read(Y):
        if Mp is None:
            return F.softmax(Y, dim=1)
        return _pool_means(F.softmax(Mp @ Y, dim=1), assign, n_cl, sel)

    def fit(gm, st, init=None):
        return _fit_logistic(M, Y_L, gm * ref, n_p, c, st, prior, init)

    warm = None
    if target_maxp > 0:
        lo, hi = 1e-8, 1e4
        for _ in range(iters):
            mid = (lo * hi) ** 0.5
            warm = fit(mid, max(steps // 4, 30), warm)[0]
            if read(warm).max(1)[0].mean() > target_maxp:
                lo = mid
            else:
                hi = mid
        gamma = (lo * hi) ** 0.5

    Y, loss, gnorm = fit(gamma, steps, warm)
    ctx = {'loss': loss, 'gnorm': gnorm, 'rank': rank, 'gamma': float(gamma * ref),
           'gamma_rel': float(gamma), 'dual': (Y, beta, kind)}
    return read(Y), ctx


def _cluster_softmax(u, assign, n_p):
    mx = torch.full((n_p,), float('-inf'), dtype=u.dtype, device=u.device)
    mx = mx.scatter_reduce(0, assign, u, 'amax', include_self=False)
    mx = torch.where(mx.isinf(), torch.zeros_like(mx), mx)
    e = (u - mx[assign]).exp()
    s = torch.zeros(n_p, dtype=u.dtype, device=u.device).index_add_(0, assign, e)
    return e / s[assign].clamp_min(1e-30)


def _pool_means(P, assign, n_p, sel=None):
    z = lambda k: torch.zeros(k, dtype=P.dtype, device=P.device)
    one = torch.ones(len(P), dtype=P.dtype, device=P.device)
    if sel is None:
        return (z(n_p * P.shape[1]).view(n_p, -1).index_add_(0, assign, P)
                / z(n_p).index_add_(0, assign, one).clamp_min(1e-30).unsqueeze(1))
    w = sel.to(P.dtype).to(P.device)
    acc = z(n_p * P.shape[1]).view(n_p, -1).index_add_(0, assign, w.unsqueeze(1) * P)
    cnt = z(n_p).index_add_(0, assign, w)
    aa = z(n_p * P.shape[1]).view(n_p, -1).index_add_(0, assign, P)
    cc = z(n_p).index_add_(0, assign, one)
    e = (cnt == 0).unsqueeze(1)
    return torch.where(e, aa, acc) / torch.where(e, cc.unsqueeze(1), cnt.unsqueeze(1))


def tangent_stats(H_pool, P, assign, n_p, rank=0, code=0):
    H, P = H_pool.double(), P.double()
    assign = assign.to(H.device)
    hb, pb = _pool_means(H, assign, n_p), _pool_means(P, assign, n_p)
    order = assign.argsort()
    bounds = torch.searchsorted(assign[order], torch.arange(n_p + 1, device=H.device))
    cells = [order[bounds[j]:bounds[j + 1]] for j in range(n_p)]

    U = torch.zeros(n_p, H.shape[1], dtype=H.dtype, device=H.device)
    for j, idx in enumerate(cells):
        if len(idx) >= 2:
            U[j] = torch.linalg.svd(H[idx] - hb[j], full_matrices=False)[2][0]
    sv = torch.linalg.svdvals(U)
    energy = (sv ** 2).cumsum(0) / (sv ** 2).sum().clamp_min(1e-30)

    G = torch.zeros(n_p, P.shape[1], dtype=H.dtype, device=H.device)
    S = torch.zeros(n_p, dtype=H.dtype, device=H.device)
    if code > 0:
        B = torch.linalg.svd(U, full_matrices=False)[2][:code]
        for j, idx in enumerate(cells):
            if len(idx) < 2:
                continue
            Sc = (H[idx] - hb[j]) @ B.T
            ss = (Sc * Sc).sum(0).clamp_min(1e-30)
            Gk = (Sc.T @ (P[idx] - pb[j])) / ss.unsqueeze(1)
            k = (Sc.std(0) * Gk.norm(dim=1)).argmax()
            U[j], S[j], G[j] = B[k], Sc[:, k].std(), Gk[k]
        return U.float(), G.float(), S.float(), energy

    if rank > 0:
        B = torch.linalg.svd(U, full_matrices=False)[2][:rank]
        U = (U @ B.T) @ B
        U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)
    for j, idx in enumerate(cells):
        if len(idx) < 2:
            continue
        sc = (H[idx] - hb[j]) @ U[j]
        ss = (sc * sc).sum()
        if ss > 0:
            S[j] = sc.std()
            G[j] = (sc.unsqueeze(1) * (P[idx] - pb[j])).sum(0) / ss
    return U.float(), G.float(), S.float(), energy


def tangent_adj(h, U, S, k=2, T=1.0):
    n = h.shape[0]
    A = torch.zeros(n, n, dtype=h.dtype, device=h.device)
    rows = torch.arange(n, device=h.device).unsqueeze(1).expand(n, k)
    for sgn in (1.0, -1.0):
        d = torch.cdist(h + sgn * S.unsqueeze(1) * U, h)
        d.fill_diagonal_(float('inf'))
        val, idx = d.topk(k, dim=1, largest=False)
        ok = val <= T * S.unsqueeze(1)
        A[rows[ok], idx[ok]] = 1.0
    A = torch.maximum(A, A.T)
    A.fill_diagonal_(0)
    return A


def relabel_shifted(pf, assign, n_p, delta, ctx, basis, sel=None):
    assign = assign.to(pf.device)
    Z = pf.double() + delta.double().to(pf.device)[assign]
    logits = (Z @ ctx['W']) if 'W' in ctx else dual_logits(Z, basis, ctx['dual'])
    return _pool_means(F.softmax(logits, dim=1), assign, n_p, sel)


def cell_diag(P, assign, n_p, tag=''):
    """Within-cell variance of the teacher posterior and the cell-size distribution."""
    P = P.double()
    assign = assign.to(P.device)
    Y = _pool_means(P, assign, n_p)
    cnt = torch.bincount(assign, minlength=n_p).double()
    keep = cnt > 0
    remap = torch.full((n_p,), -1, dtype=torch.long, device=P.device)
    remap[keep] = torch.arange(int(keep.sum()), device=P.device)
    resid = ((P - Y[remap[assign]]) ** 2).sum(1)
    tot = ((P - P.mean(0)) ** 2).sum(1).mean()
    c = cnt[keep].sort()[0]
    q = lambda f: int(c[min(int(f * (len(c) - 1)), len(c) - 1)])
    small = (cnt[keep] < 10).sum().item()
    print(f'cell diag{tag}: within-var {resid.mean():.4f} ({100 * resid.mean() / tot:.1f}% of total)  '
          f'cells {int(keep.sum())}  size min/p10/med/p90/max {q(0)}/{q(.1)}/{q(.5)}/{q(.9)}/{q(1)}  '
          f'cells<10: {small}  pool in largest {100 * c[-1] / cnt.sum():.1f}%')
    return resid.mean().item()


def teacher_mean_labels(P, assign, n_p, sel=None):
    return _pool_means(P.double(), assign.to(P.device), n_p, sel)


def knn_cells(centres, H_pool, K, chunk=64):
    idx = []
    for i in range(0, len(centres), chunk):
        d = torch.cdist(centres[i:i + chunk].to(H_pool.dtype), H_pool)
        idx.append(d.topk(K, dim=1, largest=False)[1])
    return torch.cat(idx)


def knn_means(X, idx):
    return X[idx.to(X.device)].mean(1)


def cell_std(H_pool, assign, n_p):
    H = H_pool.double()
    a = assign.to(H.device)
    m1, m2 = _pool_means(H, a, n_p), _pool_means(H * H, a, n_p)
    return (m2 - m1 * m1).clamp_min(0).sqrt().float()


def fit_mlp_teacher(H_L, Y_L, hidden=256, epochs=500, lr=1e-2, wd=5e-4, dropout=0.5, seed=0):
    torch.manual_seed(seed)
    d, c = H_L.shape[1], Y_L.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                              torch.nn.Linear(hidden, c)).to(H_L.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    X, T = H_L.float(), Y_L.float()
    for ep in range(epochs):
        if ep == epochs // 2:
            for g in opt.param_groups: g['lr'] = lr * 0.1
        net.train(); opt.zero_grad()
        F.cross_entropy(net(X), T).backward(); opt.step()
    net.eval()
    return lambda Z: net(Z.float())


def mean_entropy(P):
    P = P.clamp_min(1e-12)
    return float(-(P * P.log()).sum(1).mean())


def match_entropy(P, target, iters=40):
    logp = P.clamp_min(1e-12).log()
    lo, hi = 1e-3, 1e3
    for _ in range(iters):
        T = (lo * hi) ** 0.5
        if mean_entropy(F.softmax(logp / T, dim=1)) > target:
            hi = T
        else:
            lo = T
    T = (lo * hi) ** 0.5
    return F.softmax(logp / T, dim=1), T


def _wmean(pi, X, assign, n_p):
    return torch.zeros(n_p, X.shape[1], dtype=X.dtype, device=X.device).index_add_(
        0, assign, pi.unsqueeze(1) * X)


def solve_labels_weighted(H_L, Y_L, pf, P, assign, n_p, beta, kind, steps, lr, mu,
                          batch, seed, extra=(), target='both'):
    H_L, Y_L, pf, P = H_L.double(), Y_L.double(), pf.double(), P.double()
    assign = assign.to(pf.device)
    u = torch.zeros(len(pf), dtype=pf.dtype, device=pf.device, requires_grad=True)
    opt = torch.optim.Adam([u], lr=lr)
    g = torch.Generator(); g.manual_seed(seed)

    def split(v):
        pi = _cluster_softmax(v, assign, n_p)
        pu = _cluster_softmax(torch.zeros_like(v), assign, n_p)
        return (pu if target == 'label' else pi), (pu if target == 'feat' else pi)

    def forward(idx):
        px, py = split(u)
        Xp, Yp = _wmean(px, pf, assign, n_p), _wmean(py, P, assign, n_p)
        M, rank = _design(H_L[idx], Xp, beta, n_p, kind)
        return (F.cross_entropy(M @ Yp, Y_L[idx]) + 0.5 * mu * (u ** 2).mean(), Yp, rank)

    every = max(steps // 6, 1)
    for t in range(steps):
        idx = (torch.randperm(len(H_L), generator=g)[:batch].to(H_L.device)
               if 0 < batch < len(H_L) else slice(None))
        opt.zero_grad()
        loss, Yp, _ = forward(idx)
        loss.backward()
        opt.step()
        if t % every == 0 or t == steps - 1:
            print(f'  [w] step {t:4d}  loss {loss.item():.4f}  '
                  f'maxp {Yp.max(1)[0].mean().item():.4f}  '
                  f'pi_max {_cluster_softmax(u.detach(), assign, n_p).max().item():.2e}')

    with torch.no_grad():
        px, _ = split(u)
        loss, Yp, rank = forward(slice(None))
    ctx = {'loss': loss.item(), 'gnorm': 0.0, 'rank': rank}
    return Yp, ctx, [_wmean(px, E.double(), assign, n_p).to(E.dtype) for E in extra]


def dual_logits(H, Hp, dual):
    Y, beta, kind = dual
    return _design(H.double(), Hp.double(), beta, Hp.shape[0], kind)[0] @ Y


def dual_predictor(Hp, dual):
    Y, beta, kind = dual
    Hp = Hp.double()
    n_p, d = Hp.shape
    bw = None
    if kind == 'erf':
        bw = ((Hp * Hp).sum(1).mean() / d).clamp(min=1e-12)
    if kind == 'rbf':
        D2 = ((Hp * Hp).sum(1, keepdim=True) + (Hp * Hp).sum(1).unsqueeze(0) - 2 * (Hp @ Hp.T))
        bw = D2.clamp(min=0).flatten().median().clamp(min=1e-12)
    if kind == 'linear' and beta <= 0:
        RY = torch.linalg.pinv(Hp) @ Y
    else:
        Kss = _kernel(Hp, Hp, kind, d, bw)
        if beta <= 0:
            RY = torch.linalg.pinv(Kss) @ Y
        else:
            rank = max(int(torch.linalg.matrix_rank(Kss)), 1)
            eye = torch.eye(n_p, dtype=Kss.dtype, device=Kss.device)
            RY = torch.linalg.solve(Kss + beta * (Kss.diagonal().sum() / rank) * eye, Y)
    RY = RY.float(); Hpf = Hp.float(); bwf = None if bw is None else float(bw)
    return lambda x: _kernel(x.float(), Hpf, kind, d, bwf) @ RY


def fit_probe_W(H_L, Y_L, gamma, steps=200, init=None):
    H_L, Y_L = H_L.double(), Y_L.double()
    m, d = H_L.shape
    g = gamma * (H_L * H_L).sum() / (m * d)
    W = (init.clone() if init is not None else
         torch.zeros(d, Y_L.shape[1], dtype=H_L.dtype, device=H_L.device)).requires_grad_(True)
    opt = torch.optim.LBFGS([W], max_iter=steps, history_size=20, tolerance_grad=1e-10,
                            tolerance_change=1e-14, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(H_L @ W, Y_L) + 0.5 * g * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.enable_grad():
        loss = closure()
    return W.detach(), loss.item(), W.grad.norm().item()


def kernel_teacher(H_L, B, Y_L, gamma, steps=200, kind='erf', prior='value', bw_mult=1.0):
    """Kernel logistic regression on inducing points B with a single hyperparameter gamma.
    prior='value': features psi(h) = K(h,B) K_BB^{+}  (pseudo-inverse, eigenvalues below 1e-3 of
                   the mean dropped) and penalty gamma * ||V||^2 = gamma * ||f(B)||^2. This is the
                   beta -> 0 limit of logistic_mean, i.e. the same prior it uses, without the ridge knob.
    prior='rkhs':  features phi(h) = K(h,B) L^{-T} (Cholesky) and penalty gamma * ||W||^2 = RKHS norm.
    Both call fit_probe_W, so gamma has the linear-probe semantics. Returns (pred, W, loss, gnorm)."""
    H_L, B, Y_L = H_L.double(), B.double(), Y_L.double()
    d = B.shape[1]
    bw = ((B * B).sum(1).mean() / d).clamp(min=1e-12) if kind == 'erf' else None
    if kind == 'rbf':
        D2 = ((B * B).sum(1, keepdim=True) + (B * B).sum(1).unsqueeze(0) - 2 * (B @ B.T))
        bw = D2.clamp(min=0).flatten().median().clamp(min=1e-12)
    if bw is not None:
        bw = bw * bw_mult          # <1: sharper / more nonlinear, >1: closer to linear
    K_BB = _kernel(B, B, kind, d, bw)
    K_BB = (K_BB + K_BB.T) / 2
    if prior == 'rkhs':
        eye = torch.eye(len(B), dtype=B.dtype, device=B.device)
        L = torch.linalg.cholesky(K_BB + 1e-8 * K_BB.diagonal().mean() * eye)
        T = torch.linalg.solve_triangular(L, eye, upper=False).T          # L^{-T}
    else:
        lam, U = torch.linalg.eigh(K_BB)
        keep = lam > 1e-3 * lam.mean()
        T = U[:, keep] @ torch.diag(1.0 / lam[keep]) @ U[:, keep].T       # K_BB^{+}
    feat = lambda X: _kernel(X.double(), B, kind, d, bw) @ T
    W, loss, gnorm = fit_probe_W(feat(H_L), Y_L, gamma, steps)
    Wf, Bf, Tf, bwf = W.float(), B.float(), T.float(), (None if bw is None else float(bw))
    pred = lambda x: (_kernel(x.float().to(Bf.device), Bf, kind, d, bwf) @ Tf) @ Wf
    return pred, W, loss, gnorm


def solve_labels_kernel(H_L, B, Y_L, gamma, steps, kind, pool, assign, n_cl, sel=None, prior='value',
                        bw_mult=1.0):
    pred, A, loss, gnorm = kernel_teacher(H_L, B, Y_L, gamma, steps, kind, prior, bw_mult)
    P = F.softmax(pred(pool).double(), dim=1)
    Y = _pool_means(P, assign.to(P.device), n_cl, sel)
    ctx = {'loss': loss, 'gnorm': gnorm, 'rank': int(B.shape[0]), 'gamma_rel': float(gamma),
           'kfun': pred, 'A': A}
    return Y, ctx


def fit_probe_W_ridge(H_L, Y_L, gamma):
    H_L, Y_L = H_L.double(), Y_L.double()
    m, d = H_L.shape
    g = gamma * (H_L * H_L).sum() / (m * d)
    G = H_L.T @ H_L
    A = G + g * (G.diagonal().sum() / d) * torch.eye(d, dtype=G.dtype, device=G.device)
    return torch.linalg.solve(A, H_L.T @ Y_L)


def correct_and_smooth(Yhat, S, train_mask, Y_L, a1=0.8, a2=0.8, iters=50, scale=1.0,
                       reset=True):
    Z = Yhat.to(S.dtype)
    E = torch.zeros_like(Z)
    E[train_mask] = Y_L.to(Z.dtype) - Z[train_mask]
    E0 = E.clone()
    for _ in range(iters):
        E = (1 - a1) * E0 + a1 * torch.spmm(S, E)
    Z = Z + scale * E
    if reset:
        Z[train_mask] = Y_L.to(Z.dtype)
    Z0 = Z.clone()
    for _ in range(iters):
        Z = (1 - a2) * Z0 + a2 * torch.spmm(S, Z)
    return Z


def _cs_diff(P, S, inj, Yfull, a1, a2, iters, scale, reset=True):
    m = inj.unsqueeze(1).to(P.dtype)
    E0 = m * (Yfull - P)
    E = E0
    for _ in range(iters):
        E = (1 - a1) * E0 + a1 * torch.sparse.mm(S, E)
    Z = P + scale * E
    if reset:
        Z = torch.where(inj.unsqueeze(1), Yfull, Z)
    Z0 = Z
    for _ in range(iters):
        Z = (1 - a2) * Z0 + a2 * torch.sparse.mm(S, Z)
    return Z


def solve_labels_cs_loss(H_all, Hp, Y_all, train_mask, S, beta, gamma, folds=2,
                         a1=0.8, a2=0.8, iters=20, scale=1.0, steps=100, seed=0,
                         kind='linear', reset=True):
    """min_Y'  sum_f CE( CS_{train}(softmax(M_all Y'))[f], Y[f] ) + (g/2)||Y'||^2"""
    H_all, Hp, Y_all = H_all.float(), Hp.float(), Y_all.float()
    n, n_p, c = H_all.shape[0], Hp.shape[0], Y_all.shape[1]
    M, rank = _design(H_all, Hp, beta, n_p, kind)
    M = M.float()
    g = gamma * (M * M).sum() / (n * n_p)

    idx = train_mask.nonzero().view(-1)
    perm = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed))]
    packs = []
    for f in perm.chunk(folds):
        inj = train_mask.clone()
        inj[f] = False
        Yf = torch.zeros(n, c, dtype=M.dtype, device=M.device)
        Yf[inj] = Y_all[inj]
        packs.append((inj, Yf, f, Y_all[f]))

    Y = torch.zeros(n_p, c, dtype=M.dtype, device=M.device, requires_grad=True)
    opt = torch.optim.LBFGS([Y], max_iter=steps, history_size=10, tolerance_grad=1e-8,
                            tolerance_change=1e-12, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        P = F.softmax(M @ Y, dim=1)
        loss = 0.0
        for inj, Yf, f, tgt in packs:
            Z = _cs_diff(P, S, inj, Yf, a1, a2, iters, scale, reset)
            Z = Z.clamp(min=1e-9)
            Z = Z / Z.sum(1, keepdim=True)
            loss = loss - (tgt * Z[f].log()).sum(1).mean()
        loss = loss / len(packs) + 0.5 * g * (Y ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.enable_grad():
        loss = closure()
    ctx = {'loss': loss.item(), 'gnorm': Y.grad.norm().item(), 'rank': rank,
           'gamma_rel': float(gamma)}
    return F.softmax(Y.detach(), dim=1).double(), ctx


def solve_labels_cs(H_all, H_L, Y_L, gamma, S, train_mask, pool_mask_, assign, n_p,
                    a1=0.8, a2=0.8, iters=50, scale=1.0, steps=200, reset=True):
    if reset and bool(train_mask.all()):
        raise SystemExit('cs: train_mask covers every node (inductive split); the reset step Z[train]=Y_L overwrites the whole base prediction and cs degenerates to plain label propagation. Use --cs_reset 0 or a different --label_mode.')
    W, _, _ = fit_probe_W(H_L, Y_L, gamma, steps)
    Yhat = F.softmax(H_all.double() @ W, dim=1)
    Z = correct_and_smooth(Yhat, S, train_mask, Y_L, a1, a2, iters, scale, reset).double()
    Z = Z.clamp(min=0)
    Z = Z / Z.sum(1, keepdim=True).clamp(min=1e-12)
    Y = _cluster_means(Z[pool_mask_], assign, n_p)[0]
    ctx = {'loss': float('nan'), 'gnorm': 0.0, 'gamma_rel': float(gamma),
           'rank': n_p, 'node_pred': Z}
    return Y, ctx


def solve_labels_restricted(H_L, Hp, Y_L, gamma, steps=200, target_maxp=0.0, iters=6,
                            pool=None, assign=None, tol=1e-10):
    """CE(H_L W, y) + (g/2)||W||^2  s.t.  W in rowspace(H').  No beta."""
    H_L, Hp, Y_L = H_L.double(), Hp.double(), Y_L.double()
    m, d = H_L.shape
    g = gamma * (H_L * H_L).sum() / (m * d)
    S, Vh = torch.linalg.svd(Hp, full_matrices=False)[1:]
    V = Vh[S > tol * S[0]].T
    HLv, Hpv = H_L @ V, Hp @ V
    poolv = None if pool is None else pool.double() @ V
    n_p = Hp.shape[0]

    def read(Z):
        if poolv is None:
            return F.softmax(Hpv @ Z, dim=1)
        return _cluster_means(F.softmax(poolv @ Z, dim=1), assign, n_p)[0]

    def fit(gm, st, init=None):
        Z = (init.clone() if init is not None else
             torch.zeros(V.shape[1], Y_L.shape[1], dtype=H_L.dtype,
                         device=H_L.device)).requires_grad_(True)
        opt = torch.optim.LBFGS([Z], max_iter=st, history_size=20, tolerance_grad=1e-10,
                                tolerance_change=1e-14, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(HLv @ Z, Y_L) + 0.5 * gm * (Z ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.enable_grad():
            loss = closure()
        return Z.detach(), loss.item(), Z.grad.norm().item()

    warm = None
    if target_maxp > 0:
        lo, hi = 1e-8, 1e4
        for _ in range(iters):
            mid = (lo * hi) ** 0.5
            warm = fit(mid * (g / max(gamma, 1e-30)), max(steps // 4, 30), warm)[0]
            if read(warm).max(1)[0].mean() > target_maxp:
                lo = mid
            else:
                hi = mid
        gamma = (lo * hi) ** 0.5
        g = gamma * (H_L * H_L).sum() / (m * d)

    Z, loss, gnorm = fit(g, steps, warm)
    ctx = {'loss': loss, 'gnorm': gnorm, 'rank': V.shape[1], 'gamma_rel': float(gamma),
           'W': V @ Z}
    return read(Z), ctx


def solve_labels_ridge(H_L, Hp, Y_L, gamma, pool=None, assign=None):
    W = fit_probe_W_ridge(H_L, Y_L, gamma)
    n_p = Hp.shape[0]
    Y = (Hp.double() @ W if pool is None
         else _cluster_means(pool.double() @ W, assign, n_p)[0])
    ctx = {'loss': float('nan'), 'gnorm': 0.0, 'gamma_rel': float(gamma),
           'rank': int(torch.linalg.matrix_rank(Hp)), 'W': W}
    return Y, ctx


def teacher_targets_ridge(H_fit, H_L, Y_L, gamma):
    return H_fit.double() @ fit_probe_W_ridge(H_L, Y_L, gamma)


def teacher_targets(H_fit, H_L, Y_L, gamma, temp, steps=200, folds=0, tr_in_fit=None, seed=0):
    H_fit, H_L, Y_L = H_fit.double(), H_L.double(), Y_L.double()
    t = max(temp, 1e-6)
    W, _, _ = fit_probe_W(H_L, Y_L, gamma, steps)
    T = F.softmax(H_fit @ W / t, dim=1)
    if folds and folds > 1 and tr_in_fit is not None:
        rows = tr_in_fit.nonzero().view(-1).to(T.device)
        g = torch.Generator().manual_seed(seed)
        for f in torch.randperm(len(H_L), generator=g).chunk(folds):
            keep = torch.ones(len(H_L), dtype=torch.bool)
            keep[f] = False
            Wf, _, _ = fit_probe_W(H_L[keep], Y_L[keep], gamma, steps)
            T[rows[f.to(rows.device)]] = F.softmax(H_L[f] @ Wf / t, dim=1)
    return T


def solve_labels_probe(H_L, Hp, Y_L, gamma, steps=200, target_maxp=0.0, iters=12,
                       pool=None, assign=None, sel=None):
    H_L, Hp, Y_L = H_L.double(), Hp.double(), Y_L.double()
    n_p = Hp.shape[0]

    def read(W):
        if pool is None:
            return F.softmax(Hp @ W, dim=1)
        return _pool_means(F.softmax(pool.double() @ W, dim=1), assign, n_p, sel)

    warm = None
    if target_maxp > 0:
        lo, hi = 1e-8, 1e4
        for _ in range(iters):
            mid = (lo * hi) ** 0.5
            warm = fit_probe_W(H_L, Y_L, mid, max(steps // 4, 30), warm)[0]
            if read(warm).max(1)[0].mean() > target_maxp:
                lo = mid
            else:
                hi = mid
        gamma = (lo * hi) ** 0.5

    W, loss, gnorm = fit_probe_W(H_L, Y_L, gamma, steps, warm)
    ctx = {'loss': loss, 'gnorm': gnorm, 'rank': int(torch.linalg.matrix_rank(Hp)),
           'gamma_rel': float(gamma), 'W': W}
    return read(W), ctx
