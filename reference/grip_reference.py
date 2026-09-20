"""GRIP - the condensation method as it stood on branch `ablation-archive` (tag ablation-archive-2026-09-20),
reduced to the path that produced the main table (protocol A). Reference only: every function below is copied from
scr/label_solve.py / scr/utils.py of that branch with the ablation branches removed, so it can be read top-to-bottom
and re-implemented against.

Pipeline (transductive graph; inductive = same with the training graph as the pool)
  1. features      H_k = Â^k X, k = 0..2, Â = D^-1/2 (A + I) D^-1/2          -> pool space H = H_2 (all nodes), depth 2 fixed
  2. budget        m = sum_c max(int(n_c * rate), 1) over the training classes (rate from ratio_transfer), same as CGC
  3. teacher       NNGP kernel (relu1 or erf) on H; Nystrom basis B = 3000 random pool rows; RKHS features
                   psi(h) = K(h, B) L^-T  (K_BB = L L^T); kernel logistic regression on the training nodes with
                   penalty 0.5 * g ||W||^2,  g = gamma * ||Psi_L||_F^2 / (n_L * dim)   (L-BFGS, 1000 iters, tol 1e-6)
                   P0 = softmax(psi(H) W) on the pool, then temperature T:  P = softmax(log P0 / T)
  4. partition     k-means init (faiss) on H with k = m, then Lloyd sweeps of
                       sum_j sum_{t in C_j} ||h_t - c_j|| / s  +  mu * KL(p_t || ybar_j)
                   c_j = geometric median (Weiszfeld), ybar_j = cell mean of P, s = mean distance of the init partition
  5. condensed set x'_j = geometric median of cell j (in H), y'_j = cell mean of P (soft),  A' = I;  empty cells dropped
  6. student       2-layer GCN (hidden 256) on (x', y', I) with soft cross-entropy, Adam lr 0.01 (x0.1 at epoch 500),
                   1000 epochs, dropout / weight decay from the grid, evaluated on the ORIGINAL graph every 10 epochs,
                   test accuracy at the best-validation epoch, mean over repeats

Protocol A (main table): fixed = raw space H_2, basis 3000 random, depth 2, lr 0.01, hidden 256, 1000 epochs.
Selected on validation = kernel {relu1, erf}, gamma, feat_norm {0, 1}, T {1, 0.5, 0.25}, mu {0.2 .. 5}, dropout, wd.
Settings behind the table (val-best, repeat 10 unless noted):
  cora     1.3 / 2.6 / 5.2 %  relu1 fn0  gamma 0.1 / 0.1 / 0.01  T 1 / 2 / 1  mu 1 / 1 / 2   do 0.9  wd 5e-4 (5.2%: 5e-3)
  citeseer 0.9 / 1.8 / 3.6 %  erf   fn1  gamma 10 / 10 / 3       T 0.25       mu 0.2         do 0.3 / 0.1 / 0.3
  arxiv    0.05 / 0.25 / 0.5 %  erf / relu1 / erf  gamma 1e-3 / 1e-4 / 1e-3  T 0.25 / 0.25 / 1  mu 0.5 / 0.2 / 0.5  do 0.7 / 0.3 / 0.1
Narrowed protocol proposed on 2026-09-20 (not yet run): one condensation config per DATASET chosen by the pooled val over
the three densities from kernel x gamma(3) x fn x T {1, 0.25} x mu {0.2, 1}; downstream wd 5e-4, dropout {0.1, 0.5, 0.9} on val.
"""
import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# ----------------------------------------------------------------------------------------------------------------------
# 1. propagated features
# ----------------------------------------------------------------------------------------------------------------------
def normalize_adj_sparse(data):
    """D^-1/2 (A + I) D^-1/2 as a torch sparse tensor (self-loops added when the diagonal is empty)."""
    n = data.x.shape[0]
    mx = sp.csr_matrix((np.ones(data.edge_index.shape[1]), data.edge_index.cpu().numpy()), shape=(n, n)).tolil()
    if mx[0, 0] == 0:
        mx = mx + sp.eye(n)
    r_inv = np.power(np.array(mx.sum(1)), -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    mx = sp.diags(r_inv).dot(mx).dot(sp.diags(r_inv)).tocoo().astype(np.float32)
    idx = torch.LongTensor(np.vstack([mx.row, mx.col]))
    return torch.sparse.FloatTensor(idx, torch.FloatTensor(mx.data), torch.Size(mx.shape))


def propagate(data, depth=2):
    """[X, ÂX, Â²X, ...]; the pool space is the last one."""
    adj = normalize_adj_sparse(data).to(data.x.device)
    Hs = [data.x]
    for _ in range(depth):
        Hs.append(torch.spmm(adj, Hs[-1]))
    return Hs


# ----------------------------------------------------------------------------------------------------------------------
# 2. budget (CGC's class-proportional allocation of the training-label counts)
# ----------------------------------------------------------------------------------------------------------------------
RATIO_TRANSFER = {('cora', 0.013): 0.25, ('cora', 0.026): 0.5, ('cora', 0.052): 1.0,
                  ('citeseer', 0.009): 0.25, ('citeseer', 0.018): 0.5, ('citeseer', 0.036): 1.0,
                  ('arxiv', 0.0005): 0.001, ('arxiv', 0.0025): 0.005, ('arxiv', 0.005): 0.01}
# flickr / reddit: see ratio_transfer() in scr/utils.py of the archive branch


def budget(y_train, num_class, rate):
    """m and the per-class counts: the smallest classes first, the largest class takes the remainder (= CGC generate_labels_syn)."""
    from collections import Counter
    counter = Counter(y_train.cpu().numpy().tolist()); n = len(y_train)
    per_class, acc = {}, 0
    items = sorted(counter.items(), key=lambda x: x[1])
    for ix, (c, num) in enumerate(items):
        if ix == len(items) - 1:
            per_class[c] = int(n * rate) - acc
        else:
            per_class[c] = max(int(num * rate), 1); acc += per_class[c]
    counts = np.array([per_class[c] for c in range(num_class)], dtype=int)
    return int(counts.sum()), counts


# ----------------------------------------------------------------------------------------------------------------------
# 3. NNGP kernel teacher (Nystrom RKHS features + kernel logistic regression)
# ----------------------------------------------------------------------------------------------------------------------
def _kernel(A, B, kind, d, bw=None):
    if kind == 'erf':                                    # NNGP of a 1-hidden-layer erf network (Williams 1997)
        S = (A @ B.T) / (d * bw)
        a = (A * A).sum(1, keepdim=True) / (d * bw)
        b = (B * B).sum(1).unsqueeze(0) / (d * bw)
        r = 2 * S / torch.sqrt((1 + 2 * a) * (1 + 2 * b))
        return (2 / math.pi) * torch.asin(r.clamp(-1 + 1e-12, 1 - 1e-12))
    if kind.startswith('relu'):                          # NNGP of an L-hidden-layer ReLU MLP (arc-cosine order 1, composed)
        L = int(kind[4:] or 1)
        na = A.norm(dim=1, keepdim=True).clamp(min=1e-12)
        nb = B.norm(dim=1).unsqueeze(0).clamp(min=1e-12)
        cos = ((A @ B.T) / (na * nb)).clamp(-1 + 1e-12, 1 - 1e-12)
        scale = (na * nb) / d
        for _ in range(L):
            th = torch.acos(cos)
            cos = ((torch.sin(th) + (math.pi - th) * torch.cos(th)) / math.pi).clamp(-1 + 1e-12, 1 - 1e-12)
        return scale * cos
    raise ValueError(kind)


def _bandwidth(B, kind):
    """erf: bw = mean squared feature scale per dimension (so the kernel argument is scale free); relu: none."""
    d = B.shape[1]
    return ((B * B).sum(1).mean() / d).clamp(min=1e-12) if kind == 'erf' else None


def _chunked(fn, X, chunk):
    if len(X) <= chunk:
        return fn(X)
    first = fn(X[:chunk])
    out = torch.empty(len(X), *first.shape[1:], dtype=first.dtype, device=first.device)
    out[:chunk] = first
    for i in range(chunk, len(X), chunk):
        out[i:i + chunk] = fn(X[i:i + chunk])
    return out


def pick_basis(H, m, seed=0):
    """m uniformly sampled pool rows (H itself when m >= N, i.e. cora / citeseer at basis 3000)."""
    if m <= 0 or m >= len(H):
        return H
    g = torch.Generator(); g.manual_seed(seed)
    return H[torch.randperm(len(H), generator=g)[:m].to(H.device)]


def fit_probe_W(Psi_L, Y_L, gamma, steps=1000, tol=1e-6):
    """Multinomial logistic regression on features Psi_L with penalty 0.5 * g ||W||^2, g = gamma ||Psi_L||^2 / (n d)
    (gamma is scale free). L-BFGS with strong-Wolfe line search."""
    Psi_L, Y_L = Psi_L.double(), Y_L.double()
    m, d = Psi_L.shape
    g = gamma * Psi_L.norm() ** 2 / (m * d)
    W = torch.zeros(d, Y_L.shape[1], dtype=Psi_L.dtype, device=Psi_L.device).requires_grad_(True)
    opt = torch.optim.LBFGS([W], max_iter=steps, history_size=20, tolerance_grad=tol,
                            tolerance_change=tol ** 1.4, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Psi_L @ W, Y_L) + 0.5 * g * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach()


def kernel_teacher(H_L, B, Y_L, gamma, kind, steps=1000, tol=1e-6, chunk=8192):
    """RKHS-prior kernel logistic regression: features psi(h) = K(h, B) L^-T with K_BB = L L^T, so ||W||^2 is the RKHS norm
    of f = psi(.) W. Returns pred(X) -> logits."""
    H_L, B, Y_L = H_L.double(), B.double(), Y_L.double()
    d, bw = B.shape[1], _bandwidth(B, kind)
    K_BB = _kernel(B, B, kind, d, bw); K_BB = (K_BB + K_BB.T) / 2
    eye = torch.eye(len(B), dtype=B.dtype, device=B.device)
    L = torch.linalg.cholesky(K_BB + 1e-8 * K_BB.diagonal().mean() * eye)
    T = torch.linalg.solve_triangular(L, eye, upper=False).T                     # L^-T
    feat = lambda X: _chunked(lambda x: _kernel(x.double(), B, kind, d, bw) @ T, X, chunk)
    W = fit_probe_W(feat(H_L), Y_L, gamma, steps, tol)
    return lambda X: feat(X) @ W


def teacher_posteriors(H_pool, train_mask, y, num_class, kind, gamma, basis=3000, temp=1.0, seed=0):
    """P (pool x C): tempered posteriors of the kernel teacher fitted on the training nodes of the pool."""
    B = pick_basis(H_pool, basis, seed)
    Y_L = F.one_hot(y[train_mask], num_class).to(H_pool.dtype)
    pred = kernel_teacher(H_pool[train_mask], B, Y_L, gamma, kind)
    P = F.softmax(pred(H_pool), dim=1)
    if temp != 1.0:
        P = F.softmax(P.clamp_min(1e-12).log() / temp, dim=1)
    return P


# ----------------------------------------------------------------------------------------------------------------------
# 4. partition: k-means init, then Lloyd sweeps of the l1 + KL objective with geometric-median centres
# ----------------------------------------------------------------------------------------------------------------------
def kmeans_init(H, k):
    import faiss
    km = faiss.Kmeans(int(H.shape[1]), int(k), gpu=False)
    km.cp.min_points_per_centroid = 1
    X = H.detach().cpu().numpy().astype('float32')
    km.train(X)
    return torch.from_numpy(km.index.search(X, 1)[1].flatten()).long()


def _weiszfeld(H, assign, k, iters=30, eps=1e-9):
    """Geometric median of every cell (argmin_c sum_t ||h_t - c||) by Weiszfeld iterations from the mean.
    Returns the medians and the per-node weights 1/||h_t - c_j|| normalised within each cell (the median is their weighted mean)."""
    dt, dev = H.dtype, H.device
    cnt = torch.bincount(assign, minlength=k).clamp_min(1).to(dt)
    c = torch.zeros(k, H.shape[1], dtype=dt, device=dev).index_add_(0, assign, H) / cnt.unsqueeze(1)
    for _ in range(iters):
        w = 1.0 / (H - c[assign]).norm(dim=1).clamp_min(eps)
        num = torch.zeros(k, H.shape[1], dtype=dt, device=dev).index_add_(0, assign, w.unsqueeze(1) * H)
        den = torch.zeros(k, dtype=dt, device=dev).index_add_(0, assign, w).clamp_min(eps)
        c = num / den.unsqueeze(1)
    w = 1.0 / (H - c[assign]).norm(dim=1).clamp_min(eps)
    den = torch.zeros(k, dtype=dt, device=dev).index_add_(0, assign, w).clamp_min(eps)
    return c, w / den[assign]


def l1_assign(H, P, assign, k, mu, iters=20, chunk=8192, w_iters=30):
    """Lloyd iterations for  sum_j sum_{t in C_j} ||h_t - c_j|| / s  +  mu * KL(p_t || ybar_j).
    The unsquared distance is what the Lipschitz bound produces, so the centre is the geometric median; the label centre is
    the plain cell mean. s = mean distance to the medians of the initial partition (makes mu scale free).
    Returns (assign, medians, per-node Weiszfeld weights)."""
    H = H.double(); assign = assign.clone().to(H.device); N = len(H)
    use_lab = P is not None and mu > 0
    if use_lab:
        P = P.double().clamp_min(1e-12); ent = (P * P.log()).sum(1)

    def centres(a):
        c, w = _weiszfeld(H, a, k, w_iters)
        if not use_lab:
            return c, None, w
        cnt = torch.bincount(a, minlength=k).clamp_min(1).to(H.dtype).unsqueeze(1)
        yc = torch.zeros(k, P.shape[1], dtype=H.dtype, device=H.device).index_add_(0, a, P) / cnt
        return c, yc.clamp_min(1e-12), w

    hc, yc, w = centres(assign)
    s = (H - hc[assign]).norm(dim=1).mean().clamp_min(1e-12)
    for _ in range(iters):
        new = torch.empty_like(assign)
        logy = yc.log() if use_lab else None
        for i in range(0, N, chunk):
            cost = torch.cdist(H[i:i + chunk], hc) / s                                   # b x k, unsquared
            if use_lab:
                cost = cost + mu * (ent[i:i + chunk].unsqueeze(1) - P[i:i + chunk] @ logy.T)   # KL(p_t || ybar_j)
            new[i:i + chunk] = cost.argmin(1)
        moved = int((new != assign).sum())
        assign = new
        hc, yc, w = centres(assign)
        if moved == 0:
            break
    return assign, hc, w


def cell_wmeans(H, assign, k, w):
    """sum_t w_t h_t per cell with the per-cell normalised Weiszfeld weights = the geometric median in H's dtype; empty cells
    are dropped and the assignment is remapped to the surviving cells."""
    dtype, device = H.dtype, H.device
    assign, w = assign.to(device), w.to(device=device, dtype=dtype)
    acc = torch.zeros(k, H.shape[1], dtype=dtype, device=device).index_add_(0, assign, w.unsqueeze(1) * H)
    cnt = torch.zeros(k, dtype=dtype, device=device).index_add_(0, assign, torch.ones(len(H), dtype=dtype, device=device))
    keep = cnt > 0
    remap = torch.full((k,), -1, dtype=torch.long, device=device)
    remap[keep] = torch.arange(int(keep.sum()), device=device)
    return acc[keep], remap[assign]


def cell_means(P, assign, n_cells):
    cnt = torch.zeros(n_cells, dtype=P.dtype, device=P.device).index_add_(0, assign, torch.ones(len(P), dtype=P.dtype, device=P.device))
    return torch.zeros(n_cells, P.shape[1], dtype=P.dtype, device=P.device).index_add_(0, assign, P) / cnt.clamp_min(1e-30).unsqueeze(1)


# ----------------------------------------------------------------------------------------------------------------------
# 5. condensation
# ----------------------------------------------------------------------------------------------------------------------
def condense(H_pool, train_mask, y, num_class, m, kernel, gamma, mu, temp, basis=3000, seed=0):
    """(x', y') with A' = I.  H_pool = Â²X over the pool (all nodes; training graph for inductive data),
    train_mask / y restricted to the pool."""
    P = teacher_posteriors(H_pool, train_mask, y, num_class, kernel, gamma, basis, temp, seed)   # tempered posteriors
    assign = kmeans_init(H_pool, m)
    assign, _, w = l1_assign(H_pool, P, assign, m, mu)
    x_cond, assign = cell_wmeans(H_pool, assign.to(H_pool.device), m, w)                        # geometric medians
    y_cond = cell_means(P.double(), assign.to(P.device), len(x_cond)).float()                   # cell-mean posteriors
    return x_cond, y_cond, assign


# ----------------------------------------------------------------------------------------------------------------------
# 6. student (GCN on the condensed set, evaluated on the original graph at the best-validation epoch)
# ----------------------------------------------------------------------------------------------------------------------
class GCN(torch.nn.Module):
    def __init__(self, nin, nhid, nout, dropout):
        super().__init__()
        self.c1, self.c2, self.dropout = GCNConv(nin, nhid), GCNConv(nhid, nout), dropout

    def forward(self, data):
        x = F.relu(self.c1(data.x, data.edge_index, data.edge_attr))
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(self.c2(x, data.edge_index, data.edge_attr), dim=1)


@torch.no_grad()
def accuracies(model, data):
    model.eval(); out = model(data)
    return [out[m].argmax(1).eq(data.y[m]).float().mean().item() for m in (data.train_mask, data.val_mask, data.test_mask)]


def train_student(x_cond, y_cond, data, hidden=256, dropout=0.5, lr=0.01, wd=5e-4, epochs=1000, eval_every=10, device='cuda'):
    """Soft cross-entropy on (x', y', I); Adam with lr x0.1 at epochs/2 (CGC); returns (best val, test at best val)."""
    n = len(x_cond)
    ei = torch.eye(n).nonzero().t().to(device)
    graph = Data(x=x_cond.to(device), y=y_cond.to(device), edge_index=ei, edge_attr=torch.ones(n, device=device))
    data = data.to(device)
    model = GCN(data.num_features, hidden, y_cond.shape[1], dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val = test_at_best = 0.0
    for epoch in range(1, epochs + 1):
        if epoch == epochs // 2:
            opt = torch.optim.Adam(model.parameters(), lr=lr * 0.1, weight_decay=wd)
        model.train()
        loss = -(graph.y * model(graph)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % eval_every == 0 or epoch == epochs:
            _, val, test = accuracies(model, data)                    # inductive data: val / test graphs instead
            if val > best_val:
                best_val, test_at_best = val, test
    return best_val, test_at_best


# ----------------------------------------------------------------------------------------------------------------------
# 7. one run (dataset loading as in CGC's scr/dataloader.py; feat_norm = NormalizeFeatures for Planetoid,
#    StandardScaler on training statistics for arxiv / flickr / reddit)
# ----------------------------------------------------------------------------------------------------------------------
def run(data, name, ratio, kernel, gamma, mu, temp, dropouts=(0.1, 0.5, 0.9), wd=5e-4, repeat=5, seed=0, device='cuda'):
    torch.manual_seed(seed); np.random.seed(seed)
    num_class = int(data.y.max()) + 1
    H = propagate(data, 2)[-1]                                                            # pool = all nodes
    m, _ = budget(data.y[data.train_mask], num_class, RATIO_TRANSFER[(name, ratio)])
    x_cond, y_cond, assign = condense(H, data.train_mask, data.y, num_class, m, kernel, gamma, mu, temp, seed=seed)
    results = {}
    for do in dropouts:                                                                    # downstream selected on val
        vt = [train_student(x_cond, y_cond, data, dropout=do, wd=wd, device=device) for _ in range(repeat)]
        results[do] = (float(np.mean([v for v, _ in vt])), float(np.mean([t for _, t in vt])), float(np.std([t for _, t in vt], ddof=1)))
    best_do = max(results, key=lambda d: results[d][0])
    return best_do, results[best_do], results
