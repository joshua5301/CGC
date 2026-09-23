"""Q=I condensation for a two-layer nonlinear GCN. See docs/distance_identity.md.

For the actual nonnegative GCN operator A, delta=|A 1-1| and q=A delta+delta,
the feature cost is A^2 [||x_u-c_j||] + q_v ||c_j||. No OT or dense A^2.
Torch float64, representative batching, nonempty descent, weighted medians.
"""
import time
import numpy as np
import torch


def distances(x, z):
    return torch.cdist(x, z, compute_mode='use_mm_for_euclid_dist')


def weighted_medians(x, weights, origin_weight, initial, iters=30, tol=1e-7):
    """Weighted Vardi-Zhang iteration with an extra atom at zero; best iterate kept."""
    z = initial.clone()
    d = distances(x, z)
    r = z.norm(dim=1)
    best_cost = (weights * d).sum(0) + origin_weight * r
    best = z.clone()
    eps = 1e-10
    for _ in range(iters):
        coincident = d <= eps
        w = torch.where(coincident, 0., weights / d.clamp_min(eps))
        wz = torch.where(r <= eps, 0., origin_weight / r.clamp_min(eps))
        total = w.sum(0) + wz
        target = (w.T @ x) / total.clamp_min(eps)[:, None]
        target = torch.where((total > 0)[:, None], target, z)
        eta = (weights * coincident).sum(0) + origin_weight * (r <= eps)
        residual = total * (target - z).norm(dim=1)
        mix = torch.where(residual > eps, (eta / residual.clamp_min(eps)).clamp_max(1), 1.)
        new = (1 - mix[:, None]) * target + mix[:, None] * z
        step = (new - z).norm(dim=1).max()
        z = new
        d, r = distances(x, z), z.norm(dim=1)
        cost = (weights * d).sum(0) + origin_weight * r
        improve = cost < best_cost
        best[improve], best_cost[improve] = z[improve], cost[improve]
        if step <= tol:
            break
    return best


class DistanceIdentity:
    def __init__(self, x, operator, teacher, m, assign=None, mu=1., seed=0,
                 batch_size=32, median_iters=30, tol=1e-6, log=print):
        if not np.isfinite(mu) or mu < 0 or batch_size < 1 or median_iters < 1 or tol < 0:
            raise ValueError('mu/tol must be nonnegative; batch_size/median_iters must be positive')
        self.x = x.detach().to(dtype=torch.float64)
        self.n, self.m = len(x), int(m)
        if not 1 <= self.m <= self.n:
            raise ValueError('m must be between 1 and N')
        self.A = operator.detach().to(device=x.device, dtype=torch.float64).coalesce()
        if self.A.shape != (self.n, self.n) or not torch.isfinite(self.A.values()).all() or (self.A.values() < 0).any():
            raise ValueError('operator must be finite, nonnegative, sparse COO, and N by N')
        if not torch.isfinite(self.x).all():
            raise ValueError('features must be finite')
        self.At = self.A.transpose(0, 1).coalesce()
        delta = (torch.sparse.sum(self.A, dim=1).to_dense() - 1).abs()
        self.q = torch.sparse.mm(self.A, delta[:, None]).squeeze(1) + delta
        f = teacher.detach().to(device=x.device, dtype=torch.float64)
        if f.shape[0] != self.n or not torch.isfinite(f).all() or (f < 0).any() or not torch.allclose(f.sum(1), torch.ones(self.n, device=x.device, dtype=f.dtype), atol=1e-6):
            raise ValueError('teacher must contain probability rows')
        f = f.clamp_min(1e-12)
        self.f = f / f.sum(1, keepdim=True)
        self.entropy = (self.f * self.f.log()).sum(1)
        self.mu, self.batch_size = float(mu), int(batch_size)
        self.median_iters, self.tol, self.log = median_iters, tol, log
        if assign is None:
            from src.partition_struct import kmeans_nonempty
            assign = kmeans_nonempty(self.x.cpu().numpy(), self.m, seed)
        self.assign = torch.as_tensor(assign, dtype=torch.long, device=x.device).clone()
        if self.assign.shape != (self.n,) or self.assign.min() < 0 or self.assign.max() >= self.m:
            raise ValueError('invalid assignment')
        self.counts = torch.bincount(self.assign, minlength=self.m)
        if (self.counts == 0).any():
            raise ValueError('initial cells must be nonempty')
        self.z = torch.zeros(self.m, x.shape[1], device=x.device, dtype=self.x.dtype)
        self.z.index_add_(0, self.assign, self.x)
        self.z /= self.counts[:, None]
        self.update_labels()
        self.history = []

    def batches(self):
        for lo in range(0, self.m, self.batch_size):
            yield lo, min(self.m, lo + self.batch_size)

    def propagate(self, values, transpose=False):
        a = self.At if transpose else self.A
        return torch.sparse.mm(a, torch.sparse.mm(a, values))

    def feature_cost(self, lo, hi):
        z = self.z[lo:hi]
        return self.propagate(distances(self.x, z)) + self.q[:, None] * z.norm(dim=1)[None, :]

    def costs(self, lo, hi):
        feature = self.feature_cost(lo, hi)
        kl = (self.entropy[:, None] - self.f @ self.y[lo:hi].clamp_min(1e-12).log().T).clamp_min(0)
        return feature, kl

    def update_labels(self):
        self.counts = torch.bincount(self.assign, minlength=self.m)
        self.y = torch.zeros(self.m, self.f.shape[1], device=self.x.device, dtype=self.x.dtype)
        self.y.index_add_(0, self.assign, self.f)
        self.y /= self.counts[:, None]

    def assignment_step(self):
        best = torch.full((self.n,), float('inf'), device=self.x.device, dtype=self.x.dtype)
        proposal = self.assign.clone()
        old = torch.empty_like(best)
        for lo, hi in self.batches():
            feature, kl = self.costs(lo, hi)
            cost = feature + self.mu * kl
            value, index = cost.min(1)
            improve = value < best
            proposal[improve], best[improve] = index[improve] + lo, value[improve]
            mask = (self.assign >= lo) & (self.assign < hi)
            old[mask] = cost[mask, self.assign[mask] - lo]
        # Protect one current member per cell, selected by lowest current cost.
        # All other moves are independent decreases for frozen centers/labels.
        minima = torch.full((self.m,), float('inf'), device=self.x.device, dtype=self.x.dtype)
        minima.scatter_reduce_(0, self.assign, old, reduce='amin', include_self=True)
        ids = torch.arange(self.n, device=self.x.device)
        candidates = torch.where(old == minima[self.assign], ids, self.n)
        anchors = torch.full((self.m,), self.n, device=self.x.device, dtype=torch.long)
        anchors.scatter_reduce_(0, self.assign, candidates, reduce='amin', include_self=True)
        proposal[anchors] = self.assign[anchors]
        proposal[best >= old - 1e-10] = self.assign[best >= old - 1e-10]
        moved = int((proposal != self.assign).sum())
        self.assign = proposal
        return moved

    def center_step(self):
        for lo, hi in self.batches():
            membership = (self.assign[:, None] == torch.arange(lo, hi, device=self.x.device)[None, :]).to(self.x.dtype)
            weights = self.propagate(membership, transpose=True)
            origin_weight = (membership * self.q[:, None]).sum(0)
            self.z[lo:hi] = weighted_medians(self.x, weights, origin_weight, self.z[lo:hi], self.median_iters)

    def record(self, stage, **extra):
        feature_sum, kl_sum = 0., 0.
        for lo, hi in self.batches():
            feature, kl = self.costs(lo, hi)
            mask = (self.assign >= lo) & (self.assign < hi)
            col = self.assign[mask] - lo
            feature_sum += float(feature[mask, col].sum())
            kl_sum += float(kl[mask, col].sum())
        feature_mean, kl_mean = feature_sum / self.n, kl_sum / self.n
        counts = torch.bincount(self.assign, minlength=self.m)
        record = dict(stage=stage, J=feature_mean + self.mu * kl_mean,
                      feature=feature_mean, kl=kl_mean, min_cell=int(counts.min()),
                      max_cell=int(counts.max()), singletons=int((counts == 1).sum()), **extra)
        if not np.isfinite(record['J']):
            raise FloatingPointError('nonfinite condensation objective')
        if self.history and record['J'] > self.history[-1]['J'] + 1e-8 * max(1., abs(self.history[-1]['J'])):
            raise RuntimeError('condensation objective increased')
        self.history.append(record)
        self.log(f"  [{stage}] J={record['J']:.7g} feature={feature_mean:.7g} KL={kl_mean:.7g} cells={int(counts.min())}/{int(counts.max())}")
        return record['J']

    @torch.no_grad()
    def run(self, outer_iters=20):
        if outer_iters < 0:
            raise ValueError('outer_iters must be nonnegative')
        start = time.perf_counter()
        previous = self.record('init')
        for it in range(outer_iters):
            moved = self.assignment_step()
            self.record('assign', iteration=it + 1, moved=moved)
            self.update_labels()
            self.center_step()
            current = self.record('centers', iteration=it + 1)
            if previous - current <= self.tol * max(abs(previous), 1e-12):
                break
            previous = current
        return dict(H_cond=self.z.float(), Y_cond=self.y.float(), assign=self.assign,
                    cell_counts=self.counts, history=self.history,
                    seconds=time.perf_counter() - start,
                    config=dict(mu=self.mu, depth=2, operator='GCN symmetric normalization',
                                mass_correction=True, batch_size=self.batch_size,
                                median_iters=self.median_iters, tol=self.tol, outer_iters=outer_iters))
