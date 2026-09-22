"""Entropic solver for the 1-hop neighbourhood-OT objective of src/partition_ot.py (--ot_solver sinkhorn).

Same objective and the same four blocks, with every transport replaced by its entropic version
(Cuturi 2013; Benamou et al. 2015, iterative Bregman projections):

    OT_eps(nu, p) = min_Gamma <Gamma, C> + eps * sum Gamma (log Gamma - 1)   s.t. the marginals of Gamma,
    J_eps         = mean_t [ alpha ||H_t - Z_{a_t}|| + beta OT_eps(nu_t, P_cond[a_t]) + mu KL(F_t || Y_{a_t}) ]

The adjacency block is then the fixed-support entropic barycenter of the cell members' neighbourhood
distributions, solved by IBP, and every block still minimises its own sub-problem, so J_eps does not increase.
All of it is batched torch on the GPU in float64: Sinkhorn in the log domain, one [nnz, m] tensor for the
barycenter / evaluation passes, chunked [n, K, D, m] tensors for the assignment block.  The plan-entropy form
above (not KL against mass (x) p) is the one IBP minimises over p, so the adjacency block is exact for J_eps.

`eps` is fixed once at initialisation as `eps_rel * mean(cost)` and never re-scaled.  The exact LP path in
partition_ot.py stays the reference: the sharp transport cost <Gamma, C> is reported next to J_eps so the two
solvers can be compared on the same instance.
"""
import time
import numpy as np
import scipy.sparse as sp
import torch

NEG = -1e30          # stands in for log 0 on padded rows
MOVE_TOL = 1e-9
OBJ_TOL = 1e-6
EPS_PROB = 1e-12
MEDIAN_ITERS = 30
COINCIDENT = 1e-12


def weighted_geometric_medians(H, omega, Z0, iters=MEDIAN_ITERS):
    """Z[r] = argmin_z sum_u omega[u, r] ||H[u] - z||, batched over r, with the Vardi-Zhang coincident-point
    step; every column keeps its best iterate, so the fixed-coupling cost never increases."""
    Z = Z0.clone()
    D = torch.cdist(H, Z)
    f0 = (omega * D).sum(0)
    best_Z, best_f = Z.clone(), f0.clone()
    for _ in range(iters):
        coincident = (D < COINCIDENT) & (omega > 0)
        W = torch.where(coincident, torch.zeros_like(omega), omega / D.clamp(min=COINCIDENT))
        s = W.sum(0)
        T = torch.where((s > 0)[:, None], (W.T @ H) / s.clamp(min=COINCIDENT)[:, None], Z)
        eta = (omega * coincident).sum(0)
        Rn = s * (T - Z).norm(dim=1)
        lam = torch.where(Rn > 0, torch.clamp(eta / Rn.clamp(min=COINCIDENT), max=1.0), torch.ones_like(Rn))
        Z = (1 - lam)[:, None] * T + lam[:, None] * Z
        D = torch.cdist(H, Z)
        f = (omega * D).sum(0)
        better = f < best_f
        best_Z[better], best_f[better] = Z[better], f[better]
    return best_Z, f0, best_f


class NeighborhoodOTEntropic:
    def __init__(self, H, P, F, m, assign, alpha=1.0, beta=1.0, mu=1.0, eps_rel=0.05, sink_iters=200,
                 ibp_iters=200, candidates=16, chunk_elems=None, device=None, log=print):
        dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if chunk_elems is None:            # the assignment block holds ~16 live [n, K, D, m] float64 tensors
            free = torch.cuda.mem_get_info(dev)[0] if str(dev).startswith('cuda') else 2e9
            chunk_elems = min(8e7, max(5e5, free / 8 / 16))
        self.dev, self.log = dev, log
        P = sp.csr_matrix(P); P.sort_indices()
        assert np.allclose(np.asarray(P.sum(1)).ravel(), 1.0, atol=1e-10), 'rows of P must sum to 1'
        to_t = lambda A: (A.detach() if torch.is_tensor(A) else torch.as_tensor(np.asarray(A))).to(dtype=torch.float64, device=dev)
        self.H = to_t(H)
        F = to_t(F).clamp(min=EPS_PROB)
        self.F = F / F.sum(1, keepdim=True)
        self.entropy = (self.F * self.F.log()).sum(1)
        self.N, self.m = len(self.H), int(m)
        self.node_ptr = torch.as_tensor(P.indptr, dtype=torch.long, device=dev)
        self.indices = torch.as_tensor(P.indices, dtype=torch.long, device=dev)
        self.mass = torch.as_tensor(P.data, dtype=torch.float64, device=dev)
        self.logmass = self.mass.clamp(min=1e-300).log()
        self.deg = self.node_ptr[1:] - self.node_ptr[:-1]
        self.node_of_row = torch.repeat_interleave(torch.arange(self.N, device=dev), self.deg)
        self.nnz = len(self.indices)
        self.assign = torch.as_tensor(np.asarray(assign), dtype=torch.long, device=dev).clone()
        self.counts = torch.bincount(self.assign, minlength=self.m)
        assert (self.counts > 0).all(), 'initial assignment must leave no empty cell'
        self.alpha, self.beta, self.mu = float(alpha), float(beta), float(mu)
        self.sink_iters, self.ibp_iters = int(sink_iters), int(ibp_iters)
        self.candidates, self.chunk_elems = int(candidates), int(chunk_elems)
        self.Z = self._root_medians()
        self.Y = self._cell_means()
        self.Pc = self._mean_cell_transition(P)
        self.cost = None
        self.eps = float(eps_rel) * float(self._costs().mean())
        self.gamma = None
        self.history, self.diag = [], {'blocks': []}

    # ---- initial state
    def _root_medians(self):
        omega = torch.zeros(self.N, self.m, dtype=torch.float64, device=self.dev)
        omega[torch.arange(self.N, device=self.dev), self.assign] = 1.0
        Z0 = torch.zeros(self.m, self.H.shape[1], dtype=torch.float64, device=self.dev)
        Z0.index_add_(0, self.assign, self.H)
        Z0 /= self.counts.clamp(min=1).to(Z0.dtype)[:, None]
        return weighted_geometric_medians(self.H, omega, Z0)[0]

    def _cell_means(self):
        Y = torch.zeros(self.m, self.F.shape[1], dtype=torch.float64, device=self.dev)
        Y.index_add_(0, self.assign, self.F)
        return Y / self.counts.clamp(min=1).to(Y.dtype)[:, None]

    def _mean_cell_transition(self, P):
        a = self.assign.cpu().numpy()
        S = sp.csr_matrix((np.ones(self.N), (np.arange(self.N), a)), shape=(self.N, self.m))
        PS = np.asarray((P @ S).todense())
        Pc = np.zeros((self.m, self.m)); np.add.at(Pc, a, PS)
        Pc = Pc / np.bincount(a, minlength=self.m).clip(min=1)[:, None]
        return torch.as_tensor(Pc, dtype=torch.float64, device=self.dev)

    # ---- caches
    def _costs(self):
        if self.cost is None:
            self.cost = torch.cdist(self.H[self.indices], self.Z)      # [nnz, m]
        return self.cost

    def _seg_sum(self, X):
        out = torch.zeros(self.N, X.shape[1], dtype=X.dtype, device=self.dev)
        return out.index_add_(0, self.node_of_row, X)

    # ---- Sinkhorn / IBP over the flat neighbour layout (one destination row per node)
    def _scaling_loop(self, logp_node, iters, ibp=False):
        """ibp=False: Sinkhorn against the fixed rows logp_node [N, m].
        ibp=True: iterative Bregman projections, the cell-wise geometric mean of the current destination
        marginals is the target of every member (the fixed-support entropic barycenter).
        Returns (value_eps [N], sharp [N], Gamma [nnz, m], logp_node, residual)."""
        cost = self._costs()
        eps = self.eps
        g = torch.zeros(self.N, self.m, dtype=torch.float64, device=self.dev)
        for _ in range(iters):
            f = eps * (self.logmass - torch.logsumexp((g[self.node_of_row] - cost) / eps, dim=1))
            G = torch.exp((f[:, None] + g[self.node_of_row] - cost) / eps)
            logc = self._seg_sum(G).clamp(min=1e-300).log()             # current destination marginals [N, m]
            if ibp:
                acc = torch.zeros(self.m, self.m, dtype=torch.float64, device=self.dev)
                acc.index_add_(0, self.assign, logc)
                logp = acc / self.counts.clamp(min=1).to(acc.dtype)[:, None]
                logp_node = (logp - torch.logsumexp(logp, dim=1, keepdim=True))[self.assign]
            g = g + eps * (logp_node - logc)
        f = eps * (self.logmass - torch.logsumexp((g[self.node_of_row] - cost) / eps, dim=1))
        logG = (f[:, None] + g[self.node_of_row] - cost) / eps
        G = torch.exp(logG)
        sharp = self._seg_sum(G * cost).sum(1)
        ent = self._seg_sum(torch.where(G > 0, G * (logG - 1), torch.zeros_like(G))).sum(1)
        residual = float((self._seg_sum(G) - logp_node.exp()).abs().max())
        return sharp + eps * ent, sharp, G, logp_node, residual

    def _candidate_costs(self, cand):
        """Entropic OT cost of every node against each of its K candidate cells.  cand [N, K] -> [N, K]."""
        cost_all, eps, K = self._costs(), self.eps, cand.shape[1]
        logPc = self.Pc.clamp(min=1e-300).log()
        out = torch.empty(self.N, K, dtype=torch.float64, device=self.dev)
        order = torch.argsort(self.deg)
        degs = self.deg[order].cpu().numpy()                             # ascending: padding inside a chunk is cheap
        budget, i = self.chunk_elems, 0
        while i < self.N:
            lo, hi = 1, self.N - i                                       # largest n with n * max_deg(chunk) * K * m <= budget
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if mid * int(degs[i + mid - 1]) * K * self.m <= budget:
                    lo = mid
                else:
                    hi = mid - 1
            n = lo
            nodes = order[i:i + n]
            D = int(self.deg[nodes].max())
            ar = torch.arange(D, device=self.dev)
            mask = ar[None, :] < self.deg[nodes][:, None]
            idx = (self.node_ptr[nodes][:, None] + ar[None, :]).clamp(max=self.nnz - 1)
            cost = torch.where(mask[..., None], cost_all[idx], torch.zeros((), dtype=cost_all.dtype, device=self.dev))
            logmass = torch.where(mask, self.logmass[idx], torch.full_like(mask, NEG, dtype=torch.float64))
            logp = logPc[cand[nodes]]                                    # [n, K, m]
            g = torch.zeros(len(nodes), K, self.m, dtype=torch.float64, device=self.dev)
            for _ in range(self.sink_iters):
                f = eps * (logmass[:, None, :] - torch.logsumexp((g[:, :, None, :] - cost[:, None, :, :]) / eps, dim=-1))
                M = (f[..., None] + g[:, :, None, :] - cost[:, None, :, :]) / eps
                logc = torch.logsumexp(M, dim=-2)
                g = g + eps * (logp - logc)
            f = eps * (logmass[:, None, :] - torch.logsumexp((g[:, :, None, :] - cost[:, None, :, :]) / eps, dim=-1))
            M = (f[..., None] + g[:, :, None, :] - cost[:, None, :, :]) / eps
            G = torch.exp(M)
            sharp = (G * cost[:, None, :, :]).sum((-1, -2))
            ent = torch.where(G > 0, G * (M - 1), torch.zeros_like(G)).sum((-1, -2))
            out[nodes] = sharp + eps * ent
            i += n
        return out

    # ---- blocks
    def evaluate(self, stage, w1=None, extra=None):
        t0 = time.time()
        root = (self.H - self.Z[self.assign]).norm(dim=1)
        kl = self.entropy - (self.F * self.Y[self.assign].clamp(min=EPS_PROB).log()).sum(1)
        if self.beta == 0:
            w1 = (torch.zeros(self.N, device=self.dev), torch.zeros(self.N, device=self.dev))
        elif w1 is None:
            v, s, self.gamma, _, _ = self._scaling_loop(self.Pc.clamp(min=1e-300).log()[self.assign], self.sink_iters)
            w1 = (v, s)
        self.last_w1 = w1
        rec = {'stage': stage, 'root': float(self.alpha * root.mean()), 'w1': float(self.beta * w1[0].mean()),
               'kl': float(self.mu * kl.mean()), 'w1_sharp': float(self.beta * w1[1].mean()),
               'min_count': int(self.counts.min()), 'eval_time': round(time.time() - t0, 2)}
        rec['J'] = rec['root'] + rec['w1'] + rec['kl']
        if extra:
            rec.update(extra)
        self.history.append(rec)
        self.log(f"  [{stage:<11s}] J_eps {rec['J']:.6f} = root {rec['root']:.6f} + w1 {rec['w1']:.6f} "
                 f"+ kl {rec['kl']:.6f}   (sharp w1 {rec['w1_sharp']:.6f})"
                 + ''.join(f'  {k} {v}' for k, v in (extra or {}).items()))
        return rec

    def assignment_block(self):
        t0 = time.time()
        base = self.alpha * torch.cdist(self.H, self.Z) \
            + self.mu * (self.entropy[:, None] - self.F @ self.Y.clamp(min=EPS_PROB).log().T)
        if self.beta > 0:
            M = self._seg_sum(self.mass[:, None] * self.H[self.indices])          # neighbourhood means
            lb = base + self.beta * torch.cdist(M, self.Pc @ self.Z)              # Jensen lower bound on W1
            K = min(self.candidates, self.m)
            cand = lb.topk(K, dim=1, largest=False).indices
            cand = torch.cat([self.assign[:, None], cand], 1)                     # the incumbent is always a candidate
            cost = self._candidate_costs(cand)
            total = base.gather(1, cand) + self.beta * cost
        else:
            cand = torch.arange(self.m, device=self.dev).expand(self.N, self.m)
            total = base
        cand_np, total_np = cand.cpu().numpy(), total.cpu().numpy()
        assign, counts = self.assign.cpu().numpy(), self.counts.cpu().numpy()
        moved = 0
        for t in range(self.N):
            j0 = assign[t]
            if counts[j0] == 1:
                continue
            row, cur = cand_np[t], float(total_np[t][cand_np[t] == assign[t]][0])
            k = int(np.argmin(total_np[t]))
            if row[k] != j0 and total_np[t, k] < cur - MOVE_TOL:
                counts[j0] -= 1; counts[row[k]] += 1; assign[t] = row[k]; moved += 1
        self.assign = torch.as_tensor(assign, dtype=torch.long, device=self.dev)
        self.counts = torch.as_tensor(counts, dtype=torch.long, device=self.dev)
        self.gamma = None
        return {'moved': moved, 'candidates': int(cand.shape[1]), 'time': round(time.time() - t0, 2)}

    def label_block(self):
        self.Y = self._cell_means()

    def adjacency_block(self):
        """Fixed-support entropic barycenter per cell (IBP), all cells at once."""
        t0 = time.time()
        v, s, G, logp_node, residual = self._scaling_loop(self.Pc.clamp(min=1e-300).log()[self.assign],
                                                          self.ibp_iters, ibp=True)
        acc = torch.zeros(self.m, self.m, dtype=torch.float64, device=self.dev)
        acc.index_add_(0, self.assign, logp_node)
        p = (acc / self.counts.clamp(min=1).to(acc.dtype)[:, None]).exp()
        self.Pc = p / p.sum(1, keepdim=True)
        self.gamma = G
        return {'w1': (v, s), 'ibp_iters': self.ibp_iters, 'residual': f'{residual:.1e}', 'time': round(time.time() - t0, 2)}

    def feature_block(self):
        t0 = time.time()
        omega = torch.zeros(self.N, self.m, dtype=torch.float64, device=self.dev)
        omega[torch.arange(self.N, device=self.dev), self.assign] = self.alpha
        if self.beta > 0 and self.gamma is not None:
            omega.index_add_(0, self.indices, self.beta * self.gamma)
        Z, f0, f1 = weighted_geometric_medians(self.H, omega, self.Z)
        assert bool((f1 <= f0 + 1e-9 * f0.clamp(min=1)).all()), 'feature block increased a fixed-coupling cost'
        self.Z, self.cost = Z, None
        return {'fixed_cost_before': round(float(f0.sum()) / self.N, 6), 'fixed_cost_after': round(float(f1.sum()) / self.N, 6),
                'time': round(time.time() - t0, 2)}

    def run(self, outer_iters=5, final_polish=True):
        cfg = {'solver': 'sinkhorn', 'alpha': self.alpha, 'beta': self.beta, 'mu': self.mu, 'eps': self.eps,
               'sink_iters': self.sink_iters, 'ibp_iters': self.ibp_iters, 'candidates': self.candidates,
               'N': self.N, 'm': self.m, 'nnz_P': self.nnz, 'outer_iters': outer_iters, 'device': str(self.dev)}
        self.log(f'ot_1hop (sinkhorn): N {self.N} m {self.m} nnz(P) {self.nnz} alpha {self.alpha:g} beta {self.beta:g} '
                 f'mu {self.mu:g} eps {self.eps:.4g} device {self.dev}')
        prev = self.evaluate('init')['J']
        for it in range(outer_iters):
            self.log(f'outer {it + 1}/{outer_iters}')
            self.evaluate('assign', extra=self.assignment_block())
            self.label_block()
            self.evaluate('labels', w1=self.last_w1)
            if self.beta > 0:
                rec = self.adjacency_block()
                self.evaluate('adjacency', w1=rec.pop('w1'), extra=rec)
            self.evaluate('features', extra=self.feature_block())
            cur = self.history[-1]['J']
            if prev - cur < OBJ_TOL * max(abs(prev), 1e-12):
                self.log(f'  stop: improvement {prev - cur:.3e} below tolerance')
                break
            prev = cur
        if final_polish and self.beta > 0:
            rec = self.adjacency_block()
            self.evaluate('polish', w1=rec.pop('w1'), extra=rec)
        return {'H_cond': self.Z.cpu().numpy(), 'P_cond': self.Pc.cpu().numpy(), 'Y_cond': self.Y.cpu().numpy(),
                'assign': self.assign.cpu().numpy(), 'cell_counts': self.counts.cpu().numpy(),
                'history': self.history, 'diagnostics': self.diag, 'config': cfg}


def partition_ot_1hop_entropic(H, P, F, m, assign, alpha=1.0, beta=1.0, mu=1.0, eps_rel=0.05, sink_iters=200,
                               ibp_iters=200, candidates=16, device=None, outer_iters=5, log=print):
    return NeighborhoodOTEntropic(H, P, F, m, assign, alpha, beta, mu, eps_rel, sink_iters, ibp_iters,
                                  candidates, device=device, log=log).run(outer_iters)
