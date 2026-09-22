"""Structure-aware GRIP partition from the message-passing risk bound (opt-in: --edges structure_identity /
structure_median).  Reference implementation: float64 numpy, sequential exact local moves, no OT / LP / Sinkhorn.

Objects.  H [N, d] raw input features of the student (not propagated), P [N, N] the row-stochastic operator the
faithful student uses (P[t, u] = weight t receives from u), F [N, C] fixed teacher probabilities, a [N] a partition
into m non-empty cells with one-hot S, B = P S [N, m] the cell mass of every node's neighbourhood, and the
condensed graph (Hc [m, d], Q [m, m], Yc [m, C]).

Bound (docs/structure_bound.md).  For the bias-free student Z[l+1] = sigma_l(P Z[l] W_l) run with the same weights
on (Hc, Q), with E_l = ||Z[l] - S Zc[l]||_{2,1}, R_S = ||P S - S Q||_{2,1} and c_P = max_u sum_t P[t, u]:
    E_{l+1} <= a_l (c_P E_l + R_S ||Zc[l]||_2),      a_l = Lip(sigma_l) ||W_l||_2,
so E_K <= A E_0 + Bcoef R_S and, CE being sqrt(2)-Lipschitz in the logits,
    R_F(g) <= mean_t H(F_t) + [alpha D_H + beta D_S + D_KL] / N + sum_j n_j KL(Yc_j || g_c[j]) / N
with D_H = sum_t ||H_t - Hc[a_t]||, D_S = sum_t ||B_t - Q[a_t]||, D_KL = sum_t KL(F_t || Yc[a_t]).

Optimised objective  J = (alpha D_H + beta D_S + mu D_KL) / N.  Blocks:
    assignment : sequential moves t: a -> b with the exact Delta J of the frozen centres (all m destinations,
                 the rows U = {u : P[u, t] != 0} + {t} whose B changes are re-evaluated), accepted only if
                 Delta J < -tol, never emptying a cell
    Hc         : cell geometric medians (per-cell Weiszfeld / Vardi-Zhang, best iterate kept)
    Yc         : cell means of F
    Q          : identity (fixed, Q_j = e_j) or the cell geometric medians of the current B (learned_median)
Q modes are never mixed: in identity mode Q is not updated.  No per-term normalisation in this version.
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from src.partition_ot import build_transition, transition_to_edges, check_transition, EPS_PROB

MOVE_TOL = 1e-9
OBJ_TOL = 1e-6
MEDIAN_ITERS = 50
COINCIDENT = 1e-12


# ----------------------------------------------------------------------------- bound constants
def column_sum_max(P):
    """c_P = max_u sum_t P[t, u]: the constant of ||P M||_{2,1} <= c_P ||M||_{2,1}."""
    return float(np.asarray(sp.csr_matrix(P).sum(0)).max())


def bound_coefficients(cP, K, abar, R, m):
    """alpha, beta of the explicit-constant bound for a K-layer bias-free student with layer bounds
    a_l <= abar (scalar or list), inputs in the ball of radius R, Hc in that ball, Q row-stochastic:
    M_l = sqrt(m) R prod_{s<l} abar_s, A = prod_l (abar_l c_P), Bcoef = sum_l abar_l M_l prod_{s>l} (abar_s c_P)."""
    ab = [float(abar)] * K if np.isscalar(abar) else [float(x) for x in abar]
    assert len(ab) == K
    A = np.prod([x * cP for x in ab])
    Bc = 0.0
    for l in range(K):
        M_l = np.sqrt(m) * R * np.prod(ab[:l])
        Bc += ab[l] * M_l * np.prod([x * cP for x in ab[l + 1:]])
    return np.sqrt(2) * A, np.sqrt(2) * Bc


# ----------------------------------------------------------------------------- geometric medians per cell
def geometric_median(X, z0, iters=MEDIAN_ITERS):
    """Unweighted geometric median of the rows of X (Weiszfeld with the Vardi-Zhang step at coincident points);
    the best iterate, including the start, is returned so the cost never increases.  Iterates are convex
    combinations of the points and the start, so a simplex-valued input stays in the simplex."""
    z = z0.copy()
    d = np.linalg.norm(X - z, axis=1)
    best_z, best_f = z.copy(), d.sum()
    for _ in range(iters):
        coincident = d < COINCIDENT
        w = np.where(coincident, 0.0, 1.0 / np.maximum(d, COINCIDENT))
        s = w.sum()
        if s == 0:
            break
        T = (w @ X) / s
        eta = coincident.sum()
        r = s * np.linalg.norm(T - z)
        lam = min(1.0, eta / r) if r > 0 else 1.0
        z = (1 - lam) * T + lam * z
        d = np.linalg.norm(X - z, axis=1)
        f = d.sum()
        if f < best_f:
            best_z, best_f = z.copy(), f
    return best_z, best_f


def cell_geometric_medians(X, a, m, Z0=None):
    """Z[j] = geometric median of {X[t] : a[t] = j}, started from Z0[j] (cell means if None).
    Returns (Z, cost_before, cost_after) with the per-cell sums of distances."""
    Z = np.zeros((m, X.shape[1])) if Z0 is None else Z0.copy()
    f0, f1 = np.zeros(m), np.zeros(m)
    for j in range(m):
        idx = np.flatnonzero(a == j)
        z0 = X[idx].mean(0) if Z0 is None else Z0[j]
        f0[j] = np.linalg.norm(X[idx] - z0, axis=1).sum()
        Z[j], f1[j] = geometric_median(X[idx], z0)
    return Z, f0, f1


# ----------------------------------------------------------------------------- the algorithm
class StructureBound:
    def __init__(self, H, P, F, m, assign, alpha=1.0, beta=1.0, mu=1.0, q_mode='identity', seed=0, log=print):
        assert q_mode in ('identity', 'learned_median')
        self.H = np.asarray(H, dtype=np.float64)
        self.P = check_transition(P)
        self.Pt = sp.csr_matrix(self.P.T)                      # column access: rows u that receive from t
        self.P_diag = self.P.diagonal()
        F = np.asarray(F, dtype=np.float64)
        assert (F >= 0).all() and np.allclose(F.sum(1), 1, atol=1e-6), 'F must be row-stochastic'
        F = np.maximum(F, EPS_PROB); self.F = F / F.sum(1, keepdims=True)      # smoothed once, then fixed
        self.entropy = (self.F * np.log(self.F)).sum(1)
        self.N, self.m = len(self.H), int(m)
        self.a = np.asarray(assign).astype(np.int64).copy()
        assert self.a.min() >= 0 and self.a.max() < self.m
        self.counts = np.bincount(self.a, minlength=self.m)
        assert (self.counts > 0).all(), 'initial assignment must leave no empty cell'
        self.alpha, self.beta, self.mu = float(alpha), float(beta), float(mu)
        self.q_mode, self.log = q_mode, log
        self.order = np.random.default_rng(seed).permutation(self.N)
        self.cP = column_sum_max(self.P)
        self.B = self._PS()
        self.Hc = cell_geometric_medians(self.H, self.a, self.m)[0]
        self.Yc = self._cell_means()
        self.Q = np.eye(self.m) if q_mode == 'identity' else cell_geometric_medians(self.B, self.a, self.m)[0]
        self._refresh_residuals()
        self.history = []

    # ---- state helpers
    def _S(self):
        return sp.csr_matrix((np.ones(self.N), (np.arange(self.N), self.a)), shape=(self.N, self.m))

    def _PS(self):
        return np.asarray((self.P @ self._S()).todense())

    def _cell_means(self):
        Y = np.zeros((self.m, self.F.shape[1])); np.add.at(Y, self.a, self.F)
        return Y / self.counts[:, None]

    def _refresh_residuals(self):
        self.R = self.B - self.Q[self.a]
        self.nsq = (self.R ** 2).sum(1)
        self.D_H = np.linalg.norm(self.H - self.Hc[self.a], axis=1).sum()
        self.D_S = np.sqrt(self.nsq).sum()
        self.D_KL = self._kl_rows()[np.arange(self.N), self.a].sum()

    def _kl_rows(self):
        return self.entropy[:, None] - self.F @ np.log(np.maximum(self.Yc, EPS_PROB)).T

    def objective(self, D_H=None, D_S=None, D_KL=None):
        D_H = self.D_H if D_H is None else D_H
        D_S = self.D_S if D_S is None else D_S
        D_KL = self.D_KL if D_KL is None else D_KL
        return (self.alpha * D_H + self.beta * D_S + self.mu * D_KL) / self.N

    def evaluate_objective(self):
        """Full recomputation from a, Hc, Q, Yc (B rebuilt by a fresh sparse product)."""
        B = self._PS()
        D_H = np.linalg.norm(self.H - self.Hc[self.a], axis=1).sum()
        D_S = np.linalg.norm(B - self.Q[self.a], axis=1).sum()
        D_KL = self._kl_rows()[np.arange(self.N), self.a].sum()
        return self.objective(D_H, D_S, D_KL), D_H, D_S, D_KL, B

    def verify(self, tol=1e-8):
        J, D_H, D_S, D_KL, B = self.evaluate_objective()
        assert np.abs(B - self.B).max() < tol, 'cached B differs from P S'
        inc = self.objective()
        assert abs(J - inc) < tol * max(1.0, abs(J)), f'incremental objective {inc} != recomputed {J}'
        return J

    # ---- exact move deltas
    def exact_move_delta(self, t, DH_rows, KL_rows):
        """Delta J (and its three parts) of moving t to every destination cell, centres frozen.
        Returns arrays of length m; entry a[t] is 0."""
        a, m = self.a[t], self.m
        dDH = DH_rows[t] - DH_rows[t, a]
        dKL = KL_rows[t] - KL_rows[t, a]
        lo, hi = self.Pt.indptr[t], self.Pt.indptr[t + 1]
        us, ps = self.Pt.indices[lo:hi], self.Pt.data[lo:hi]
        keep = us != t
        us, ps = us[keep], ps[keep]
        dDS = np.zeros(m)
        if len(us):                                   # rows u != t: only coordinates a and b of the residual change
            Ru, old = self.R[us], np.sqrt(self.nsq[us])
            ra = Ru[:, a]
            base = self.nsq[us] - ra ** 2 + (ra - ps) ** 2                     # coordinate a after the move
            new = base[:, None] - Ru ** 2 + (Ru + ps[:, None]) ** 2            # coordinate b after the move
            dDS = (np.sqrt(np.maximum(new, 0.0)) - old[:, None]).sum(0)
        Mt = self.B[t][None, :] - self.Q                                       # row t against every destination row
        p_tt = self.P_diag[t]
        if p_tt != 0:
            Mt[np.arange(m), np.arange(m)] += p_tt
            Mt[:, a] -= p_tt
        dDS = dDS + np.linalg.norm(Mt, axis=1) - np.sqrt(self.nsq[t])
        dDS[a] = dDH[a] = dKL[a] = 0.0
        dJ = (self.alpha * dDH + self.beta * dDS + self.mu * dKL) / self.N
        dJ[a] = 0.0
        return dJ, dDH, dDS, dKL

    def apply_move(self, t, b, dDH, dDS, dKL):
        a = self.a[t]
        lo, hi = self.Pt.indptr[t], self.Pt.indptr[t + 1]
        for u, p in zip(self.Pt.indices[lo:hi], self.Pt.data[lo:hi]):
            self.B[u, a] -= p; self.B[u, b] += p
            if u != t:
                self.R[u, a] -= p; self.R[u, b] += p
                self.nsq[u] = self.R[u] @ self.R[u]
        self.a[t] = b
        self.counts[a] -= 1; self.counts[b] += 1
        self.R[t] = self.B[t] - self.Q[b]
        self.nsq[t] = self.R[t] @ self.R[t]
        self.D_H += dDH; self.D_S += dDS; self.D_KL += dKL

    def assignment_sweep(self):
        t0 = time.time()
        DH_rows, KL_rows = cdist(self.H, self.Hc), self._kl_rows()
        moved = 0
        for t in self.order:
            a = self.a[t]
            if self.counts[a] == 1:
                continue
            dJ, dDH, dDS, dKL = self.exact_move_delta(t, DH_rows, KL_rows)
            b = int(np.argmin(dJ))
            if b != a and dJ[b] < -MOVE_TOL:
                self.apply_move(t, b, dDH[b], dDS[b], dKL[b])
                moved += 1
        return {'moved': moved, 'time': round(time.time() - t0, 2)}

    def update_centers(self):
        t0 = time.time()
        self.Hc, f0, f1 = cell_geometric_medians(self.H, self.a, self.m, self.Hc)
        assert (f1 <= f0 + 1e-9 * np.maximum(f0, 1)).all(), 'feature medians increased a cell cost'
        self.Yc = self._cell_means()
        rec = {'median_H': (round(f0.sum() / self.N, 6), round(f1.sum() / self.N, 6))}
        if self.q_mode == 'learned_median':
            self.Q, g0, g1 = cell_geometric_medians(self.B, self.a, self.m, self.Q)
            assert (g1 <= g0 + 1e-9 * np.maximum(g0, 1)).all(), 'Q medians increased a cell cost'
            assert (self.Q >= -1e-12).all() and np.allclose(self.Q.sum(1), 1, atol=1e-9), 'Q left the simplex'
            rec['median_Q'] = (round(g0.sum() / self.N, 6), round(g1.sum() / self.N, 6))
        self._refresh_residuals()
        rec['time'] = round(time.time() - t0, 2)
        return rec

    # ---- logging / driver
    def record(self, stage, extra=None):
        c = self.counts
        rec = {'stage': stage, 'J': self.objective(), 'D_H': self.D_H / self.N, 'D_S': self.D_S / self.N,
               'D_KL': self.D_KL / self.N, 'cells': (int(c.min()), int(np.median(c)), int(c.max())),
               'singletons': int((c == 1).sum())}
        if extra:
            rec.update(extra)
        self.history.append(rec)
        self.log(f"  [{stage:<9s}] J {rec['J']:.6f} = a*{rec['D_H']:.6f} + b*{rec['D_S']:.6f} + mu*{rec['D_KL']:.6f}"
                 f"  cells(min/med/max) {rec['cells']} singletons {rec['singletons']}"
                 + ''.join(f'  {k} {v}' for k, v in (extra or {}).items()))
        return rec

    def run(self, outer_iters=10):
        self.log(f'structure_bound[{self.q_mode}]: N {self.N} m {self.m} nnz(P) {self.P.nnz} c_P {self.cP:.3f} '
                 f'alpha {self.alpha:g} beta {self.beta:g} mu {self.mu:g}')
        prev = self.record('init')['J']
        for it in range(outer_iters):
            self.log(f'outer {it + 1}/{outer_iters}')
            rec = self.assignment_sweep()
            rec['full_J'] = round(self.verify(), 6)
            self.record('assign', rec)
            self.record('centers', self.update_centers())
            cur = self.verify()
            if prev - cur < OBJ_TOL * max(abs(prev), 1e-12):
                self.log(f'  stop: improvement {prev - cur:.3e} below tolerance')
                break
            prev = cur
        return {'H_cond': self.Hc, 'Q': self.Q, 'Y_cond': self.Yc, 'assign': self.a, 'cell_counts': self.counts,
                'history': self.history,
                'config': {'q_mode': self.q_mode, 'alpha': self.alpha, 'beta': self.beta, 'mu': self.mu, 'c_P': self.cP,
                           'N': self.N, 'm': self.m, 'nnz_P': int(self.P.nnz), 'outer_iters': outer_iters,
                           'move_tol': MOVE_TOL, 'objective_tol': OBJ_TOL}}


def structure_bound(H, P, F, m, assign, alpha=1.0, beta=1.0, mu=1.0, q_mode='identity', outer_iters=10, seed=0, log=print):
    return StructureBound(H, P, F, m, assign, alpha, beta, mu, q_mode, seed, log).run(outer_iters)
