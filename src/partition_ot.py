"""1-hop neighbourhood-OT extension of GRIP (opt-in: --edges ot_1hop). Exact reference implementation.

Fixed objective, for a partition a, representatives H_cond [m, d], a row-stochastic condensed
propagation matrix P_cond [m, m] and soft labels Y_cond [m, C]:

    J = (1/N) sum_t [ alpha * ||H[t] - H_cond[a_t]||_2
                    + beta  * W1(nu_t, nu'_{a_t})
                    + mu    * KL(F[t] || Y_cond[a_t]) ]

    nu_t   = sum_u P[t, u] delta_{H[u]}          (original 1-hop neighbourhood of t)
    nu'_j  = sum_r P_cond[j, r] delta_{H_cond[r]} (condensed 1-hop neighbourhood of j)

Block coordinate descent (every block is exact for its own sub-problem, so J never increases):
    assignment  : sequential, strictly cost-decreasing moves under a non-empty-cell constraint
    labels      : cell means of F (minimiser of the forward KL)
    adjacency   : per cell, one LP over the common row p_j and the couplings Gamma_t of its members
    features    : weighted geometric medians with weights collected from all couplings (per-column rollback)

Everything is float64 numpy / scipy on the CPU; transport problems are unregularised LPs solved with HiGHS.
This is the 1-hop normalised neighbourhood-OT objective, not tree mover's distance.
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

EPS_PROB = 1e-12      # F is smoothed once with this value and renormalised; Y_cond is clipped at it only inside log
MOVE_TOL = 1e-9       # a node moves only if the new cell is cheaper by more than this (absolute)
LP_FEAS_TOL = 1e-7    # tolerance on the marginal residuals of an LP solution
MEDIAN_ITERS = 30     # inner Weiszfeld / Vardi-Zhang iterations of the feature block
OBJ_TOL = 1e-6        # relative improvement of J over an outer iteration below which the loop stops
COINCIDENT = 1e-12    # a data point closer than this to the current median counts as coincident


# ----------------------------------------------------------------------------- propagation matrices
def build_transition(edge_index, num_nodes):
    """Row-stochastic 1-hop transition P [N, N] (csr): P[t, u] = weight node t receives from neighbour u.
    Rule: edges are binarised, the diagonal is removed (the root term is separate), rows are normalised;
    a node without neighbours gets P[t, t] = 1.  Returns (P, metadata)."""
    src, dst = np.asarray(edge_index[0]), np.asarray(edge_index[1])
    A = sp.csr_matrix((np.ones(len(src)), (dst, src)), shape=(num_nodes, num_nodes))
    A.data[:] = 1.0
    A.setdiag(0.0)
    A.eliminate_zeros()
    deg = np.asarray(A.sum(1)).ravel()
    isolated = np.flatnonzero(deg == 0)
    inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0.0)
    P = sp.diags(inv) @ A
    if len(isolated):
        P = P + sp.csr_matrix((np.ones(len(isolated)), (isolated, isolated)), shape=(num_nodes, num_nodes))
    P = sp.csr_matrix(P)
    P.sort_indices()
    meta = {'rule': 'binarised edges, diagonal removed, row-normalised', 'isolated_policy': 'P[t,t] = 1',
            'num_isolated': int(len(isolated)), 'nnz': int(P.nnz)}
    return P, meta


def check_transition(P):
    P = sp.csr_matrix(P)
    rs = np.asarray(P.sum(1)).ravel()
    assert P.shape[0] == P.shape[1], 'P must be square'
    assert (P.data >= 0).all(), 'P must be non-negative'
    assert np.allclose(rs, 1.0, atol=1e-10), 'rows of P must sum to 1'
    return P


def transition_to_edges(P):
    """P[j, r] (target j receives from source r) -> (edge_index [source; target], edge_attr)."""
    P = sp.coo_matrix(P)
    return np.vstack([P.col, P.row]), P.data.astype(np.float64)


# ----------------------------------------------------------------------------- exact transport LPs
def _transport_lp(cost, a, b):
    """Exact W1 between weighted point sets: min <Gamma, cost>, Gamma 1 = a, Gamma^T 1 = b, Gamma >= 0."""
    n, m = cost.shape
    A_src = sp.kron(sp.eye(n), np.ones((1, m)), format='csr')
    A_dst = sp.kron(np.ones((1, n)), sp.eye(m), format='csr')
    A = sp.vstack([A_src, A_dst[:-1]], format='csr')            # one destination row is implied by the others
    res = linprog(cost.ravel(), A_eq=A, b_eq=np.concatenate([a, b[:-1]]), bounds=(0, None), method='highs')
    if res.status != 0:
        raise RuntimeError(f'transport LP failed: {res.message}')
    return res.fun, res.x.reshape(n, m)


def pair_neighbor_ot(cost, mass, p):
    """W1(nu_t, nu'_j) for fixed p = P_cond[j].  cost [deg, m], mass [deg], p [m].
    Restricts the LP to the supports of mass and p (exact).  Returns (value, Gamma [deg, m])."""
    rows, cols = np.flatnonzero(mass > 0), np.flatnonzero(p > 0)
    Gamma = np.zeros_like(cost)
    if len(rows) == 1:                                           # a single source: the coupling is forced
        Gamma[rows[0], cols] = p[cols]
        return float(cost[rows[0], cols] @ p[cols]), Gamma
    if len(cols) == 1:
        Gamma[rows, cols[0]] = mass[rows]
        return float(cost[rows, cols[0]] @ mass[rows]), Gamma
    val, G = _transport_lp(cost[np.ix_(rows, cols)], mass[rows], p[cols])
    Gamma[np.ix_(rows, cols)] = G
    return val, Gamma


def solve_cell_neighbor_lp(costs, masses, m):
    """Joint LP of one cell: common destination row p [m] and one coupling per member.
    costs[i] [deg_i, m], masses[i] [deg_i].  Variables: all Gamma_i (row-major) then p.
    Returns dict(value, p, gammas, residual, nvar, ncon, time)."""
    degs = [len(a) for a in masses]
    k, nG = len(degs), sum(d * m for d in degs)
    A_src = sp.block_diag([sp.kron(sp.eye(d), np.ones((1, m))) for d in degs], format='csr')       # [sum deg, nG]
    A_dst = sp.block_diag([sp.kron(np.ones((1, d)), sp.eye(m)) for d in degs], format='csr')       # [k m, nG]
    top = sp.hstack([A_src, sp.csr_matrix((A_src.shape[0], m))], format='csr')
    bot = sp.hstack([A_dst, -sp.vstack([sp.eye(m)] * k, format='csr')], format='csr')
    A = sp.vstack([top, bot], format='csr')
    b = np.concatenate([np.concatenate(masses), np.zeros(k * m)])
    c = np.concatenate([cst.ravel() for cst in costs] + [np.zeros(m)])
    t0 = time.time()
    res = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method='highs')
    if res.status != 0:
        raise RuntimeError(f'cell LP failed: {res.message}')
    x = res.x
    p = np.maximum(x[nG:], 0.0)
    gammas, off = [], 0
    for d in degs:
        gammas.append(x[off:off + d * m].reshape(d, m)); off += d * m
    residual = float(np.abs(A @ x - b).max())
    return {'value': float(res.fun), 'p': p, 'gammas': gammas, 'residual': residual,
            'nvar': int(nG + m), 'ncon': int(A.shape[0]), 'time': time.time() - t0}


# ----------------------------------------------------------------------------- feature block
def weighted_geometric_medians(H, omega, Z0, iters=MEDIAN_ITERS):
    """Column-wise weighted geometric medians: Z[r] = argmin_z sum_u omega[u, r] ||H[u] - z||.
    Batched Weiszfeld with the Vardi-Zhang correction for coincident points; every column keeps the best
    iterate seen (including its start), so the fixed-coupling cost never increases.  Returns (Z, f_before, f_after)."""
    Z = Z0.copy()
    D = cdist(H, Z)
    f = (omega * D).sum(0)
    f0 = f.copy()
    best_Z, best_f = Z.copy(), f.copy()
    for _ in range(iters):
        coincident = (D < COINCIDENT) & (omega > 0)
        W = np.where(coincident, 0.0, omega / np.maximum(D, COINCIDENT))
        s = W.sum(0)
        active = s > 0
        T = np.where(active[:, None], (W.T @ H) / np.maximum(s, COINCIDENT)[:, None], Z)
        eta = (omega * coincident).sum(0)
        Rn = s * np.linalg.norm(T - Z, axis=1)
        lam = np.where(Rn > 0, np.minimum(1.0, eta / np.maximum(Rn, COINCIDENT)), 1.0)
        Z = (1 - lam)[:, None] * T + lam[:, None] * Z
        D = cdist(H, Z)
        f = (omega * D).sum(0)
        better = f < best_f
        best_Z[better], best_f[better] = Z[better], f[better]
    return best_Z, f0, best_f


# ----------------------------------------------------------------------------- the algorithm
class NeighborhoodOT:
    def __init__(self, H, P, F, m, assign, alpha=1.0, beta=1.0, mu=1.0, max_lp_variables=2_000_000, log=print):
        self.H = np.asarray(H, dtype=np.float64)
        self.P = check_transition(P)
        F = np.asarray(F, dtype=np.float64)
        F = np.maximum(F, EPS_PROB); self.F = F / F.sum(1, keepdims=True)      # smoothed once, then fixed
        self.logF = np.log(self.F)
        self.entropy = (self.F * self.logF).sum(1)
        self.N, self.m = len(self.H), int(m)
        self.assign = np.asarray(assign).astype(np.int64).copy()
        assert self.assign.min() >= 0 and self.assign.max() < self.m
        self.counts = np.bincount(self.assign, minlength=self.m)
        assert (self.counts > 0).all(), 'initial assignment must leave no empty cell'
        self.alpha, self.beta, self.mu = float(alpha), float(beta), float(mu)
        self.max_lp_variables, self.log = int(max_lp_variables), log
        self.indptr, self.indices, self.mass = self.P.indptr, self.P.indices, self.P.data
        self.M = self.P @ self.H                                               # neighbourhood means (W1 lower bound)
        self.Z = self._geometric_medians_root()
        self.Y = self._cell_means()
        self.Pc = self._mean_cell_transition()
        self.gammas = None                                                     # couplings, valid for the current (assign, Z, Pc)
        self.cost_all = None                                                   # cdist(H[indices], Z), valid for the current Z
        self.history = []
        self.diag = {'lp': [], 'time': {}}

    # ---- helpers
    def _geometric_medians_root(self):
        omega = np.zeros((self.N, self.m)); omega[np.arange(self.N), self.assign] = 1.0
        Z0 = np.zeros((self.m, self.H.shape[1]))
        np.add.at(Z0, self.assign, self.H); Z0 /= self.counts[:, None]
        return weighted_geometric_medians(self.H, omega, Z0)[0]

    def _cell_means(self):
        Y = np.zeros((self.m, self.F.shape[1])); np.add.at(Y, self.assign, self.F)
        return Y / self.counts[:, None]

    def _mean_cell_transition(self):
        S = sp.csr_matrix((np.ones(self.N), (np.arange(self.N), self.assign)), shape=(self.N, self.m))
        PS = (self.P @ S).toarray()
        Pc = np.zeros((self.m, self.m)); np.add.at(Pc, self.assign, PS)
        return Pc / self.counts[:, None]

    def _costs(self):
        if self.cost_all is None:
            self.cost_all = cdist(self.H[self.indices], self.Z)
        return self.cost_all

    def cost_t(self, t):
        return self._costs()[self.indptr[t]:self.indptr[t + 1]]

    def mass_t(self, t):
        return self.mass[self.indptr[t]:self.indptr[t + 1]]

    def kl_matrix(self):
        return self.entropy[:, None] - self.F @ np.log(np.maximum(self.Y, EPS_PROB)).T

    def root_matrix(self):
        return cdist(self.H, self.Z)

    def w1(self, t, j):
        return pair_neighbor_ot(self.cost_t(t), self.mass_t(t), self.Pc[j])

    # ---- objective
    def evaluate(self, stage, w1_from='fresh', extra=None):
        """Exact J with its three parts.  w1_from: 'fresh' (one pair OT per node), 'gammas' (stored couplings, valid
        right after the LP block) or 'previous' (unchanged since the last evaluation, e.g. after the label block)."""
        t0 = time.time()
        root = np.linalg.norm(self.H - self.Z[self.assign], axis=1)
        kl = self.entropy - (self.F * np.log(np.maximum(self.Y[self.assign], EPS_PROB))).sum(1)
        if self.beta == 0:
            w1 = np.zeros(self.N)
        elif w1_from == 'previous':
            w1 = self.last_w1
        elif w1_from == 'gammas' and self.gammas is not None:
            w1 = np.array([(self.gammas[t] * self.cost_t(t)).sum() for t in range(self.N)])
        else:
            w1 = np.array([self.w1(t, self.assign[t])[0] for t in range(self.N)])
        self.last_w1 = w1
        rec = {'stage': stage, 'root': self.alpha * root.mean(), 'w1': self.beta * w1.mean(), 'kl': self.mu * kl.mean(),
               'min_count': int(self.counts.min()), 'eval_time': time.time() - t0}
        rec['J'] = rec['root'] + rec['w1'] + rec['kl']
        if extra:
            rec.update(extra)
        self.history.append(rec)
        self.log(f"  [{stage:<11s}] J {rec['J']:.6f} = root {rec['root']:.6f} + w1 {rec['w1']:.6f} + kl {rec['kl']:.6f}"
                 + ''.join(f'  {k} {v}' for k, v in (extra or {}).items()))
        return rec

    # ---- blocks
    def assignment_block(self):
        """Sequential strictly-decreasing moves; a node never leaves a singleton cell; ties keep the current cell.
        Exact W1 is only computed for cells whose lower bound (distance of neighbourhood means) beats the incumbent."""
        t0 = time.time()
        R, KL = self.root_matrix(), self.kl_matrix()
        base = self.alpha * R + self.mu * KL
        LB = cdist(self.M, self.Pc @ self.Z) if self.beta > 0 else np.zeros_like(base)
        moved, n_ot = 0, 0
        for t in range(self.N):
            j0 = self.assign[t]
            if self.counts[j0] == 1:
                continue
            if self.beta == 0:
                j = int(np.argmin(base[t]))
                if base[t, j] < base[t, j0] - MOVE_TOL:
                    self.assign[t] = j; self.counts[j0] -= 1; self.counts[j] += 1; moved += 1
                continue
            cost_t, mass_t = self.cost_t(t), self.mass_t(t)
            best_j, best_c = j0, base[t, j0] + self.beta * pair_neighbor_ot(cost_t, mass_t, self.Pc[j0])[0]
            n_ot += 1
            lb = base[t] + self.beta * LB[t]
            for j in np.argsort(lb):
                if j == j0:
                    continue
                if lb[j] >= best_c - MOVE_TOL:
                    break
                c = base[t, j] + self.beta * pair_neighbor_ot(cost_t, mass_t, self.Pc[j])[0]
                n_ot += 1
                if c < best_c - MOVE_TOL:
                    best_c, best_j = c, j
            if best_j != j0:
                self.assign[t] = best_j; self.counts[j0] -= 1; self.counts[best_j] += 1; moved += 1
        self.gammas = None
        return {'moved': moved, 'pair_ot': n_ot, 'time': round(time.time() - t0, 2)}

    def label_block(self):
        self.Y = self._cell_means()

    def adjacency_block(self):
        """Per cell: joint LP over the common row p_j and all member couplings."""
        t0 = time.time()
        gammas, stats = [None] * self.N, []
        cells = [np.flatnonzero(self.assign == j) for j in range(self.m)]
        nvar = [self.m * sum(self.indptr[t + 1] - self.indptr[t] for t in c) + self.m for c in cells]
        self.log(f'  cell LPs: {self.m} problems, variables max {max(nvar)} total {sum(nvar)}, '
                 f'coupling storage ~{sum(nvar) * 8 / 1e6:.1f} MB')
        if max(nvar) > self.max_lp_variables:
            raise RuntimeError(f'largest cell LP has {max(nvar)} variables > max_lp_variables={self.max_lp_variables}')
        for j, c in enumerate(cells):
            out = solve_cell_neighbor_lp([self.cost_t(t) for t in c], [self.mass_t(t) for t in c], self.m)
            if out['residual'] > LP_FEAS_TOL:
                self.log(f'  warning: cell {j} LP residual {out["residual"]:.2e}')
            self.Pc[j] = out['p'] / out['p'].sum()
            for t, G in zip(c, out['gammas']):
                gammas[t] = G
            stats.append((out['nvar'], out['ncon'], out['residual'], out['time']))
        self.gammas = gammas
        st = np.array(stats)
        rec = {'cells': self.m, 'nvar_max': int(st[:, 0].max()), 'residual_max': float(st[:, 2].max()),
               'lp_time': round(float(st[:, 3].sum()), 2), 'time': round(time.time() - t0, 2)}
        self.diag['lp'].append(rec)
        return rec

    def feature_block(self):
        """Weighted geometric medians with weights from the root term and from every coupling that uses r."""
        t0 = time.time()
        omega = np.zeros((self.N, self.m))
        omega[np.arange(self.N), self.assign] = self.alpha
        if self.beta > 0 and self.gammas is not None:
            for t in range(self.N):
                np.add.at(omega, self.indices[self.indptr[t]:self.indptr[t + 1]], self.beta * self.gammas[t])
        Z, f0, f1 = weighted_geometric_medians(self.H, omega, self.Z)
        assert (f1 <= f0 + 1e-9 * np.maximum(f0, 1)).all(), 'feature block increased a fixed-coupling cost'
        self.Z, self.cost_all = Z, None
        return {'fixed_cost_before': round(float(f0.sum()) / self.N, 6), 'fixed_cost_after': round(float(f1.sum()) / self.N, 6),
                'time': round(time.time() - t0, 2)}

    # ---- driver
    def run(self, outer_iters=5, final_polish=True):
        cfg = {'alpha': self.alpha, 'beta': self.beta, 'mu': self.mu, 'N': self.N, 'm': self.m, 'nnz_P': int(self.P.nnz),
               'outer_iters': outer_iters, 'max_lp_variables': self.max_lp_variables,
               'tolerances': {'move': MOVE_TOL, 'lp_feasibility': LP_FEAS_TOL, 'objective': OBJ_TOL, 'prob_eps': EPS_PROB}}
        self.log(f'ot_1hop: N {self.N} m {self.m} nnz(P) {self.P.nnz} alpha {self.alpha:g} beta {self.beta:g} mu {self.mu:g}')
        prev = self.evaluate('init')['J']
        for it in range(outer_iters):
            self.log(f'outer {it + 1}/{outer_iters}')
            self.evaluate('assign', extra=self.assignment_block())
            self.label_block()
            self.evaluate('labels', w1_from='previous')
            if self.beta > 0:
                self.evaluate('adjacency', w1_from='gammas', extra=self.adjacency_block())
            self.evaluate('features', w1_from='fresh', extra=self.feature_block())
            cur = self.history[-1]['J']
            if prev - cur < OBJ_TOL * max(abs(prev), 1e-12):
                self.log(f'  stop: improvement {prev - cur:.3e} below tolerance')
                break
            prev = cur
        if final_polish and self.beta > 0:
            self.evaluate('polish', w1_from='gammas', extra=self.adjacency_block())
        return {'H_cond': self.Z, 'P_cond': self.Pc, 'Y_cond': self.Y, 'assign': self.assign, 'cell_counts': self.counts,
                'history': self.history, 'diagnostics': self.diag, 'config': cfg}


def partition_ot_1hop(H, P, F, m, assign, alpha=1.0, beta=1.0, mu=1.0, outer_iters=5, max_lp_variables=2_000_000, log=print):
    return NeighborhoodOT(H, P, F, m, assign, alpha, beta, mu, max_lp_variables, log).run(outer_iters)
