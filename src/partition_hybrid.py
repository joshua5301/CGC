"""Propagated-feature GRIP plus ||PS-S||_{1,1}/(2N), Q=I.

Exact sequential assignment moves, including incoming and outgoing edges.
No OT; no dense N-by-N or N-by-m structural matrix. See docs/hybrid_structure.md.
"""
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist

from src.partition_ot import check_transition
from src.partition_struct import cell_geometric_medians


def cut_cost(P, assignment):
    """Sum of directed transition mass crossing cells (half the L1 residual)."""
    coo = sp.coo_matrix(P)
    return float(coo.data[assignment[coo.row] != assignment[coo.col]].sum())


def move_cut_deltas(P, PT, assignment, node, m):
    """Exact change of cut_cost for every destination; self edges never cross."""
    masses = np.zeros(m)
    for matrix in (P, PT):
        lo, hi = matrix.indptr[node:node+2]
        neighbors, weights = matrix.indices[lo:hi], matrix.data[lo:hi]
        keep = neighbors != node
        masses += np.bincount(assignment[neighbors[keep]], weights=weights[keep], minlength=m)
    return masses[assignment[node]] - masses


class HybridIdentity:
    def __init__(self, H, F, P, assignment, centers, labels, feature_scale,
                 kl_scale, structure_weight=.1, label_weight=.5, seed=0):
        self.H = np.asarray(H, dtype=np.float64)
        self.F = np.asarray(F, dtype=np.float64)
        self.P = check_transition(P)
        self.PT = self.P.T.tocsr()
        self.a = np.asarray(assignment, dtype=np.int64).copy()
        self.C = np.asarray(centers, dtype=np.float64).copy()
        self.Y = np.asarray(labels, dtype=np.float64).copy()
        self.n, self.m = len(self.H), len(self.C)
        self.scale, self.kl_scale = float(feature_scale), float(kl_scale)
        self.lam, self.mu = float(structure_weight), float(label_weight)
        if (self.n == 0 or self.m == 0 or self.P.shape != (self.n, self.n)
                or self.a.shape != (self.n,) or self.a.min() < 0 or self.a.max() >= self.m
                or self.H.ndim != 2 or self.C.shape[1:] != self.H.shape[1:]
                or self.F.ndim != 2 or self.F.shape[0] != self.n
                or self.Y.shape != (self.m, self.F.shape[1])):
            raise ValueError('Inconsistent feature/label/assignment shapes')
        if (not all(np.isfinite(v).all() for v in (self.H, self.F, self.C, self.Y))
                or (self.F < 0).any() or (self.Y <= 0).any()
                or not np.allclose(self.F.sum(1), 1) or not np.allclose(self.Y.sum(1), 1)):
            raise ValueError('Features must be finite; targets and labels must be probabilities')
        if not np.isfinite([self.scale, self.kl_scale, self.lam, self.mu]).all() or min(self.scale, self.kl_scale) <= 0 or min(self.lam, self.mu) < 0:
            raise ValueError('Invalid objective scales/weights')
        self.counts = np.bincount(self.a, minlength=self.m)
        if (self.counts == 0).any():
            raise ValueError('Initial cells must be nonempty')
        self.entropy = (self.F * np.log(np.maximum(self.F, 1e-12))).sum(1)
        self.order = np.random.default_rng(seed).permutation(self.n)
        self.history = []

    def costs(self):
        return cdist(self.H, self.C)/self.scale + self.mu*(
            self.entropy[:, None] - self.F @ np.log(self.Y).T)/self.kl_scale

    def objective(self):
        feature = float(np.linalg.norm(self.H-self.C[self.a], axis=1).mean())
        kl = float((self.entropy-(self.F*np.log(self.Y[self.a])).sum(1)).mean())
        structure = cut_cost(self.P, self.a)/self.n
        return dict(J=feature/self.scale+self.mu*kl/self.kl_scale+self.lam*structure,
                    feature=feature, kl=kl, structure=structure,
                    residual_l1=2*structure)

    def run(self, outer_iters=20, log=print):
        if outer_iters < 0:
            raise ValueError('outer_iters must be nonnegative')
        self.history = [dict(stage='initial', **self.objective())]
        for iteration in range(outer_iters):
            before = self.history[-1]['J']
            costs = self.costs()
            moved = 0
            for node in self.order:
                old = self.a[node]
                if self.counts[old] == 1:
                    continue
                delta = costs[node]-costs[node, old]
                if self.lam:
                    delta += self.lam*move_cut_deltas(self.P, self.PT, self.a, node, self.m)
                dest = int(delta.argmin())
                if delta[dest] < -1e-10:
                    self.a[node] = dest
                    self.counts[old] -= 1
                    self.counts[dest] += 1
                    moved += 1
            assignment_J = self.objective()['J']
            if assignment_J > before + 1e-9*max(1., abs(before)):
                raise RuntimeError('Assignment increased the exact objective')
            self.C = cell_geometric_medians(self.H, self.a, self.m, self.C)[0]
            self.Y = np.zeros_like(self.Y)
            np.add.at(self.Y, self.a, self.F)
            self.Y /= self.counts[:, None]
            # Teacher probabilities in the runner are strictly positive.
            self.Y = np.maximum(self.Y, 1e-300)
            self.Y /= self.Y.sum(1, keepdims=True)
            record = dict(stage=f'iteration_{iteration+1}', moved=moved, **self.objective())
            if record['J'] > assignment_J + 1e-9*max(1., abs(assignment_J)):
                raise RuntimeError('Center update increased the exact objective')
            self.history.append(record)
            log(record)
            if before-record['J'] <= 1e-7*max(1., abs(before)):
                break
        return dict(x=self.C, y=self.Y, assign=self.a, counts=self.counts,
                    history=self.history, feature_scale=self.scale, kl_scale=self.kl_scale)
