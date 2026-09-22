"""Checks of src/partition_struct.py (run: python -m pytest tests/test_partition_struct.py -q)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from src.partition_ot import build_transition, transition_to_edges
from src.partition_struct import (StructureBound, column_sum_max, bound_coefficients, cell_geometric_medians,
                                  geometric_median, MOVE_TOL)
from src.models import PropGNN

rng = np.random.default_rng(0)
quiet = lambda *a: None


def toy(N=50, d=4, C=3, m=5, directed=False, isolated=2):
    y = rng.integers(0, C, N)
    H = rng.normal(size=(N, d)) + 2.0 * np.eye(C, d)[y]
    prob = np.where(y[:, None] == y[None, :], .3, .04)
    A = rng.random((N, N)) < prob
    if not directed:
        A = np.triu(A, 1); A = A | A.T
    np.fill_diagonal(A, False)
    A[:isolated, :] = False; A[:, :isolated] = False          # isolated nodes -> self transition
    src, dst = np.nonzero(A)
    P, meta = build_transition(np.vstack([src, dst]), N)
    assert meta['num_isolated'] == isolated
    lg = 2.5 * np.eye(C)[y] + rng.normal(size=(N, C))
    F = np.exp(lg); F /= F.sum(1, keepdims=True)
    a = rng.integers(0, m, N)
    for j in range(m):
        if not (a == j).any():
            a[rng.integers(0, N)] = j
    return H, P, F, m, a


def full_J(sb):
    return sb.evaluate_objective()[0]


def test_row_stochastic_counterexample():                  # 1.
    P = sp.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0]]))
    M = np.array([[1.0], [0.0]])
    n21 = lambda X: np.linalg.norm(X, axis=1).sum()
    assert n21(M) == 1.0 and n21(P @ M) == 2.0
    assert column_sum_max(P) == 2.0
    assert n21(P @ M) <= column_sum_max(P) * n21(M)


def test_move_delta_matches_full_recomputation():          # 2. directed / self-loop / isolated, both Q modes
    for directed in (False, True):
        for q_mode in ('identity', 'learned_median'):
            H, P, F, m, a = toy(directed=directed)
            sb = StructureBound(H, P, F, m, a, alpha=1.3, beta=0.7, mu=0.5, q_mode=q_mode, log=quiet)
            from scipy.spatial.distance import cdist
            DH, KL = cdist(sb.H, sb.Hc), sb._kl_rows()
            for t in [0, 1, 7, 23, 41]:                      # 0, 1 are isolated (P[t,t] = 1)
                dJ, dDH, dDS, dKL = sb.exact_move_delta(t, DH, KL)
                J0 = full_J(sb)
                for b in range(m):
                    if b == sb.a[t]:
                        assert dJ[b] == 0.0
                        continue
                    trial = StructureBound(H, P, F, m, sb.a, 1.3, 0.7, 0.5, q_mode, log=quiet)
                    trial.Hc, trial.Yc, trial.Q = sb.Hc, sb.Yc, sb.Q
                    trial.a = sb.a.copy(); trial.a[t] = b
                    if np.bincount(trial.a, minlength=m).min() == 0:
                        continue
                    J1 = trial.evaluate_objective()[0]
                    assert abs((J1 - J0) - dJ[b]) < 1e-9, (directed, q_mode, t, b, J1 - J0, dJ[b])


def test_apply_move_keeps_caches_consistent():
    H, P, F, m, a = toy(directed=True)
    sb = StructureBound(H, P, F, m, a, q_mode='learned_median', log=quiet)
    from scipy.spatial.distance import cdist
    DH, KL = cdist(sb.H, sb.Hc), sb._kl_rows()
    for t in [3, 9, 0, 15]:
        dJ, dDH, dDS, dKL = sb.exact_move_delta(t, DH, KL)
        b = int(np.argmin(dJ))
        if b != sb.a[t] and sb.counts[sb.a[t]] > 1:
            sb.apply_move(t, b, dDH[b], dDS[b], dKL[b])
    sb.verify()
    assert np.allclose(sb.R, sb.B - sb.Q[sb.a]) and np.allclose(sb.nsq, (sb.R ** 2).sum(1))


def test_full_run_monotone_and_valid():                    # 3, 4.
    for q_mode in ('identity', 'learned_median'):
        H, P, F, m, a = toy()
        out = StructureBound(H, P, F, m, a, q_mode=q_mode, log=quiet).run(outer_iters=4)
        J = [h['J'] for h in out['history']]
        assert np.isfinite(J).all() and all(J[i + 1] <= J[i] + 1e-9 for i in range(len(J) - 1)), J
        assert (out['cell_counts'] > 0).all() and out['cell_counts'].sum() == len(H)
        Q = out['Q']
        assert (Q >= -1e-12).all() and np.allclose(Q.sum(1), 1)
        assert np.allclose(out['Y_cond'].sum(1), 1) and (out['Y_cond'] >= 0).all()


def test_q_modes():                                        # 5.
    H, P, F, m, a = toy()
    sb = StructureBound(H, P, F, m, a, q_mode='identity', log=quiet)
    sb.run(outer_iters=3)
    assert np.array_equal(sb.Q, np.eye(m))
    sb = StructureBound(H, P, F, m, a, q_mode='learned_median', log=quiet)
    means = sb.B.copy(); Qmean = np.zeros((m, m)); np.add.at(Qmean, sb.a, means); Qmean /= sb.counts[:, None]
    cost = lambda Q: np.linalg.norm(sb.B - Q[sb.a], axis=1).sum()
    Qmed, g0, g1 = cell_geometric_medians(sb.B, sb.a, m, Qmean)
    assert cost(Qmed) <= cost(Qmean) + 1e-12 and abs(g0.sum() - cost(Qmean)) < 1e-9 and abs(g1.sum() - cost(Qmed)) < 1e-9


def test_beta_zero_is_feature_kl_objective():             # 6.
    H, P, F, m, a = toy()
    sb = StructureBound(H, P, F, m, a, beta=0.0, log=quiet)
    out = sb.run(outer_iters=3)
    for h in out['history']:
        assert abs(h['J'] - (sb.alpha * h['D_H'] + sb.mu * h['D_KL'])) < 1e-12
    J = [h['J'] for h in out['history']]
    assert all(J[i + 1] <= J[i] + 1e-9 for i in range(len(J) - 1))


def test_recursion_and_ce_bounds_numerically():            # 7.
    H, P, F, m, a = toy()
    sb = StructureBound(H, P, F, m, a, q_mode='learned_median', log=quiet); sb.run(outer_iters=2)
    S = sb._S().toarray()
    Ws = [rng.normal(size=(H.shape[1], 6)) / 3, rng.normal(size=(6, F.shape[1])) / 3]
    Pd = sb.P.toarray()
    Z, Zc = sb.H, sb.Hc
    n21 = lambda X: np.linalg.norm(X, axis=1).sum()
    R_S = n21(Pd @ S - S @ sb.Q)
    cP = sb.cP
    E = [n21(Z - S @ Zc)]
    for l, W in enumerate(Ws):
        act = (lambda X: np.maximum(X, 0)) if l < len(Ws) - 1 else (lambda X: X)
        Zn, Zcn = act(Pd @ Z @ W), act(sb.Q @ Zc @ W)
        a_l = np.linalg.norm(W, 2)
        M_l = np.linalg.norm(Zc, 2)
        E.append(n21(Zn - S @ Zcn))
        assert E[-1] <= a_l * (cP * E[-2] + R_S * M_l) + 1e-9
        Z, Zc = Zn, Zcn
    logp = lambda z: z - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)) - z.max(1, keepdims=True)
    ce = lambda y, z: -(y * logp(z)).sum(1)
    assert (ce(sb.F, Z) <= ce(sb.F, (S @ Zc)) + np.sqrt(2) * np.linalg.norm(Z - S @ Zc, axis=1) + 1e-9).all()
    alpha, beta = bound_coefficients(cP, 2, 1.0, 1.0, m)
    assert alpha > 0 and beta > 0


def test_operator_matches_student():                       # 8.
    H, P, F, m, a = toy(directed=True)
    ei, ea = transition_to_edges(P)
    data = Data(x=torch.tensor(H, dtype=torch.float32), edge_index=torch.from_numpy(ei).long(), edge_attr=torch.from_numpy(ea).float())
    model = PropGNN(H.shape[1], 4, H.shape[1], 1, 0.0).eval()
    with torch.no_grad():
        torch.nn.init.eye_(model.lins[0].weight)
        out = model(data)
    ref = torch.log_softmax(torch.tensor(P @ H, dtype=torch.float32), dim=1)
    assert torch.allclose(out, ref, atol=1e-5)
    assert np.allclose(np.asarray(P.sum(1)).ravel(), 1) and (P.diagonal()[2:] == 0).all() and (P.diagonal()[:2] == 1).all()


def test_geometric_median_coincident_and_simplex():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    z, f = geometric_median(X, X[2])
    assert np.allclose(z, [0.5, 0.5]) and abs(f - np.sqrt(2)) < 1e-9
    assert (z >= 0).all() and abs(z.sum() - 1) < 1e-12
