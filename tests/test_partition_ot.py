"""Correctness checks of src/partition_ot.py (run: python -m pytest tests -q)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial.distance import cdist
from src.partition_ot import (build_transition, transition_to_edges, pair_neighbor_ot, solve_cell_neighbor_lp,
                              weighted_geometric_medians, NeighborhoodOT, LP_FEAS_TOL)
from src.models import SAGE
from torch_geometric.data import Data

rng = np.random.default_rng(0)


def toy_graph(N=40, d=3, C=3, m=5, p_in=0.5, p_out=0.05):
    y = rng.integers(0, C, N)
    H = rng.normal(size=(N, d)) + 2.0 * np.eye(C)[y]
    prob = np.where(y[:, None] == y[None, :], p_in, p_out)
    A = np.triu(rng.random((N, N)) < prob, 1)
    A = A | A.T
    src, dst = np.nonzero(A)
    P, meta = build_transition(np.vstack([src, dst]), N)
    logits = 3.0 * np.eye(C)[y] + rng.normal(size=(N, C))
    Fp = np.exp(logits); Fp /= Fp.sum(1, keepdims=True)
    assign = rng.integers(0, m, N)
    for j in range(m):                                            # non-empty initial assignment
        if not (assign == j).any():
            assign[rng.integers(0, N)] = j
    return H, P, Fp, m, assign


def test_fixed_ot_toy():                                          # 1. mass .5 at 0 and 4 vs .5 at 1 and 3 -> W1 = 1
    src, dst = np.array([[0.0], [4.0]]), np.array([[1.0], [3.0]])
    val, G = pair_neighbor_ot(cdist(src, dst), np.array([.5, .5]), np.array([.5, .5]))
    assert abs(val - 1.0) < 1e-9
    assert np.allclose(G.sum(1), .5) and np.allclose(G.sum(0), .5)


def test_cell_lp_marginals_and_bound():                           # 2. cell LP feasibility and <= fixed-p solution
    H, P, Fp, m, assign = toy_graph()
    Z = rng.normal(size=(m, H.shape[1]))
    cell = np.flatnonzero(assign == 0)
    costs = [cdist(H[P.indices[P.indptr[t]:P.indptr[t + 1]]], Z) for t in cell]
    masses = [P.data[P.indptr[t]:P.indptr[t + 1]] for t in cell]
    out = solve_cell_neighbor_lp(costs, masses, m)
    p = out['p']
    assert (p >= 0).all() and abs(p.sum() - 1) < 1e-8 and out['residual'] < LP_FEAS_TOL
    for G, a in zip(out['gammas'], masses):
        assert np.allclose(G.sum(1), a, atol=1e-7) and np.allclose(G.sum(0), p, atol=1e-7) and (G >= -1e-12).all()
    p0 = rng.random(m); p0 /= p0.sum()                            # any fixed row is feasible for the joint LP
    fixed = sum(pair_neighbor_ot(c, a, p0)[0] for c, a in zip(costs, masses))
    assert out['value'] <= fixed + 1e-8


def test_weighted_median_nonincrease_with_coincident_points():   # 3.
    H = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [3.0, 3.0]])
    omega = np.array([[5.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 0.0]])   # column 0: heavy weight on point 0
    Z0 = np.array([[0.0, 0.0], [2.0, 2.0]])                              # column 0 starts exactly on point 0
    Z, f0, f1 = weighted_geometric_medians(H, omega, Z0)
    assert (f1 <= f0 + 1e-12).all()
    assert np.allclose(Z[0], H[0])                                       # the heavy point is the median (weight 5 > 3)
    D = cdist(H, Z); assert (omega * D).sum(0)[1] <= f0[1]


def test_tiny_full_run_monotone():                                # 4.
    H, P, Fp, m, assign = toy_graph()
    logs = []
    out = NeighborhoodOT(H, P, Fp, m, assign, log=logs.append).run(outer_iters=3)
    J = [h['J'] for h in out['history']]
    assert np.isfinite(J).all()
    assert all(J[i + 1] <= J[i] + 1e-8 for i in range(len(J) - 1)), J
    assert (out['cell_counts'] > 0).all() and out['cell_counts'].sum() == len(H)
    Pc = out['P_cond']
    assert (Pc >= 0).all() and np.allclose(Pc.sum(1), 1, atol=1e-8)
    assert np.allclose(out['Y_cond'].sum(1), 1)


def test_beta_zero_reduces_to_root_kl():                          # 5.
    H, P, Fp, m, assign = toy_graph()
    logs = []
    out = NeighborhoodOT(H, P, Fp, m, assign, beta=0.0, log=logs.append).run(outer_iters=3)
    for h in out['history']:
        assert h['w1'] == 0.0
    J = [h['J'] for h in out['history']]
    assert all(J[i + 1] <= J[i] + 1e-8 for i in range(len(J) - 1))
    assert np.allclose(out['P_cond'], NeighborhoodOT(H, P, Fp, m, assign, beta=0.0, log=logs.append).Pc)  # unchanged


def test_operator_direction():                                    # 6. sparse edge form == P @ H, no hidden self-loops
    H, P, Fp, m, assign = toy_graph()
    ei, ea = transition_to_edges(P)
    data = Data(x=torch.tensor(H, dtype=torch.float32), edge_index=torch.from_numpy(ei).long(), edge_attr=torch.from_numpy(ea).float())
    model = SAGE(H.shape[1], 4, 3, 1, 0.0).eval()
    with torch.no_grad():
        torch.nn.init.zeros_(model.root[0].weight); torch.nn.init.zeros_(model.root[0].bias); torch.nn.init.eye_(model.nbr[0].weight[:, :3])
        out = model(data)
    ref = torch.log_softmax(torch.tensor(P @ H, dtype=torch.float32), dim=1)
    assert torch.allclose(out, ref, atol=1e-5)
    assert P.diagonal().sum() == 0 and np.allclose(np.asarray(P.sum(1)).ravel(), 1)


def test_isolated_and_singleton():                               # 7.
    N = 6
    src, dst = np.array([0, 1, 1, 2]), np.array([1, 0, 2, 1])          # node 3, 4, 5 isolated
    P, meta = build_transition(np.vstack([src, dst]), N)
    assert meta['num_isolated'] == 3 and P[3, 3] == 1 and P[0, 0] == 0
    H = rng.normal(size=(N, 2)); Fp = np.full((N, 2), .5)
    assign = np.array([0, 0, 0, 1, 2, 3])                            # three singleton cells
    logs = []
    out = NeighborhoodOT(H, P, Fp, 4, assign, log=logs.append).run(outer_iters=2)
    assert (out['cell_counts'] > 0).all()
    J = [h['J'] for h in out['history']]
    assert all(J[i + 1] <= J[i] + 1e-8 for i in range(len(J) - 1))
    single = NeighborhoodOT(H[:1], sp.csr_matrix(np.ones((1, 1))), Fp[:1], 1, np.array([0]), log=logs.append).run(outer_iters=1)
    assert single['history'][-1]['J'] == 0.0
