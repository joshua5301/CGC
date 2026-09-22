"""Checks of the entropic solver (src/partition_ot_entropic.py) against the exact one."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from src.partition_ot import build_transition, NeighborhoodOT
from src.partition_ot_entropic import NeighborhoodOTEntropic

rng = np.random.default_rng(0)
quiet = lambda *a: None


def toy(N=60, d=4, C=3, m=6):
    y = rng.integers(0, C, N)
    H = rng.normal(size=(N, d)) + 2.0 * np.eye(C, d)[y]
    prob = np.where(y[:, None] == y[None, :], .3, .03)
    A = np.triu(rng.random((N, N)) < prob, 1); A = A | A.T
    P, _ = build_transition(np.vstack(np.nonzero(A)), N)
    lg = 2.5 * np.eye(C)[y] + rng.normal(size=(N, C))
    F = np.exp(lg); F /= F.sum(1, keepdims=True)
    assign = rng.integers(0, m, N)
    for j in range(m):
        if not (assign == j).any():
            assign[rng.integers(0, N)] = j
    return H, P, F, m, assign


def test_monotone_and_valid_output():
    H, P, F, m, assign = toy()
    out = NeighborhoodOTEntropic(H, P, F, m, assign, eps_rel=0.05, device='cpu', log=quiet).run(outer_iters=3)
    J = [h['J'] for h in out['history']]
    assert np.isfinite(J).all()
    assert all(J[i + 1] <= J[i] + 1e-7 for i in range(len(J) - 1)), J
    assert (out['cell_counts'] > 0).all() and out['cell_counts'].sum() == len(H)
    Pc = out['P_cond']
    assert (Pc >= 0).all() and np.allclose(Pc.sum(1), 1, atol=1e-6)
    assert np.allclose(out['Y_cond'].sum(1), 1)


def test_approaches_exact_as_eps_decreases():
    """The sharp transport cost of the entropic solution converges to the exact LP objective."""
    H, P, F, m, assign = toy()
    ex = NeighborhoodOT(H, P, F, m, assign, log=quiet); ex.run(outer_iters=3)
    w1_exact = ex.history[-1]['w1']
    errs = []
    for eps_rel in (0.2, 0.02):
        en = NeighborhoodOTEntropic(H, P, F, m, assign, eps_rel=eps_rel, device='cpu', log=quiet)
        en.run(outer_iters=3)
        errs.append(abs(en.history[-1]['w1_sharp'] - w1_exact))
    assert errs[1] < errs[0], errs
    assert errs[1] < 0.02 * w1_exact, errs


def test_beta_zero_matches_root_kl_descent():
    H, P, F, m, assign = toy()
    en = NeighborhoodOTEntropic(H, P, F, m, assign, beta=0.0, device='cpu', log=quiet)
    out = en.run(outer_iters=3)
    for h in out['history']:
        assert h['w1'] == 0.0
    ex = NeighborhoodOT(H, P, F, m, assign, beta=0.0, log=quiet).run(outer_iters=3)
    assert abs(out['history'][-1]['J'] - ex['history'][-1]['J']) < 1e-6      # same blocks, no transport involved


def test_pair_cost_matches_exact_ot_at_small_eps():
    """One node's entropic cost against every cell vs the exact pair OT of the reference implementation."""
    H, P, F, m, assign = toy()
    en = NeighborhoodOTEntropic(H, P, F, m, assign, eps_rel=0.005, sink_iters=2000, candidates=m, device='cpu', log=quiet)
    cand = torch.arange(m).expand(en.N, m)
    approx = en._candidate_costs(cand).numpy()
    ex = NeighborhoodOT(H, P, F, m, assign, log=quiet)
    ex.Z, ex.Pc, ex.cost_all = en.Z.numpy(), en.Pc.numpy(), None
    for t in (0, 7, 23):
        exact = np.array([ex.w1(t, j)[0] for j in range(m)])
        assert np.abs(approx[t] - exact).max() < 5e-2 * max(exact.mean(), 1e-9), (t, approx[t], exact)
