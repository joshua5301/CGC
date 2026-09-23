import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from scipy.optimize import linprog
from torch_geometric import seed_everything
from src.partition import partition
from src.partition_robust import worst_labels, distance_radii, fit_labels, robust_partition


@pytest.mark.parametrize('radius', [0., .03, .5, 1., 2.])
def test_adversary_against_independent_linear_program(radius):
    rng = np.random.default_rng(7)
    for k in [2, 4, 7]:
        p, q = rng.dirichlet(np.ones(k), 2)
        logq = np.log(q)
        # Variables p_worst and absolute deviation auxiliaries.
        A = np.block([[np.eye(k), -np.eye(k)], [-np.eye(k), -np.eye(k)]])
        A = np.vstack([A, np.r_[np.zeros(k), np.ones(k)]])
        b = np.r_[p, -p, radius]
        r = linprog(np.r_[logq, np.zeros(k)], A_ub=A, b_ub=b,
                    A_eq=[np.r_[np.ones(k), np.zeros(k)]], b_eq=[1.], bounds=(0, None), method='highs')
        assert r.success
        got = worst_labels(torch.tensor(p), torch.tensor(logq), torch.tensor(radius))
        assert np.isclose(float(got @ torch.tensor(logq)), r.fun, atol=1e-7)
        assert got.min() >= -1e-12
        assert abs(float(got.sum())-1) < 1e-12
        assert float((got-torch.tensor(p)).abs().sum()) <= radius+1e-7


def test_ties_and_broadcast():
    p = torch.tensor([[.2, .3, .5], [.8, .1, .1]], dtype=torch.float64)
    q = torch.tensor([[1/3]*3, [.1, .1, .8]], dtype=torch.float64)
    result = worst_labels(p[:, None], q[None].log(), torch.ones(2, 1, dtype=torch.float64))
    torch.testing.assert_close(result.sum(-1), torch.ones(2, 2, dtype=torch.float64))
    assert result.min() >= -1e-12
    for i in range(2):
        for j in range(2):
            torch.testing.assert_close(result[i, j], worst_labels(p[i], q[j].log(), torch.tensor(1.)))


def fixture():
    seed_everything(5)
    return torch.randn(30, 4, dtype=torch.float64), torch.randn(30, 3, dtype=torch.float64).softmax(1)


def test_zero_radius_exact_baseline():
    x, f = fixture()
    seed_everything(9)
    expected = partition(x, f, 3, .5)
    seed_everything(9)
    actual = robust_partition(x, f, 3, torch.zeros(len(x)), .5)
    for name, value in zip(['H_cond', 'Y_cond', 'assign'], expected):
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0)


def test_label_update_and_full_uncertainty():
    _, p = fixture()
    a = torch.arange(len(p)) % 3
    initial = torch.tensor([[.9, .05, .05]]*3, dtype=torch.float64)
    labels, gap = fit_labels(p, torch.full((len(p),), 2.), a, initial, steps=20)
    torch.testing.assert_close(labels, torch.full_like(labels, 1/3))
    assert gap >= 0


def test_monotonic_objective_and_risk_inequality():
    x, f = fixture()
    radius = torch.full((len(x),), .2, dtype=torch.float64)
    result = robust_partition(x, f, 3, radius, outer_iters=3, label_steps=20, log=lambda _: None)
    values = [r['objective'] for r in result['history']]
    assert all(b <= a+1e-8 for a, b in zip(values, values[1:]))
    c, y, a = (result[k] for k in ['H_cond', 'Y_cond', 'assign'])
    assert torch.bincount(a).min() > 0
    # Same linear classifier on original H and condensed C; true p in its ball.
    w = torch.randn(4, 3, dtype=torch.float64)
    log_g, log_q = (x@w).log_softmax(1), (c@w).log_softmax(1)
    p = worst_labels(f, log_g, radius)
    risk = -(p*log_g).sum(1).mean()
    adversary = worst_labels(f, y[a].log(), radius)
    robust_ce = -(adversary*y[a].log()).sum(1)
    residual = (y.log()-log_q).max(1).values[a]
    geometry = 2**.5 * torch.linalg.matrix_norm(w, ord=2)*(x-c[a]).norm(dim=1)
    assert risk <= (geometry+robust_ce+residual).mean()+1e-10


def test_distance_radius_is_training_only_and_monotone():
    x = torch.tensor([[0.], [1.], [2.], [5.]], dtype=torch.float64)
    mask = torch.tensor([True, True, False, False])
    r, meta = distance_radii(x, mask, cap=.5, floor=.05)
    torch.testing.assert_close(r[:2], torch.full((2,), .05, dtype=torch.float64))
    assert r[1] < r[2] < r[3] < .5
    assert meta['radius_certificate'] is False
    constant, _ = distance_radii(x, mask, cap=.5, mode='constant')
    torch.testing.assert_close(constant, torch.full_like(r, .5))
