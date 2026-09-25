from itertools import product

import numpy as np
import pytest
import torch

from src.risk_partition import risk_partition
from src.risk_sdp import normalize_features, partition_value, dual_lower_bound, solve_risk_sdp, round_risk_sdp


def data():
    H = np.array([[-3., 0.], [-2., 1.], [-1., -.5], [1., .3], [2., 1.], [3., -.2]])
    Q = np.array([[.9, .1], [.7, .3], [.8, .2], [.3, .7], [.1, .9], [.2, .8]])
    return H, Q


def test_matrix_form_matches_all_hard_partitions():
    H, Q = data()
    X, B = normalize_features(H), 1.7
    for tail in product(range(2), repeat=5):
        a = np.array((0,) + tail)
        if a.max() == 0:
            continue
        A = np.eye(2)[a]
        Z = (A / A.sum(0)) @ A.T
        residual = X.T @ (np.eye(6) - Z)
        value = B * B / 4 * np.trace(residual @ X) / 6 + 2 * B * np.linalg.norm(residual @ Q) / 6
        assert partition_value(X, Q, a, B) == pytest.approx(value, abs=1e-12)


def test_repaired_arbitrary_dual_is_below_exhaustive_optimum():
    H, Q = data()
    X, B = normalize_features(H), 1.7
    optimum = min(partition_value(X, Q, np.array((0,) + a), B)
                  for a in product(range(2), repeat=5) if 1 in a)
    rng = np.random.default_rng(13)
    for _ in range(5):
        result = dual_lower_bound(X, Q, 2, B, rng.normal(size=(2, 2)),
            rng.normal(size=6), -4., rng.normal(size=(6, 6)))
        assert result['lower_bound'] <= optimum + 1e-10
        assert np.linalg.norm(result['dual_U']) <= 1
        assert result['dual_M'].min() >= 0
        assert result['dual_repaired_min_eigenvalue'] > 0


@pytest.mark.parametrize('m', [1, 2, 6])
def test_solver_bound_and_rounding(m):
    H, Q = data()
    X, B = normalize_features(H), 1.7
    result = solve_risk_sdp(H, Q, m, B, eps=1e-7, max_iters=30000)
    if m == 2:
        optimum = min(partition_value(X, Q, np.array((0,) + a), B)
                      for a in product(range(2), repeat=5) if 1 in a)
    else:
        optimum = partition_value(X, Q, np.zeros(6, dtype=int) if m == 1 else np.arange(6), B)
    assert result['lower_bound'] <= optimum + 1e-7
    assert result['relaxed_value'] <= optimum + 1e-4
    if m in (1, 6):
        assert abs(result['lower_bound'] - optimum) < 1e-3
    for seed in range(3):
        assignment = round_risk_sdp(result['Z'], m, seed)
        assert len(np.unique(assignment)) == m
        state = dict(assignment=torch.as_tensor(assignment),
                     generator_state=torch.Generator().manual_seed(seed).get_state())
        refined = risk_partition(torch.from_numpy(H), torch.from_numpy(Q), m, B,
            initial_state=state, max_sweeps=20, return_assignment=True)
        value = partition_value(X, Q, refined['assignment'].numpy(), B)
        assert refined['J'] == pytest.approx(value, abs=1e-10)
        assert refined['J'] <= partition_value(X, Q, assignment, B) + 1e-10
        assert result['lower_bound'] <= value + 1e-7


def test_constant_embedding_has_nonempty_rounding():
    assignment = round_risk_sdp(np.zeros((8, 8)), 4, seed=0)
    assert len(np.unique(assignment)) == 4


def test_dense_limit_is_explicit():
    H, Q = data()
    with pytest.raises(ValueError, match='explicitly increase'):
        solve_risk_sdp(H, Q, 2, 1., max_nodes=5)
