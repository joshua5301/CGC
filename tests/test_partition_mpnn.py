"""Independent objective/adjoint checks and a bounded nonlinear MPNN example."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from src.partition_mpnn import MPNNIdentity, closed_transition


def fixture(batch=2, depth=2, device='cpu'):
    torch.manual_seed(7)
    x = torch.randn(15, 4, dtype=torch.float64).to(device)
    # Deliberately directed, duplicates, existing self-loop, and isolated nodes.
    edges = torch.tensor([[0, 0, 0, 2, 3, 4, 5, 6, 0],
                          [1, 1, 0, 1, 2, 2, 4, 5, 3]], device=device)
    f = torch.softmax(torch.randn(15, 3, dtype=torch.float64, device=device), dim=1)
    return MPNNIdentity(x, edges, f, 3, torch.arange(15, device=device) % 3,
                        batch_size=batch, depth=depth, mu=.3, log=lambda _: None), edges


def test_closed_neighborhood_direction_duplicates_and_isolation():
    _, edges = fixture()
    p, degree = closed_transition(edges, 15)
    dense = p.to_dense()
    torch.testing.assert_close(dense.sum(1), torch.ones(15, dtype=torch.float64))
    assert degree[1] == 3  # itself, 0, 2: duplicate 0->1 contributes once
    assert degree[0] == 1  # existing self-loop is not doubled
    assert dense[14, 14] == 1
    assert dense[1, 0] == 1/3 and dense[0, 1] == 0


def test_dense_multihop_objective_and_transposed_center_weights():
    obj, edges = fixture()
    p, _ = closed_transition(edges, 15)
    p = p.to_dense()
    d = torch.linalg.vector_norm(obj.x[:, None] - obj.z[None, :], dim=2)
    expected = (d + 2 * p @ d + p @ p @ d) / 4
    torch.testing.assert_close(obj.feature_cost(0, 3), expected)
    assert torch.count_nonzero(obj.q) == 0
    s = torch.nn.functional.one_hot(obj.assign, 3).double()
    torch.testing.assert_close(expected[torch.arange(15), obj.assign].sum(),
                               (obj.propagate(s, transpose=True) * d).sum())


def test_raw_ablation_and_initialization_match():
    raw, _ = fixture(depth=0)
    mpnn, _ = fixture(depth=2)
    torch.testing.assert_close(raw.z, mpnn.z)
    torch.testing.assert_close(raw.assign, mpnn.assign)
    torch.testing.assert_close(raw.feature_cost(0, 3), torch.cdist(raw.x, raw.z))


def test_descent_nonempty_simplex_and_batching():
    obj, _ = fixture(1)
    other, _ = fixture(3)
    a, b = obj.run(8), other.run(8)
    assert np.all(np.diff([r['J'] for r in a['history']]) <= 1e-8)
    assert a['cell_counts'].min() >= 1
    torch.testing.assert_close(a['Y_cond'].sum(1), torch.ones(3))
    torch.testing.assert_close(a['H_cond'], b['H_cond'])
    assert a['config']['mass_correction'] is False
    assert a['config']['risk_certificate'] is False
    assert a['diagnostics']['imbalance_multiplier'] >= 1


@pytest.mark.parametrize('aggregation', ['sum', 'mean', 'max', 'attention'])
def test_common_sensitivity_bound_with_structural_remainder(aggregation):
    # Scalar messages, root skip, and nonlinear attention on a bounded domain.
    # For attention sum softmax(z)_i z_i, one-coordinate derivative is bounded
    # by 1+2R for |z_i|<=R. One deletion changes mean/max/attention by <=2R;
    # SUM changes by <=R. Thus the SAME budgets cover all four aggregators.
    x = torch.tensor([-.4, .7, 1., -.8], dtype=torch.float64)
    c = torch.tensor([-.2, .5], dtype=torch.float64)
    edges = torch.tensor([[0, 2, 3, 0], [1, 1, 1, 2]])
    p, degree = closed_transition(edges, 4)
    p = p.to_dense()
    r = (torch.eye(4, dtype=torch.float64) + p) / 2
    d = (x[:, None] - c).abs()
    h, hc = x.clone(), c.clone()
    remainder = torch.zeros(4, dtype=torch.float64)
    scale, radius = 1., 1.
    for _ in range(2):
        values = []
        for v in range(4):
            neighbors = h[p[v] > 0]
            if aggregation == 'sum':
                agg = neighbors.sum()
            elif aggregation == 'mean':
                agg = neighbors.mean()
            elif aggregation == 'max':
                agg = neighbors.max()
            else:
                agg = (neighbors.softmax(0) * neighbors).sum()
            values.append(h[v] + agg)
        h, hc = torch.stack(values), 2 * hc
        common_lipschitz = 1 + 2 * radius
        remainder = 2 * common_lipschitz * r @ remainder + 2 * radius * (degree - 1)
        scale *= 2 * common_lipschitz
        d = r @ d
        assert torch.all((h[:, None] - hc).abs() <= scale * d + remainder[:, None] + 1e-10)
        radius *= 1 + float(degree.max())


def test_constant_features_need_structural_remainder():
    x = torch.ones(4, 1, dtype=torch.float64)
    edges = torch.tensor([[0, 2, 3], [1, 1, 1]])
    p, degree = closed_transition(edges, 4)
    assert torch.cdist(x, x[:1]).sum() == 0
    # SUM gives degree, while the condensed self-loop gives one.
    assert degree[1] - 1 == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_matches_cpu():
    cpu, edges = fixture()
    gpu = MPNNIdentity(cpu.x.cuda(), edges.cuda(), cpu.f.cuda(), 3, cpu.assign.cuda(),
                       batch_size=2, mu=.3, log=lambda _: None)
    a, b = cpu.run(4), gpu.run(4)
    np.testing.assert_allclose(a['history'][-1]['J'], b['history'][-1]['J'], rtol=1e-6)
    torch.testing.assert_close(a['H_cond'], b['H_cond'].cpu(), rtol=1e-5, atol=1e-5)
