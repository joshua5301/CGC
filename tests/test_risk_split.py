import torch

from src.risk_split import split_delta, split_partition


def test_delta_matches_direct_moments():
    x = torch.tensor([[0., 1.], [1., 0.], [3., 2.], [4., -1.]], dtype=torch.float64)
    q = torch.tensor([[.9, .1], [.7, .3], [.2, .8], [.1, .9]], dtype=torch.float64)
    error = (x - x.mean(0)).T @ (q - q.mean(0)) / 4
    variance = (x - x.mean(0)).square().sum(1).mean()
    a, b = x[:2].mean(0) - x[2:].mean(0), q[:2].mean(0) - q[2:].mean(0)
    alpha, beta = .7, 1.3
    actual_e = sum((xx - xx.mean(0)).T @ (qq - qq.mean(0)) for xx, qq in
                   ((x[:2], q[:2]), (x[2:], q[2:]))) / 4
    actual_v = sum((xx - xx.mean(0)).square().sum() for xx in (x[:2], x[2:])) / 4
    expected = alpha * (actual_v - variance) + beta * (actual_e.norm() - error.norm())
    actual = split_delta(error, a, b, x.new_tensor(.25), alpha, beta)
    torch.testing.assert_close(actual, expected)


def test_zero_features_reach_exact_nonempty_budget():
    x = torch.zeros(6, 2, dtype=torch.float64)
    q = torch.full((6, 2), .5, dtype=torch.float64)
    assignment, info = split_partition(x, q, 6, 1., torch.Generator().manual_seed(0))
    assert torch.bincount(assignment).tolist() == [1] * 6
    assert info['split_history'] == [0.] * 6


def test_split_trace_matches_final_direct_objective():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(12, 3, generator=generator, dtype=torch.float64)
    q = torch.randn(12, 2, generator=generator, dtype=torch.float64).softmax(1)
    assignment, info = split_partition(x, q, 4, 2., generator)
    error, variance = torch.zeros(3, 2, dtype=x.dtype), x.new_tensor(0.)
    for cell in range(4):
        xx, qq = x[assignment == cell], q[assignment == cell]
        variance += (xx - xx.mean(0)).square().sum() / len(x)
        error += (xx - xx.mean(0)).T @ (qq - qq.mean(0)) / len(x)
    torch.testing.assert_close(x.new_tensor(info['split_history'][-1]), variance + 4 * error.norm())
