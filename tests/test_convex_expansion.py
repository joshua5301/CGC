import torch

from src.convex_expansion import linear_oracle, refine_hull
from src.empirical_ntk_study import make_network


def test_block_oracle_searches_all_nodes_and_respects_cluster_support():
    h = torch.tensor([[.1], [.3], [1.], [2.]], dtype=torch.float64)
    gradient = torch.tensor([[-1.], [1.]], dtype=torch.float64)
    assignment = torch.tensor([0, 0, 1, 1])
    ids, values = linear_oracle(h, gradient, block_size=1)
    assert ids.tolist() == [3, 0]
    torch.testing.assert_close(values, torch.tensor([-2., .1], dtype=torch.float64))
    ids, _ = linear_oracle(h, gradient, assignment, block_size=2)
    assert ids.tolist() == [1, 2]


def test_global_expansion_improves_and_saved_recipe_is_convex(tmp_path):
    h = torch.tensor([[.1], [.3], [1.], [2.]])
    model = make_network(h, torch.empty(2, 0, dtype=torch.long), 'gcn', 1, 1, 0)
    with torch.no_grad():
        model.layers[0].lin.weight.fill_(1)
        model.layers[0].bias.zero_()
    target = torch.tensor([[1.]])
    initial = torch.tensor([[.2]])
    within = refine_hull(model, h, target, initial, tmp_path / 'within.pt',
                         assignment=torch.tensor([0, 0, 1, 1]), max_steps=100, tolerance=1e-8)
    torch.testing.assert_close(within['x'], torch.tensor([[.3]], dtype=torch.float64))
    global_fit = refine_hull(model, h, target, within['x'], tmp_path / 'global.pt', max_steps=1000, tolerance=1e-7)
    assert global_fit['converged']
    assert float((global_fit['x'] - target).square().sum()) < 1e-10
    reconstructed = within['x'].clone()
    for ids, rate in zip(global_fit['vertices'], global_fit['rates']):
        assert bool(((rate >= 0) & (rate <= 1)).all())
        reconstructed = (1 - rate[:, None]) * reconstructed + rate[:, None] * h[ids].double()
    torch.testing.assert_close(reconstructed, global_fit['x'])
    losses = [row['reconstruction_mse'] for row in global_fit['history']]
    assert all(b <= a + 1e-12 for a, b in zip(losses, losses[1:]))
    resumed = refine_hull(model, h, target, within['x'], tmp_path / 'global.pt', max_steps=1000, tolerance=1e-7)
    torch.testing.assert_close(resumed['x'], global_fit['x'])


def test_iteration_limit_is_not_reported_as_convergence(tmp_path):
    h = torch.tensor([[.1], [2.]])
    model = make_network(h, torch.empty(2, 0, dtype=torch.long), 'gcn', 1, 1, 0)
    with torch.no_grad():
        model.layers[0].lin.weight.fill_(1)
        model.layers[0].bias.zero_()
    result = refine_hull(model, h, torch.tensor([[.8]]), torch.tensor([[.1]]),
                         tmp_path / 'limited.pt', max_steps=1, tolerance=1e-12)
    assert result['status'] == 'iteration_limit'
    assert not result['converged']
