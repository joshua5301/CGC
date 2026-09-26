import copy

import torch

from src.convex_representatives import convex_features, fit_convex_representatives, identity_hidden, median_weights
from src.empirical_ntk_study import make_network
from src.ntk_readout_study import readout_features
from src.partition import geometric_medians


def test_identity_readout_ignores_original_graph_cache():
    x = torch.tensor([[1., 0.], [0., 1.], [2., 1.]])
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    model = make_network(x, edges, 'gcn', 4, 2, 10)
    fresh = copy.deepcopy(model)
    for layer in fresh.layers:
        layer._cached_edge_index = None
        layer._cached_adj_t = None
    representatives = torch.tensor([[.3, .7], [1.2, .4]], requires_grad=True)
    expected, _, _, _ = readout_features(fresh, representatives.detach(), torch.empty(2, 0, dtype=torch.long), 'gcn')
    actual = identity_hidden(model, representatives)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert representatives.grad is not None


def test_coefficients_reproduce_medians_and_cluster_support():
    h = torch.tensor([[0., 0.], [2., 0.], [0., 2.], [4., 4.], [6., 4.], [4., 6.]], dtype=torch.float64)
    assignment = torch.tensor([0, 0, 0, 1, 1, 1])
    weights = median_weights(h, assignment, 2)
    centers = convex_features(h, assignment, weights, 2)
    torch.testing.assert_close(centers, geometric_medians(h, assignment, 2))
    torch.testing.assert_close(torch.zeros(2, dtype=h.dtype).index_add_(0, assignment, weights), torch.ones(2, dtype=h.dtype))
    modified = h.clone()
    modified[3:] += 100
    torch.testing.assert_close(convex_features(modified, assignment, weights, 2)[0], centers[0])


def test_reconstruction_improves_without_changing_teacher():
    h = torch.tensor([[0., 0.], [2., 0.], [0., 2.], [4., 4.], [6., 4.], [4., 6.]])
    assignment = torch.tensor([0, 0, 0, 1, 1, 1])
    model = make_network(h, torch.empty(2, 0, dtype=torch.long), 'gcn', 2, 2, 0)
    with torch.no_grad():
        model.layers[0].lin.weight.copy_(torch.eye(2))
        model.layers[0].bias.zero_()
    original = {name: p.clone() for name, p in model.state_dict().items()}
    baseline = geometric_medians(h.double(), assignment, 2).float()
    target = torch.tensor([[.9, .3], [4.9, 4.3]])
    result = fit_convex_representatives(model, h, assignment, target, baseline, steps=100, lr=.1)
    before = (identity_hidden(model, baseline) - target).square().sum()
    after = (identity_hidden(model, result['x']) - target).square().sum()
    assert after < before
    torch.testing.assert_close(convex_features(h.double(), assignment, result['weights'], 2).float(), result['x'])
    assert bool((result['weights'] >= 0).all())
    for name, p in model.state_dict().items():
        torch.testing.assert_close(p, original[name])
