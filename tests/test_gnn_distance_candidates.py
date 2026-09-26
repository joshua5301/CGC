import numpy as np
import pytest
import torch

from src.gnn_distance_candidates import (build_candidate_distances, empirical_ntk_features,
                                          projected_jacobian, random_gnn_features, scattering_features)
from src.gnn_distance_probe import ProbeGNN
from src.node_distances import neighborhood_mmd_distance


def tiny_graph():
    x = torch.tensor([[1., 0.], [.1, 2.], [2., 1.], [.3, .5]])
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    return x, edges


def test_projected_jacobian_matches_linear_derivative():
    x = torch.tensor([[1., 2.], [0., 3.], [2., -1.]], dtype=torch.float64)
    parameters = (torch.ones(2, 1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
    function = lambda weight, bias: (x @ weight + bias).flatten()
    actual = projected_jacobian(function, parameters, projections=5, seed=10)
    generator = torch.Generator().manual_seed(10)
    expected = []
    for _ in range(5):
        weight = torch.randint(0, 2, (2, 1), generator=generator).double() * 2 - 1
        bias = torch.randint(0, 2, (1,), generator=generator).double() * 2 - 1
        expected.append((x @ weight + bias).flatten().numpy())
    np.testing.assert_allclose(actual, np.stack(expected, axis=1) / np.sqrt(5), atol=1e-12)


@pytest.mark.parametrize('architecture', ['gcn', 'sage', 'gin'])
def test_random_features_are_equivariant_and_do_not_change_rng(architecture):
    x, edges = tiny_graph()
    before = torch.get_rng_state().clone()
    first = random_gnn_features(x, edges, torch.arange(4), architecture, width=8, seed=2000)
    assert torch.equal(before, torch.get_rng_state())
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = permutation.argsort()
    second = random_gnn_features(x[permutation], inverse[edges], torch.arange(4), architecture, width=8, seed=2000)
    np.testing.assert_allclose(second, first[permutation.numpy()], atol=1e-6)


@pytest.mark.parametrize('architecture', ['gcn', 'sage', 'gin'])
def test_projected_gnn_ntk_matches_explicit_parameter_gradients(architecture):
    x, edges = tiny_graph()
    projections, seed, direction_seed = 4, 3000, 4000
    actual = empirical_ntk_features(x, edges, torch.arange(4), architecture, 4, seed, projections, direction_seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = ProbeGNN(architecture, 2, 4, 1, dropout=0.).eval()
    parameters = tuple(network.parameters())
    output = network(x, edges)[:, 0]
    jacobian = []
    for scalar in output:
        gradients = torch.autograd.grad(scalar, parameters, retain_graph=True, allow_unused=True)
        jacobian.append(torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                                   for g, p in zip(gradients, parameters)]))
    jacobian = torch.stack(jacobian)
    generator = torch.Generator().manual_seed(direction_seed)
    directions = []
    for _ in range(projections):
        directions.append(torch.cat([(torch.randint(0, 2, p.shape, generator=generator).float() * 2 - 1).flatten()
                                     for p in parameters]))
    expected = (jacobian @ torch.stack(directions).T).detach().numpy() / np.sqrt(projections)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_sum_mmd_retains_duplicate_neighbor_count():
    x = np.array([[0.], [0.], [4.], [4.]])
    edges = np.array([[2, 2, 3], [0, 1, 1]])
    settings = dict(depth=1, width=32, seed=2026)
    mean, _ = neighborhood_mmd_distance(x, edges, np.arange(4), **settings)
    total, _ = neighborhood_mmd_distance(x, edges, np.arange(4), aggregation='sum', **settings)
    np.testing.assert_allclose(mean[0, 1], 0., atol=1e-12)
    assert total[0, 1] > .01


def test_scattering_matches_two_hop_bandpass_definition():
    x = np.array([[0.], [2.], [5.]])
    edges = np.array([[0, 1, 2], [1, 2, 0]])
    embedding, scales = scattering_features(x, edges, np.arange(3))
    blocks = [np.array([[2.], [5.], [0.]]), np.array([[3.], [5.], [2.]]), np.array([[3.], [5.], [2.]])]
    expected = np.concatenate([block / scale for block, scale in zip(blocks, scales)], axis=1) / np.sqrt(3)
    np.testing.assert_allclose(embedding, expected)


def test_candidate_cache_reuses_completed_embeddings(tmp_path, monkeypatch):
    import src.gnn_distance_candidates as module
    x, edges = tiny_graph()
    settings = dict(depth=2, root_weight=.5, self_loops=False, rff_width=16, rff_seed=2026)
    options = dict(architectures=('gcn',), rf_width=8, rf_seeds=(2000,), ntk_width=4,
                   ntk_seeds=(3000,), ntk_projections=2, device='cpu')
    first, _, _ = build_candidate_distances(x.numpy(), edges.numpy(), np.arange(4), tmp_path, settings, **options)

    def forbidden(*args, **kwargs):
        raise RuntimeError('Completed embeddings must be reused')

    monkeypatch.setattr(module, 'random_gnn_features', forbidden)
    monkeypatch.setattr(module, 'empirical_ntk_features', forbidden)
    second, _, _ = build_candidate_distances(x.numpy(), edges.numpy(), np.arange(4), tmp_path, settings, **options)
    assert set(first) == {'rf_gcn', 'entk_gcn', 'sum_mmd', 'scattering'}
    for method in first:
        np.testing.assert_array_equal(first[method], second[method])
        np.testing.assert_allclose(np.diag(first[method]), 0.)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Colab GPU interface check')
@pytest.mark.parametrize('architecture', ['gcn', 'sage', 'gin'])
def test_cuda_candidate_interfaces(architecture):
    x, edges = tiny_graph()
    x, edges = x.cuda(), edges.cuda()
    ids = torch.arange(4, device=x.device)
    rf = random_gnn_features(x, edges, ids, architecture, 4, 2000)
    ntk = empirical_ntk_features(x, edges, ids, architecture, 4, 3000, 2, 4000)
    assert rf.shape == (4, 4) and ntk.shape == (4, 2)
    assert np.isfinite(rf).all() and np.isfinite(ntk).all()
