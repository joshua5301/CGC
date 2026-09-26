import math

import torch
import torch.nn.functional as F
from torch_geometric import seed_everything

from src.fsw import FSWEncoder, aggregate, neighborhoods, normalized_adjacency, realize_graph
from src.models import GCN
from src.risk_experiment import _forward, _train_student


def test_quantiles_distinguish_equal_means():
    x = torch.tensor([[0.], [-2.], [-1.], [1.], [2.], [-2.], [0.], [0.], [2.]])
    first = neighborhoods(torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]]), len(x))
    second = neighborhoods(torch.tensor([[0, 0, 0, 0], [5, 6, 7, 8]]), len(x))
    directions, frequencies = torch.ones(1, 2), torch.tensor([0., .5])
    a = aggregate(x, first, directions, frequencies)[0]
    b = aggregate(x, second, directions, frequencies)[0]
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[1], torch.tensor(-3 * (2 + math.sqrt(2)) / math.pi))
    torch.testing.assert_close(b[1], torch.tensor(-6 * math.sqrt(2) / math.pi))


def test_encoder_is_equivariant_and_has_feature_gradients():
    x = torch.tensor([[1., 0.], [0., 2.], [3., 1.], [-1., 2.]])
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    encoder = FSWEncoder(2, depth=2, width=8)
    expected = encoder.fit_transform(x, neighborhoods(edges, 4))
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = permutation.argsort()
    layout = neighborhoods(inverse[edges], 4)
    values = x[permutation].clone().requires_grad_()
    actual = encoder(values, layout)
    torch.testing.assert_close(actual, expected[permutation])
    actual.square().sum().backward()
    assert torch.isfinite(values.grad).all() and values.grad.norm() > 0


def test_cardinality_and_isolated_nodes_are_retained():
    x = torch.tensor([[1.], [1.], [1.], [1.]])
    edges = torch.tensor([[0, 2, 2], [1, 1, 3]])
    encoder = FSWEncoder(1, depth=1, width=8)
    result = encoder.fit_transform(x, neighborhoods(edges, 4))
    torch.testing.assert_close(result[0, 1:-1], result[2, 1:-1])
    assert result[0, -1] != result[2, -1]
    assert torch.isfinite(result).all()


def test_identity_partition_reconstructs_original_graph():
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    encoder = FSWEncoder(2, depth=2, width=8)
    embedding = encoder.fit_transform(x, neighborhoods(edges, 3))
    result = realize_graph(x, edges, torch.arange(3), embedding, encoder, neighbors=3, steps=2)
    torch.testing.assert_close(result['x'], x)
    torch.testing.assert_close(result['adjacency'], normalized_adjacency(edges, 3))
    assert result['realization_final'] < 1e-6


def test_realization_keeps_best_and_respects_triangle_bound():
    x = torch.tensor([[1., 0.], [2., 1.], [0., 2.], [-1., 3.]])
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    encoder = FSWEncoder(2, depth=1, width=8)
    embedding = encoder.fit_transform(x, neighborhoods(edges, 4))
    assignment = torch.tensor([0, 0, 0, 1])
    centers = torch.stack((embedding[:3].mean(0), embedding[3]))
    result = realize_graph(x, edges, assignment, centers, encoder, steps=3)
    direct = (embedding - result['realized_embedding'][assignment]).norm(dim=1).mean()
    fit = (embedding - centers[assignment]).norm(dim=1).mean()
    assert result['realization_final'] <= result['realization_initial']
    assert direct <= fit + result['realization_final'] + 1e-5


def test_student_uses_graph_and_mass_weighted_ce():
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    q = torch.tensor([[.8, .2], [.1, .9], [.4, .6]])
    counts = torch.tensor([1, 2, 5])
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    adj = normalized_adjacency(edges, 3)
    graph = dict(x=x, y=q.argmax(1), adj=adj)
    params = dict(dropout=0., lr=.01, weight_decay=0.)
    settings = dict(hidden=4, epochs=1, eval_every=1, loss_weighting='mass')
    _, _, _, trained = _train_student(x, q, (graph, None), params, 7, settings,
                                      counts=counts, return_model=True, adjacency=adj)
    seed_everything(7)
    model = GCN(2, 4, 2, 2, 0.)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    h = F.relu(adj @ model.layers[0].lin(x) + model.layers[0].bias)
    logits = adj @ model.layers[1].lin(h) + model.layers[1].bias
    loss = (-(q * logits.log_softmax(1)).sum(1) * counts / counts.sum()).sum()
    loss.backward()
    optimizer.step()
    for expected, actual in zip(model.parameters(), trained.parameters()):
        torch.testing.assert_close(expected, actual)
    torch.testing.assert_close(_forward(trained, x, adj), _forward(trained, x, adj.to_sparse()))
