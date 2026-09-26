import torch

from src.centroid_candidates import candidate_statistics, nearest_candidates
from src.convex_representatives import convex_features, cluster_softmax


def test_nearest_counts_overlap_and_stable_ties():
    features = torch.tensor([[0.], [1.], [2.], [3.]])
    centers = torch.tensor([[.5], [1.5]])
    nodes, groups = nearest_candidates(features, centers, [2, 3], block_size=1)
    assert nodes.tolist() == [0, 1, 1, 2, 0]
    assert groups.tolist() == [0, 0, 1, 1, 1]
    stats = candidate_statistics(nodes, groups, torch.tensor([0, 0, 1, 1]))
    assert stats['unique_candidates'] == 3
    assert stats['max_candidate_reuse'] == 2
    assert abs(stats['outside_cluster_fraction'] - .4) < 1e-6


def test_shared_nodes_have_independent_convex_coefficients():
    features = torch.tensor([[0.], [1.], [2.]])
    nodes, groups = nearest_candidates(features, torch.tensor([[.5], [1.5]]), [2, 2])
    scores = torch.tensor([0., 1., 2., 0.], requires_grad=True)
    weights = cluster_softmax(scores, groups, 2)
    centers = convex_features(features[nodes], groups, weights, 2)
    centers[0].sum().backward()
    assert bool((scores.grad[groups == 1] == 0).all())
    assert bool((scores.grad[groups == 0] != 0).any())
    torch.testing.assert_close(torch.zeros(2).index_add_(0, groups, weights), torch.ones(2))
