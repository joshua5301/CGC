import numpy as np
import torch

import src.partition as module
from src.grip_visualization import align_labels, compare_partitions


def test_label_permutation_is_not_reassignment():
    first = np.array([0, 0, 1, 1, 2, 2])
    second = np.array([7, 7, 4, 4, 9, 9])
    np.testing.assert_array_equal(align_labels(first, second)[0], first)
    metrics = compare_partitions(first, second)
    assert metrics['ARI'] == 1
    assert metrics['reassigned_percent'] == 0


def test_unmatched_clusters_get_distinct_colors():
    aligned, mapping = align_labels(np.array([0, 0, 1, 1]), np.array([3, 4, 5, 5]))
    assert len(set(mapping.values())) == 3
    assert len(np.unique(aligned)) == 3


def test_initial_snapshot_does_not_modify_grip(monkeypatch):
    assignment = torch.tensor([0, 1, 0, 1])
    monkeypatch.setattr(module, 'kmeans_init', lambda x, n, **kw: assignment.clone())
    x = torch.tensor([[0.], [.1], [2.], [2.1]])
    q = torch.tensor([[.9, .1], [.8, .2], [.2, .8], [.1, .9]])
    before = module.partition(x, q, 2, return_diagnostics=True)
    saved = module.partition(x, q, 2, return_diagnostics=True, return_initial_state=True)
    torch.testing.assert_close(saved['initial_assignment'], assignment)
    for key in ('x', 'y', 'assignment'):
        torch.testing.assert_close(saved[key], before[key])
