import json
from functools import lru_cache
from itertools import permutations

import numpy as np
import pytest

from src.tree_distance import exact_tree_distances


def graph():
    x = np.array([[1., 0.], [0., 2.], [3., 1.], [2., 2.]])
    edges = np.array([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]])
    return x, edges


def test_exact_distances_match_recursive_exhaustive_matching(tmp_path):
    x, edges = graph()
    neighbors = [edges[1, edges[0] == i].tolist() for i in range(len(x))]
    weight = .7

    @lru_cache(None)
    def reference(i, j, depth):
        left_x = np.zeros(x.shape[1]) if i is None else x[i]
        right_x = np.zeros(x.shape[1]) if j is None else x[j]
        root = np.linalg.norm(left_x - right_x)
        if depth == 0:
            return root
        left = [] if i is None else neighbors[i]
        right = [] if j is None else neighbors[j]
        size = max(len(left), len(right))
        if not size:
            return root
        left = left + [None] * (size - len(left))
        right = right + [None] * (size - len(right))
        cost = min(sum(reference(a, b, depth - 1) for a, b in zip(left, order))
                   for order in permutations(right))
        return root + weight * cost

    records = exact_tree_distances(x, edges, tmp_path, max_depth=2, weight=weight,
                                   checkpoint_rows=1, benchmark_pairs=0)
    for record in records:
        depth = record['depth']
        expected = np.array([[reference(i, j, depth) for j in range(len(x))] for i in range(len(x))])
        np.testing.assert_allclose(np.load(record['path']), expected, atol=1e-12)
        np.testing.assert_allclose(np.load(record['blank_path']),
                                   [reference(i, None, depth) for i in range(len(x))], atol=1e-12)


def test_permutation_duplicate_edges_and_self_loops(tmp_path):
    x, edges = graph()
    records = exact_tree_distances(x, edges, tmp_path / 'first', max_depth=1, benchmark_pairs=0)
    order = np.array([2, 0, 3, 1])
    inverse = order.argsort()
    modified = np.concatenate((inverse[edges], inverse[edges], np.tile(np.arange(4), (2, 1))), axis=1)
    shuffled = exact_tree_distances(x[order], modified, tmp_path / 'second', max_depth=1, benchmark_pairs=0)
    for a, b in zip(records, shuffled):
        np.testing.assert_allclose(np.load(b['path']), np.load(a['path'])[np.ix_(order, order)])


def test_resume_skips_flushed_rows_and_can_extend_depth(tmp_path, monkeypatch):
    import src.tree_distance as module
    x, edges = graph()
    original = module._assignment_cost
    calls = []

    def interrupted(*args):
        calls.append(args[-2:])
        if len(calls) == 5:
            raise RuntimeError('interruption')
        return original(*args)

    monkeypatch.setattr(module, '_assignment_cost', interrupted)
    with pytest.raises(RuntimeError, match='interruption'):
        exact_tree_distances(x, edges, tmp_path, max_depth=1, checkpoint_rows=1, benchmark_pairs=0)
    assert json.loads((tmp_path / 'depth_1.json').read_text())['next_row'] == 1
    calls.clear()
    exact_tree_distances(x, edges, tmp_path, max_depth=1, checkpoint_rows=1, benchmark_pairs=0)
    assert calls == [(1, 2), (1, 3), (2, 3)]
    monkeypatch.setattr(module, '_assignment_cost', original)
    resumed = exact_tree_distances(x, edges, tmp_path, max_depth=2, benchmark_pairs=0)
    fresh = exact_tree_distances(x, edges, tmp_path / 'fresh', max_depth=2, benchmark_pairs=0)
    np.testing.assert_array_equal(np.load(resumed[-1]['path']), np.load(fresh[-1]['path']))


def test_cache_rejects_different_graph_or_metric(tmp_path):
    x, edges = graph()
    exact_tree_distances(x, edges, tmp_path, max_depth=0, benchmark_pairs=0)
    with pytest.raises(ValueError, match='another graph or metric'):
        exact_tree_distances(x + 1, edges, tmp_path, max_depth=0, benchmark_pairs=0)
    with pytest.raises(ValueError, match='another graph or metric'):
        exact_tree_distances(x, edges, tmp_path, max_depth=0, weight=.5, benchmark_pairs=0)


def test_self_loop_option_and_single_node(tmp_path):
    x = np.array([[3., 4.]])
    edges = np.empty((2, 0), dtype=np.int64)
    records = exact_tree_distances(x, edges, tmp_path, max_depth=2, self_loops=True, benchmark_pairs=10)
    np.testing.assert_array_equal(np.load(records[-1]['path']), [[0.]])
    np.testing.assert_allclose(np.load(records[-1]['blank_path']), [15.])
