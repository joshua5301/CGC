import json
from itertools import permutations

import numpy as np
import pandas as pd

from src.tree_distance import exact_tree_distances
from src.tree_distance_factorial import factorial_effects, mean_depth_one, normalize_rows, run_tree_distance_factorial


def test_normalization_preserves_zero_rows_and_removes_positive_row_scale():
    x = np.array([[3., 4.], [0., 0.], [-1., 2.]])
    result = normalize_rows(x)
    np.testing.assert_allclose(result, normalize_rows(x * np.array([[2.], [3.], [.5]])))
    np.testing.assert_allclose(result[0], [.6, .8])
    np.testing.assert_array_equal(result[1], [0., 0.])


def test_only_transport_is_averaged_and_isolates_are_finite():
    root = np.array([[0., 2., 3.], [1., 2., 3.]])
    total = root + np.array([[0., 4., 6.], [2., 4., 9.]])
    result = mean_depth_one(root, total, np.array([0, 1]), np.array([0, 2, 3]))
    np.testing.assert_allclose(result, [[0., 4., 5.], [3., 4., 6.]])


def test_effects_compare_the_same_knn_setting():
    rows = []
    for k in (1, 3):
        for name, accuracy in dict(raw_sum=40, raw_mean=45, l2_sum=60, l2_mean=70).items():
            rows.append(dict(method=name, k=k, voting='uniform', val_accuracy=accuracy + k))
    effects = factorial_effects(pd.DataFrame(rows))
    np.testing.assert_allclose(effects['normalization_effect_sum'], 20)
    np.testing.assert_allclose(effects['mean_effect_raw'], 5)
    np.testing.assert_allclose(effects['mean_effect_l2'], 10)
    np.testing.assert_allclose(effects['interaction'], 5)


def test_factorial_matches_exhaustive_mean_matching_and_reuses_raw_cache(tmp_path):
    x = np.array([[1., 0.], [0., 2.], [3., 1.], [2., 2.], [0., 0.]])
    edges = np.array([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]])
    raw = tmp_path / 'raw'
    exact_tree_distances(x, edges, raw, max_depth=2, weight=.7, benchmark_pairs=0)
    original_summary = (raw / 'summary.json').read_text()
    train = np.array([True, True, False, False, False])
    val = np.array([False, False, True, True, False])
    args = dict(x=x, edge_index=edges, train_mask=train, val_mask=val,
                propagated_features=x, raw_distance_dir=raw,
                normalized_distance_dir=tmp_path / 'normalized', ks=[1, 2], benchmark_pairs=0)
    report = run_tree_distance_factorial(y=np.array([0, 1, 0, 1, 0]), output_dir=tmp_path / 'first', **args)
    second = run_tree_distance_factorial(y=np.array([0, 1, 0, 1, 999]), output_dir=tmp_path / 'second', **args)
    for name in report:
        pd.testing.assert_frame_equal(report[name], second[name])
    assert len(report['best']) == 8
    assert (raw / 'summary.json').read_text() == original_summary
    blocks = np.load(tmp_path / 'first' / 'validation_blocks.npz')
    for name, features in [('raw_mean', x), ('l2_mean', normalize_rows(x))]:
        for row, i in enumerate([2, 3]):
            for col, j in enumerate([0, 1]):
                left, right = edges[1, edges[0] == i].tolist(), edges[1, edges[0] == j].tolist()
                size = max(len(left), len(right))
                left += [None] * (size - len(left))
                right += [None] * (size - len(right))

                def feature(node):
                    return np.zeros(features.shape[1]) if node is None else features[node]

                cost = min(sum(np.linalg.norm(feature(a) - feature(b)) for a, b in zip(left, order))
                           for order in permutations(right)) if size else 0.
                expected = np.linalg.norm(features[i] - features[j]) + .7 * cost / max(1, size)
                np.testing.assert_allclose(blocks[name][row, col], expected)
    config = json.loads((tmp_path / 'first' / 'protocol.json').read_text())
    assert config['test_labels_used'] is False and config['depth'] == 1
