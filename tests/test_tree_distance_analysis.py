import numpy as np
import pandas as pd
import pytest

from src.tree_distance import exact_tree_distances
from src.tree_distance_analysis import analyze_tree_distances, degree_matched_auc, evaluate_distance_blocks


def test_voting_modes_and_class_recall():
    result = evaluate_distance_blocks(
        {'example': np.array([[1., 2., 3.]])}, np.array([1, 0, 0]), np.array([1]),
        np.ones(3), np.ones(1), ks=[3])
    sweep = result['sweep'].set_index('voting')
    assert sweep.loc['uniform', 'val_accuracy'] == 0
    assert sweep.loc['distance', 'val_accuracy'] == 100
    selected = result['best_class'].set_index('label')
    assert selected.loc[1, 'recall'] == 100
    assert selected.loc[1, 'val_support'] == 1
    assert np.isnan(selected.loc[0, 'recall'])


def test_class_ties_use_nearest_member_and_zero_distances_dominate():
    tied = evaluate_distance_blocks({'d': np.array([[2., 1.]])}, np.array([0, 1]), np.array([1]),
                                    np.ones(2), np.ones(1), ks=[2], voting=['uniform'])
    assert tied['best'].iloc[0]['val_accuracy'] == 100
    zero = evaluate_distance_blocks({'d': np.array([[0., .01, .01]])}, np.array([1, 0, 0]), np.array([1]),
                                    np.ones(3), np.ones(1), ks=[3], voting=['distance'])
    assert zero['best'].iloc[0]['val_accuracy'] == 100


def test_exact_degree_matching_removes_degree_only_ranking():
    train_labels, val_labels = np.array([0, 1, 0, 1]), np.array([0])
    degree = np.array([1, 1, 3, 3])
    control = degree_matched_auc(np.array([[2., 2., 0., 0.]]), train_labels, val_labels, degree)
    signal = degree_matched_auc(np.array([[0., 1., 2., 3.]]), train_labels, val_labels, degree)
    assert control['degree_matched_auc'] == .5
    assert signal['degree_matched_auc'] == 1
    assert signal['matched_queries'] == 1 and signal['matched_comparisons'] == 2


def test_rank_diagnostics_and_votes_are_invariant_to_positive_scale():
    block = np.array([[1., 2., 4., 5.], [5., 1., 3., 2.]])
    result = evaluate_distance_blocks({'a': block, 'b': 7 * block}, np.array([0, 1, 0, 1]),
                                      np.array([0, 1]), np.array([1, 1, 3, 3]), np.array([2, 3]), ks=[1, 3])
    for name in ('sweep', 'diagnostics'):
        table = result[name]
        a = table[table.method == 'a'].drop(columns='method').reset_index(drop=True)
        b = table[table.method == 'b'].drop(columns='method').reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


def test_unused_labels_do_not_affect_analysis_and_cache_is_verified(tmp_path):
    x = np.array([[1., 0.], [0., 1.], [2., 1.], [1., 3.]])
    edges = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    cache = tmp_path / 'distances'
    exact_tree_distances(x, edges, cache, max_depth=0, benchmark_pairs=0)
    train = np.array([True, True, False, False])
    val = np.array([False, False, True, False])
    args = dict(x=x, edge_index=edges, train_mask=train, val_mask=val,
                propagated_features=x, distance_dir=cache, depths=[], ks=[1, 2])
    first = analyze_tree_distances(y=np.array([0, 1, 0, 0]), output_dir=tmp_path / 'first', **args)
    second = analyze_tree_distances(y=np.array([0, 1, 0, 999]), output_dir=tmp_path / 'second', **args)
    for key in first:
        pd.testing.assert_frame_equal(first[key], second[key])
    args['x'] = x + 1
    with pytest.raises(ValueError, match='does not match'):
        analyze_tree_distances(y=np.array([0, 1, 0, 0]), output_dir=tmp_path / 'wrong', **args)
