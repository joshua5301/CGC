import numpy as np
import pandas as pd
import pytest
import torch

from src.gnn_distance_probe import distance_geometry, fit_probe, output_stability


def test_distance_normalization_is_invariant_to_global_scale():
    values = np.array([[0., 1., 2., 3.], [1., 0., 4., 5.], [2., 4., 0., 6.], [3., 5., 6., 0.]])
    geometry, scales = distance_geometry({'a': values, 'b': 1000 * values})
    np.testing.assert_allclose(geometry['a']['distance'], geometry['b']['distance'])
    logits = np.array([[1., 0.], [2., 0.], [0., 2.], [0., 3.]])
    report = output_stability(logits, geometry, ks=[1, 2])
    for name in ('local', 'tails'):
        table = report[name]
        a = table[table.method == 'a'].drop(columns='method').reset_index(drop=True)
        b = table[table.method == 'b'].drop(columns='method').reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
    assert scales.iloc[1].distance_scale == 1000 * scales.iloc[0].distance_scale


def test_output_metrics_ignore_per_node_common_logit_offsets():
    points = np.arange(4.)
    geometry, _ = distance_geometry({'d': np.abs(points[:, None] - points[None, :])})
    logits = np.array([[1., 0.], [2., 1.], [0., 2.], [0., 3.]])
    first = output_stability(logits, geometry, ks=[1])
    second = output_stability(logits + np.array([[5.], [-7.], [2.], [9.]]), geometry, ks=[1])
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_zero_distance_violations_and_constant_models_are_not_hidden():
    matrix = np.array([[0., 0., 1.], [0., 0., 1.], [1., 1., 0.]])
    geometry, _ = distance_geometry({'d': matrix})
    result = output_stability(np.array([[2., 0.], [0., 2.], [1., 1.]]), geometry, ks=[1])
    assert (result['tails'].zero_distance_violations == 1).all()
    assert np.isinf(result['tails'].ratio_max).all()
    constant = output_stability(np.ones((3, 2)), geometry, ks=[1])
    assert constant['health'].degenerate.all()
    assert constant['local'].relative_gap.isna().all()


def test_close_pairs_measure_output_similarity_not_labels():
    matrix = np.array([[0., 1., 10., 11.], [1., 0., 11., 10.], [10., 11., 0., 1.], [11., 10., 1., 0.]])
    geometry, _ = distance_geometry({'d': matrix})
    logits = np.array([[2., 0.], [2., 0.], [0., 2.], [0., 2.]])
    result = output_stability(logits, geometry, ks=[1])
    np.testing.assert_allclose(result['local'].gap_mean, 0.)
    np.testing.assert_allclose(result['local'].relative_gap, 0.)


@pytest.mark.parametrize('model', ['gcn', 'sage', 'gin'])
@pytest.mark.parametrize('soft', [False, True])
def test_models_train_with_only_selected_labels(model, soft):
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [.2, .4]])
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    labels = np.array([[.8, .2], [.1, .9]]) if soft else np.array([0, 1])
    result = fit_probe(x, edges, np.array([0, 1]), labels, np.array([2, 3]),
                       model, 2, 7, hidden=4, dropout=0., epochs=2)
    assert result['initial'].shape == result['trained'].shape == (2, 2)
    assert np.isfinite(result['trained']).all()
    assert np.isfinite(result['train_ce'])
