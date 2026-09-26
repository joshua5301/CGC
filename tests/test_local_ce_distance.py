import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from src.local_ce_distance import (ce_change_matrix, evaluate_local_pairs,
                                    fit_l1_scale, local_pair_masks, matched_mean_distance,
                                    run_local_ce_comparison, plot_local_ce_comparison)
from src.node_distances import array_digest
from src.tree_distance import _neighbors


def test_ce_matrix_keeps_source_target_and_is_directed():
    logits = np.array([[3., 0.], [0., 1.], [1., 2.]])
    targets = np.array([[.9, .1], [.5, .5], [.2, .8]])
    result = ce_change_matrix(logits, targets)
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    for i in range(3):
        for j in range(3):
            np.testing.assert_allclose(result[i, j], abs(targets[i] @ (logp[i] - logp[j])), atol=1e-12)
    assert not np.allclose(result, result.T)
    np.testing.assert_allclose(np.diag(result), 0.)
    np.testing.assert_allclose(result, ce_change_matrix(logits + np.array([10., -4., 7.])[:, None], targets), atol=1e-12)


def test_l1_fit_uses_distance_weighted_median_and_ignores_zero_distance():
    distance = np.array([0., 1., 1., 10.])
    change = np.array([100., 1., 2., 30.])
    np.testing.assert_allclose(fit_l1_scale(distance, change), 3.)
    np.testing.assert_allclose(fit_l1_scale(100 * distance, change), .03)
    np.testing.assert_allclose(fit_l1_scale(np.zeros(3), np.ones(3)), 0.)


def test_threshold_ties_are_included_and_knn_has_fixed_coverage():
    distance = np.ones((5, 5)) - np.eye(5)
    masks = local_pair_masks(distance, fractions=(.01,), ks=(2,))
    assert masks[0][3].sum() == 20
    np.testing.assert_array_equal(masks[1][3].sum(1), 2)
    assert not np.diag(masks[1][3]).any()
    report = evaluate_local_pairs(distance, 2 * distance, 2., fractions=(.01,), ks=(2,))
    assert report[0]['pair_fraction'] == 1.
    assert report[1]['pair_fraction'] == .5
    assert all(row['fit_mae'] == 0 for row in report)


def test_small_fit_error_does_not_imply_small_ce_change():
    distance = np.array([[0., 1., 3.], [1., 0., 2.], [3., 2., 0.]])
    result = evaluate_local_pairs(distance, 100 * distance, 100., fractions=(.2,), ks=(1,))
    assert result[0]['fit_mae'] == 0
    assert result[0]['ce_mean'] == 100
    scaled = evaluate_local_pairs(20 * distance, 100 * distance, 5., fractions=(.2,), ks=(1,))
    for first, second in zip(result, scaled):
        for key in ('ce_mean', 'ce_p95', 'relative_ce_mean', 'relative_fit_mae', 'pair_fraction'):
            np.testing.assert_allclose(first[key], second[key])


def test_collapsed_outputs_have_undefined_relative_scores():
    distance = np.ones((3, 3)) - np.eye(3)
    result = evaluate_local_pairs(distance, np.zeros((3, 3)), 0., fractions=(.1,), ks=(1,))
    assert all(row['degenerate'] and np.isnan(row['relative_ce_mean']) for row in result)


def test_matched_mean_uses_incoming_edges_and_absorbing_isolates():
    x = np.array([[0.], [2.], [4.]])
    edges = np.array([[0, 2], [1, 1]])
    settings = dict(self_loops=False, root_weight=.5, depth=2)
    result = matched_mean_distance(x, edges, np.arange(3), settings)
    np.testing.assert_allclose(result, [[0., 2., 4.], [2., 0., 2.], [4., 2., 0.]])


def test_saved_pipeline_uses_disjoint_nodes_and_scale_invariant_scores(tmp_path):
    x = np.column_stack((np.arange(8.), np.arange(8.) ** 2))
    edges = np.array([np.arange(8), np.roll(np.arange(8), 1)])
    _, canonical = _neighbors(edges, 8, False)
    graph_hash = hashlib.sha256(x.tobytes() + canonical.tobytes()).hexdigest()
    run, cache, result = [tmp_path / name for name in ('run', 'cache', 'result')]
    for path in (run, cache, result):
        path.mkdir()
    np.save(run / 'probe_ids.npy', np.arange(8))
    targets = np.tile([.6, .4], (8, 1))
    np.savez(run / 'teacher_predictions.npz', probabilities=targets)
    logits, train_ids = x / 10, np.array([0, 3])
    filename = 'gcn_n2_subset0_seed100.npz'
    np.savez(run / filename, trained=logits, train_ids=train_ids)
    distance = np.abs(np.arange(8.)[:, None] - np.arange(8.))
    np.savez(cache / 'distances.npz', grip_S2X=distance, probability_ot=100 * distance)
    config = dict(source_dir=str(run), distance_cache=str(cache), models=['gcn'],
                  distance_settings=dict(root_weight=.5, depth=2, self_loops=False),
                  source_protocol=dict(distance_protocol=dict(self_loops=False, input_sha256=graph_hash),
                                       models=['gcn', 'gin'], train_sizes=[2], subset_seeds=[0], model_seeds=[100]),
                  teacher_sha256=array_digest(targets),
                  source_logits_sha256={filename: array_digest(logits, train_ids)})
    (result / 'protocol.json').write_text(json.dumps(config))
    np.savez(run / 'gin_n2_subset0_seed100.npz', trained=2 * logits, train_ids=train_ids)
    report = run_local_ce_comparison(result, x, edges, fractions=(.1,), ks=(1,))
    protocol = json.loads((Path(report['folder']) / 'protocol.json').read_text())
    assert not set(protocol['calibration_nodes']) & set(protocol['evaluation_nodes'])
    assert len(protocol['calibration_nodes']) == 2
    assert len(protocol['evaluation_nodes']) == 6
    assert report['models'] == ['gcn', 'gin']
    assert 'gin_n2_subset0_seed100.npz' in protocol['source_logits_sha256']
    assert set(report['health'].model) == {'gcn', 'gin'}
    scores = report['per_run']
    left = scores[scores.method == 'grip_S2X']
    right = scores[scores.method == 'probability_ot']
    for key in ('ce_mean', 'ce_p95', 'relative_fit_mae'):
        np.testing.assert_allclose(left[key], right[key], atol=1e-12)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for figure in plot_local_ce_comparison(report):
        plt.close(figure)
    from src.gnn_distance_candidates import plot_candidate_distances
    for figure in plot_candidate_distances(report):
        plt.close(figure)
    from src.gcn_kernel_features import run_gcn_kernel_study
    analytic = run_gcn_kernel_study(result, x, edges, report['folder'], models=['gcn', 'gin'], device='cpu')
    assert {'gcn2_ntk', 'gcn2_nngp'} <= set(analytic['methods'])
    old = report['summary'].sort_values(['model', 'method', 'selection', 'cutoff'])
    same = analytic['summary'][analytic['summary'].method.isin(report['methods'])].sort_values(
        ['model', 'method', 'selection', 'cutoff'])
    np.testing.assert_allclose(old.ce_mean_mean, same.ce_mean_mean, atol=1e-12)
    from src.empirical_ntk_study import run_empirical_ntk_study, plot_empirical_ntk_study
    options = dict(architectures=('gcn',), widths=(4,), network_seeds=(5000, 5001),
                   ensemble_sizes=(1, 2), projections=(2, 4), sketch_seeds=(6000, 7000),
                   outputs=2, audit_nodes=2, device='cpu')
    deep = run_empirical_ntk_study(report['folder'], x, edges, **options)
    assert len(deep['metadata']) == 10
    assert len(deep['exact_audit']) == 4
    assert set(deep['ensemble_agreement'].family) == {'rf', 'ntk'}
    original = deep['summary'][deep['summary'].method.isin(report['methods'])].sort_values(
        ['model', 'method', 'selection', 'cutoff'])
    np.testing.assert_allclose(old.ce_mean_mean, original.ce_mean_mean, atol=1e-12)
    import src.empirical_ntk_study as study_module
    from unittest.mock import patch
    with patch.object(study_module, 'make_network', side_effect=RuntimeError('Must reuse cache')):
        repeated = run_empirical_ntk_study(report['folder'], x, edges, **options)
    np.testing.assert_allclose(deep['summary'].ce_mean_mean, repeated['summary'].ce_mean_mean)
    for figure in plot_empirical_ntk_study(deep, k=1):
        plt.close(figure)
