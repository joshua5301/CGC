import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.spatial.distance import cdist

from src.node_distances import (multiscale_distance, neighborhood_mmd_distance,
                                probability_ot_distances, uniform_transport)
from src.node_distance_comparison import (ce_preservation, compare_node_distances,
                                          plot_node_distance_comparison, select_medoids)
from src.tree_distance import _neighbors


def test_probability_ot_uses_fractional_mass_without_padding():
    np.testing.assert_allclose(uniform_transport(np.array([[0., 0.]])), 0.)
    np.testing.assert_allclose(uniform_transport(np.array([[0., 1., 3.], [3., 2., 0.]])), .5)
    np.testing.assert_allclose(uniform_transport(np.array([[4., 1.], [2., 5.]])), 1.5)


def test_query_closure_matches_full_recursion_and_preserves_query_order(tmp_path):
    x = np.array([[0., 1.], [2., 1.], [1., 0.], [3., 4.], [0., 0.]])
    edges = np.array([[0, 1, 2, 2, 3, 4], [1, 0, 1, 3, 2, 3]])
    full, _ = probability_ot_distances(x, edges, np.arange(5), tmp_path / 'full', depth=2)
    ids = np.array([4, 1, 0])
    subset, _ = probability_ot_distances(x, edges, ids, tmp_path / 'subset', depth=2)
    np.testing.assert_allclose(subset, full[np.ix_(ids, ids)])
    np.testing.assert_allclose(full, full.T)
    np.testing.assert_allclose(np.diag(full), 0.)
    for k in range(5):
        assert (full <= full[:, k, None] + full[None, k, :] + 1e-10).all()


def test_isolated_nodes_are_absorbing_not_blank_mass(tmp_path):
    x = np.array([[0.], [3.]])
    result, _ = probability_ot_distances(x, np.empty((2, 0), dtype=int), np.arange(2), tmp_path, depth=2)
    np.testing.assert_allclose(result, [[0., 3.], [3., 0.]])


def test_probability_ot_resumes_and_rejects_mismatched_inputs(tmp_path, monkeypatch):
    import src.node_distances as module
    x = np.arange(4.)[:, None]
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    original, calls = module.uniform_transport, []

    def interrupted(cost):
        calls.append(1)
        if len(calls) == 4:
            raise RuntimeError('interrupted')
        return original(cost)

    monkeypatch.setattr(module, 'uniform_transport', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        probability_ot_distances(x, edges, np.arange(4), tmp_path, depth=1, checkpoint_rows=1)
    assert json.loads((tmp_path / 'depth_1.json').read_text())['next_row'] == 1
    calls.clear()
    resumed, _ = probability_ot_distances(x, edges, np.arange(4), tmp_path, depth=1, checkpoint_rows=1)
    assert len(calls) == 3
    monkeypatch.setattr(module, 'uniform_transport', original)
    fresh, _ = probability_ot_distances(x, edges, np.arange(4), tmp_path / 'fresh', depth=1)
    np.testing.assert_array_equal(resumed, fresh)
    with pytest.raises(ValueError, match='cache mismatch'):
        probability_ot_distances(x + 1, edges, np.arange(4), tmp_path, depth=1)


def test_multiscale_is_invariant_to_positive_block_scaling():
    x = np.array([[0., 1.], [2., 1.], [1., 0.], [3., 4.]])
    stages = [x, x ** 2, x[:, ::-1]]
    first, _ = multiscale_distance(stages, np.arange(4))
    second, _ = multiscale_distance([stages[0] * 7, stages[1] * .1, stages[2] * 4], np.arange(4))
    np.testing.assert_allclose(first, second)


def test_mmd_averages_features_after_the_nonlinearity():
    x = np.array([[0.], [0.], [-1.], [1.], [0.], [0.]])
    edges = np.array([[2, 3, 4, 5], [0, 0, 1, 1]])
    distance, _ = neighborhood_mmd_distance(x, edges, np.arange(6), depth=1, width=256)
    assert distance[0, 1] > .1
    duplicate = x.copy()
    duplicate[2:4] = 0.
    distance, _ = neighborhood_mmd_distance(duplicate, edges, np.arange(6), depth=1, width=256)
    np.testing.assert_allclose(distance[0, 1], 0., atol=1e-12)


def test_mmd_duplicate_neighbor_mass_and_global_scale_invariance():
    x = np.array([[0.], [0.], [4.], [4.]])
    edges = np.array([[2, 2, 3], [0, 1, 1]])
    first, _ = neighborhood_mmd_distance(x, edges, np.arange(4), depth=2, width=32)
    second, _ = neighborhood_mmd_distance(7 * x, edges, np.arange(4), depth=2, width=32)
    np.testing.assert_allclose(first, second, atol=1e-12)
    np.testing.assert_allclose(first[0, 1], 0., atol=1e-12)


def test_medoids_nonempty_at_zero_distance_and_scale_invariance():
    distance = cdist(np.array([[0.], [0.], [1.], [4.], [4.]]), np.array([[0.], [0.], [1.], [4.], [4.]]))
    first, _ = select_medoids(distance, 4, seeds=(0, 1))
    second, _ = select_medoids(distance * 1000, 4, seeds=(0, 1))
    np.testing.assert_array_equal(first['medoids'], second['medoids'])
    assert (np.bincount(first['assignment'], minlength=4) > 0).all()
    zero, _ = select_medoids(np.zeros((5, 5)), 4, seeds=(0,))
    assert len(np.unique(zero['medoids'])) == 4
    assert len(np.unique(zero['assignment'])) == 4


def test_ce_bounds_and_exact_mass_weighted_cluster_label_identity():
    logits = np.array([[2., -1., 0.], [-2., 1., 3.], [0., 3., 2.], [4., 0., -1.]])
    targets = np.array([[.7, .2, .1], [.1, .3, .6], [.2, .5, .3], [.4, .4, .2]])
    representatives = np.array([0, 0, 2, 2])
    result = ce_preservation(logits, targets, representatives)
    chain = [result[key] for key in ('risk_gap', 'E_CE', 'E_robust', 'range_bound', 'centered_l2_bound')]
    assert (np.diff(chain) >= -1e-12).all()
    labels = np.array([targets[:2].mean(0), targets[2:].mean(0)])
    logp = logits[[0, 2]] - logsumexp(logits[[0, 2]], axis=1, keepdims=True)
    np.testing.assert_allclose(result['representative_risk'], -(labels * logp).sum(1).mean())
    shifted = ce_preservation(logits + np.array([100., -40., 7., 11.])[:, None], targets, representatives)
    for key in result:
        np.testing.assert_allclose(result[key], shifted[key], atol=1e-12)
    identity = ce_preservation(logits, targets, np.arange(4))
    np.testing.assert_allclose(identity['E_CE'], 0.)
    mask = np.array([False, True, True, False])
    masked = ce_preservation(logits, targets, representatives, mask)
    full_logp = logits - logsumexp(logits, axis=1, keepdims=True)
    direct = -(targets[mask] * full_logp[representatives[mask]]).sum(1).mean()
    np.testing.assert_allclose(masked['representative_risk'], direct)


def test_saved_output_comparison_without_retraining(tmp_path):
    x = np.array([[0., 1.], [2., 1.], [1., 0.], [3., 4.], [0., 2.], [3., 0.]], dtype=np.float32)
    edges = np.array([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])
    _, canonical = _neighbors(edges, 6, False)
    digest = hashlib.sha256(x.astype(np.float64).tobytes() + canonical.tobytes()).hexdigest()
    run = tmp_path / 'source'
    run.mkdir()
    source = dict(target_kind='teacher_soft_labels', graph='full_original_graph', models=['gcn', 'sage'],
                  train_sizes=[2], subset_seeds=[0], model_seeds=[100],
                  distance_protocol=dict(input_sha256=digest, shape=list(x.shape), self_loops=False))
    (run / 'protocol.json').write_text(json.dumps(source))
    np.save(run / 'probe_ids.npy', np.arange(6))
    np.savez(run / 'teacher_predictions.npz', probabilities=np.tile([[.6, .4]], (6, 1)))
    for model in source['models']:
        np.savez(run / f'{model}_n2_subset0_seed100.npz', trained=x, train_ids=np.array([0, 3]))
    settings = dict(models=('gcn', 'sage'), budgets=(2,), depth=2, medoid_seeds=(0, 1), ks=(1,), rff_width=16)
    first = compare_node_distances(x, edges, [x, x * .7, x * .3], run, tmp_path / 'output', **settings)
    second = compare_node_distances(x, edges, [x, x * .7, x * .3], run, tmp_path / 'output', **settings)
    pd.testing.assert_frame_equal(first['per_run'], second['per_run'])
    assert len(first['per_run']) == 2 * 4 * 2 * 2
    np.testing.assert_allclose(first['paired'].query('method == "grip_S2X"').delta_E_CE, 0.)
    for budget in settings['budgets']:
        with np.load(f"{first['folder']}/own_representatives_grip_S2X_budget{budget}.npz") as saved:
            np.testing.assert_allclose(saved['weights'].sum(), 1.)
            np.testing.assert_allclose(saved['labels'].sum(1), 1.)
    with pytest.raises(ValueError, match='Graph/features differ'):
        compare_node_distances(x + 1, edges, [x + 1] * 3, run, tmp_path / 'output', **settings)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for figure in plot_node_distance_comparison(first):
        plt.close(figure)
