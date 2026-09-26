import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from src.tree_distance import _neighbors, exact_tree_distances
from src.tree_distance_analysis import evaluate_distance_blocks, load_distance_blocks


def normalize_rows(x):
    x = np.ascontiguousarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def mean_depth_one(root, total, query_degree, candidate_degree):
    mass = np.maximum(query_degree[:, None], candidate_degree[None, :]).clip(min=1)
    return root + np.maximum(total - root, 0) / mass


def factorial_effects(sweep):
    pivot = sweep.pivot(index=['k', 'voting'], columns='method', values='val_accuracy')
    effects = pivot[['raw_sum', 'raw_mean', 'l2_sum', 'l2_mean']].copy()
    effects['normalization_effect_sum'] = pivot['l2_sum'] - pivot['raw_sum']
    effects['normalization_effect_mean'] = pivot['l2_mean'] - pivot['raw_mean']
    effects['mean_effect_raw'] = pivot['raw_mean'] - pivot['raw_sum']
    effects['mean_effect_l2'] = pivot['l2_mean'] - pivot['l2_sum']
    effects['interaction'] = effects['mean_effect_l2'] - effects['mean_effect_raw']
    return effects.reset_index().rename_axis(columns=None)


def run_tree_distance_factorial(x, edge_index, y, train_mask, val_mask, propagated_features,
                                raw_distance_dir, normalized_distance_dir, output_dir,
                                ks=(1, 3, 5, 7, 11, 15, 21, 31), voting=('uniform', 'distance'),
                                checkpoint_rows=64, benchmark_pairs=5000, seed=0, on_benchmark=None):
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.asarray(y)
    train_mask, val_mask = np.asarray(train_mask, dtype=bool), np.asarray(val_mask, dtype=bool)
    if train_mask.shape != (len(x),) or val_mask.shape != train_mask.shape or (train_mask & val_mask).any():
        raise ValueError('Require disjoint train/validation masks matching the graph')
    train_ids, val_ids = np.flatnonzero(train_mask), np.flatnonzero(val_mask)
    if not len(train_ids) or not len(val_ids):
        raise ValueError('Require nonempty train and validation splits')
    protocol, raw = load_distance_blocks(x, edge_index, raw_distance_dir, train_ids, val_ids, [1])
    normalized = normalize_rows(x)
    records = exact_tree_distances(
        normalized, edge_index, normalized_distance_dir, max_depth=1,
        weight=protocol['weight'], dtype=protocol['dtype'], self_loops=protocol['self_loops'],
        checkpoint_rows=checkpoint_rows, benchmark_pairs=benchmark_pairs, seed=seed, on_benchmark=on_benchmark)
    normalized_protocol, l2 = load_distance_blocks(
        normalized, edge_index, normalized_distance_dir, train_ids, val_ids, [1])
    metric_neighbors, _ = _neighbors(edge_index, len(x), protocol['self_loops'])
    metric_degree = np.array([len(v) for v in metric_neighbors])
    neighbors, _ = _neighbors(edge_index, len(x), False)
    degree = np.array([len(v) for v in neighbors])
    h = np.asarray(propagated_features)
    if h.ndim != 2 or len(h) != len(x) or not np.isfinite(h).all():
        raise ValueError('Invalid propagated features')
    blocks = dict(
        raw_euclidean=raw['raw'], l2_euclidean=l2['raw'],
        raw_sum=raw['tree_1'],
        raw_mean=mean_depth_one(raw['raw'], raw['tree_1'], metric_degree[val_ids], metric_degree[train_ids]),
        l2_sum=l2['tree_1'],
        l2_mean=mean_depth_one(l2['raw'], l2['tree_1'], metric_degree[val_ids], metric_degree[train_ids]),
        grip_S2X=cdist(h[val_ids], h[train_ids]),
        degree_only=np.abs(np.log1p(degree[val_ids, None]) - np.log1p(degree[None, train_ids])))
    report = evaluate_distance_blocks(blocks, y[train_ids], y[val_ids], degree[train_ids], degree[val_ids], ks, voting)
    report['effects'] = factorial_effects(report['sweep'])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in report.items():
        table.to_csv(output / f'{name}.csv', index=False)
    np.savez_compressed(output / 'validation_blocks.npz', train_ids=train_ids, val_ids=val_ids, **blocks)
    digest = hashlib.sha256()
    for values in (train_ids, val_ids, y[train_ids], y[val_ids]):
        digest.update(np.ascontiguousarray(values, dtype=np.int64).tobytes())
    config = dict(version=1, depth=1, raw_protocol=protocol, normalized_protocol=normalized_protocol,
                  feature_normalization='nodewise_l2_zero_rows_unchanged',
                  mean_definition='D0 + (D1_sum - D0) / max(1, degree_i, degree_j)',
                  normalized_distance_timing=records, split_sha256=digest.hexdigest(),
                  ks=list(ks), voting=list(voting), test_labels_used=False,
                  baseline='grip_S2X_from_original_features_without_additional_normalization')
    (output / 'protocol.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    return report
