import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr

from src.tree_distance import _neighbors


def _correlation(a, b):
    return float(spearmanr(a, b).statistic) if np.ptp(a) and np.ptp(b) else float('nan')


def degree_matched_auc(distance, train_labels, val_labels, train_degree):
    groups = [np.flatnonzero(train_degree == degree) for degree in np.unique(train_degree)]
    scores, comparisons = [], 0
    for row, label in zip(distance, val_labels):
        wins, pairs = 0., 0
        for group in groups:
            positive = train_labels[group] == label
            count = int(positive.sum())
            total = count * (len(group) - count)
            if not total:
                continue
            ranks = rankdata(row[group], method='average')
            higher = ranks[positive].sum() - count * (count + 1) / 2
            wins += total - higher
            pairs += total
        if pairs:
            scores.append(wins / pairs)
            comparisons += pairs
    return dict(degree_matched_auc=float(np.mean(scores)) if scores else float('nan'),
                matched_queries=len(scores), matched_comparisons=comparisons)


def evaluate_distance_blocks(blocks, train_labels, val_labels, train_degree, val_degree,
                             ks=(1, 3, 5, 7, 11, 15, 21, 31), voting=('uniform', 'distance')):
    train_labels, val_labels = np.asarray(train_labels), np.asarray(val_labels)
    train_degree, val_degree = np.asarray(train_degree), np.asarray(val_degree)
    ks = sorted(set(ks))
    if not ks or any(int(k) != k or k < 1 or k > len(train_labels) for k in ks):
        raise ValueError('Each k must be an integer between 1 and the number of training nodes')
    if not voting or set(voting) - {'uniform', 'distance'} or not len(val_labels):
        raise ValueError('Require validation labels and uniform/distance voting')
    classes = np.unique(np.concatenate((train_labels, val_labels)))
    encoded = np.searchsorted(classes, train_labels)
    degree_gap = np.abs(np.log1p(val_degree[:, None]) - np.log1p(train_degree[None, :]))
    degree_sum = np.log1p(val_degree[:, None]) + np.log1p(train_degree[None, :])
    same_class = val_labels[:, None] == train_labels[None, :]
    sweeps, recalls, diagnostics = [], [], []
    for name, distance in blocks.items():
        distance = np.asarray(distance, dtype=np.float64)
        if distance.shape != same_class.shape or not np.isfinite(distance).all() or (distance < 0).any():
            raise ValueError(f'Invalid validation-to-training distance block: {name}')
        order = np.argsort(distance, axis=1, kind='stable')[:, :max(ks)]
        near_distance = np.take_along_axis(distance, order, axis=1)
        near_labels = encoded[order]
        diagnostics.append(dict(
            method=name,
            distance_degree_gap_spearman=_correlation(distance.ravel(), degree_gap.ravel()),
            distance_degree_sum_spearman=_correlation(distance.ravel(), degree_sum.ravel()),
            **degree_matched_auc(distance, train_labels, val_labels, train_degree)))
        for k in ks:
            labels, distances = near_labels[:, :k], near_distance[:, :k]
            purity = float((classes[labels] == val_labels[:, None]).mean())
            neighbor_gap = float(np.take_along_axis(degree_gap, order[:, :k], axis=1).mean())
            for mode in voting:
                weights = np.ones_like(distances)
                if mode == 'distance':
                    weights = np.divide(distances[:, :1], distances, out=np.zeros_like(distances), where=distances > 0)
                    zero = distances[:, 0] == 0
                    weights[zero] = distances[zero] == 0
                scores = (np.eye(len(classes))[labels] * weights[:, :, None]).sum(1)
                tied = np.take_along_axis(scores, labels, axis=1) == scores.max(1, keepdims=True)
                predicted = classes[labels[np.arange(len(labels)), tied.argmax(1)]]
                class_recalls = []
                for label in classes:
                    mask = val_labels == label
                    recall = float((predicted[mask] == label).mean()) if mask.any() else float('nan')
                    if mask.any():
                        class_recalls.append(recall)
                    recalls.append(dict(method=name, k=k, voting=mode, label=int(label),
                                        train_support=int((train_labels == label).sum()),
                                        val_support=int(mask.sum()), val_predicted=int((predicted == label).sum()),
                                        recall=100 * recall))
                sweeps.append(dict(method=name, k=k, voting=mode,
                                   val_accuracy=100 * float((predicted == val_labels).mean()),
                                   val_macro_recall=100 * float(np.mean(class_recalls)),
                                   neighbor_purity=100 * purity,
                                   random_reference_purity=100 * float(same_class.mean()),
                                   neighbor_log_degree_gap=neighbor_gap,
                                   random_reference_log_degree_gap=float(degree_gap.mean())))
    sweep, per_class, diagnostic = pd.DataFrame(sweeps), pd.DataFrame(recalls), pd.DataFrame(diagnostics)
    best = sweep.sort_values(['method', 'val_accuracy', 'k', 'voting'], ascending=[True, False, True, True],
                             kind='stable').drop_duplicates('method').reset_index(drop=True)
    best = best.merge(diagnostic, on='method', validate='one_to_one')
    best_class = per_class.merge(best[['method', 'k', 'voting']], on=['method', 'k', 'voting'], validate='many_to_one')
    return dict(sweep=sweep, best=best, per_class=per_class, best_class=best_class, diagnostics=diagnostic)


def analyze_tree_distances(x, edge_index, y, train_mask, val_mask, propagated_features,
                           distance_dir, output_dir, depths=(1, 2, 3),
                           ks=(1, 3, 5, 7, 11, 15, 21, 31), voting=('uniform', 'distance')):
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.asarray(y)
    train_mask, val_mask = np.asarray(train_mask, dtype=bool), np.asarray(val_mask, dtype=bool)
    if train_mask.shape != (len(x),) or val_mask.shape != train_mask.shape or (train_mask & val_mask).any():
        raise ValueError('Require disjoint train/validation masks matching the graph')
    train_ids, val_ids = np.flatnonzero(train_mask), np.flatnonzero(val_mask)
    if not len(train_ids) or not len(val_ids):
        raise ValueError('Require nonempty train and validation splits')
    directory = Path(distance_dir)
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    _, edges = _neighbors(edge_index, len(x), protocol['self_loops'])
    digest = hashlib.sha256()
    digest.update(x.tobytes())
    digest.update(edges.tobytes())
    if protocol['version'] != 1 or protocol['input_sha256'] != digest.hexdigest() or protocol['shape'] != list(x.shape):
        raise ValueError('Distance cache does not match the input graph and features')
    neighbors, _ = _neighbors(edge_index, len(x), False)
    degree = np.array([len(v) for v in neighbors])
    blocks = {}
    for depth in sorted(set((0, *depths))):
        state = json.loads((directory / f'depth_{depth}.json').read_text(encoding='utf-8'))
        if state['next_row'] != len(x):
            raise ValueError(f'Distance depth {depth} is incomplete')
        matrix = np.load(directory / f'distance_{depth}.npy', mmap_mode='r')
        if matrix.shape != (len(x), len(x)) or matrix.dtype != np.dtype(protocol['dtype']):
            raise ValueError('Distance matrix shape or dtype mismatch')
        block = np.asarray(matrix[np.ix_(val_ids, train_ids)])
        if not np.allclose(block, matrix[np.ix_(train_ids, val_ids)].T) or not np.allclose(matrix.diagonal(), 0):
            raise ValueError('Distance cache is not symmetric with a zero diagonal')
        blocks['raw' if depth == 0 else f'tree_{depth}'] = block
    h = np.asarray(propagated_features)
    if h.ndim != 2 or len(h) != len(x) or not np.isfinite(h).all():
        raise ValueError('Invalid propagated feature matrix')
    blocks['grip_S2X'] = cdist(h[val_ids], h[train_ids], metric='euclidean')
    blocks['degree_only'] = np.abs(np.log1p(degree[val_ids, None]) - np.log1p(degree[None, train_ids]))
    result = evaluate_distance_blocks(blocks, y[train_ids], y[val_ids], degree[train_ids], degree[val_ids], ks, voting)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        table.to_csv(output / f'{name}.csv', index=False)
    split_digest = hashlib.sha256()
    for values in (train_ids, val_ids, y[train_ids], y[val_ids]):
        split_digest.update(np.ascontiguousarray(values, dtype=np.int64).tobytes())
    config = dict(distance_protocol=protocol, split_sha256=split_digest.hexdigest(),
                  depths=list(depths), ks=list(ks), voting=list(voting),
                  training_nodes=len(train_ids), validation_nodes=len(val_ids),
                  selection='validation_accuracy_then_smallest_k_then_voting_name',
                  tie_break='nearest_member_of_tied_classes_then_training_node_id',
                  degree_matching='exact_candidate_degree_within_each_validation_query',
                  test_labels_used=False)
    (output / 'protocol.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    return result
