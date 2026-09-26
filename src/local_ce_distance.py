import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from tqdm.auto import tqdm

from src.node_distances import array_digest, pair_scale, transition_neighbors
from src.tree_distance import _neighbors, _write_json


def ce_change_matrix(logits, targets):
    logits, targets = np.asarray(logits, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if logits.shape != targets.shape or logits.ndim != 2 or not np.isfinite(logits).all() or not np.isfinite(targets).all():
        raise ValueError('Require matching finite logits and targets')
    if (targets < 0).any() or not np.allclose(targets.sum(1), 1):
        raise ValueError('Targets must be probability vectors')
    targets = targets / targets.sum(1, keepdims=True)
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    cross = -targets @ logp.T
    change = np.abs(cross - np.diag(cross)[:, None])
    np.fill_diagonal(change, 0.)
    return change


def fit_l1_scale(distance, change):
    distance, change = np.asarray(distance).ravel(), np.asarray(change).ravel()
    good = distance > 0
    if not good.any():
        return 0.
    ratios, weights = change[good] / distance[good], distance[good]
    order = np.argsort(ratios, kind='stable')
    index = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)
    return float(ratios[order[index]])


def local_pair_masks(distance, fractions, ks):
    nodes = len(distance)
    off = ~np.eye(nodes, dtype=bool)
    masks = []
    for fraction in fractions:
        threshold = float(np.quantile(distance[off], fraction))
        masks.append(('quantile', float(fraction), threshold, off & (distance <= threshold)))
    ranked = distance.copy()
    np.fill_diagonal(ranked, np.inf)
    order = np.argsort(ranked, axis=1, kind='stable')
    for k in ks:
        mask = np.zeros_like(off)
        mask[np.arange(nodes)[:, None], order[:, :k]] = True
        masks.append(('knn', int(k), np.nan, mask))
    return masks


def evaluate_local_pairs(distance, change, slope, fractions=(.01, .02, .05, .1), ks=(1, 5, 10, 20)):
    off = ~np.eye(len(distance), dtype=bool)
    reference = float(change[off].mean())
    denominator = reference if reference > 1e-12 else np.nan
    rows = []
    for selection, cutoff, threshold, mask in local_pair_masks(distance, fractions, ks):
        values = change[mask]
        residual = np.abs(slope * distance[mask] - values)
        rows.append(dict(selection=selection, cutoff=cutoff, threshold=threshold,
                         pairs=int(mask.sum()), pair_fraction=float(mask.sum() / off.sum()),
                         query_coverage=float(mask.any(1).mean()),
                         zero_distance_fraction=float((distance[mask] == 0).mean()),
                         ce_mean=float(values.mean()), ce_p95=float(np.quantile(values, .95)),
                         relative_ce_mean=float(values.mean() / denominator),
                         relative_ce_p95=float(np.quantile(values, .95) / denominator),
                         fit_mae=float(residual.mean()), relative_fit_mae=float(residual.mean() / denominator),
                         all_pair_ce_mean=reference, degenerate=reference <= 1e-12))
    return rows


def matched_mean_distance(x, edge_index, ids, settings):
    neighbors = transition_neighbors(edge_index, len(x), settings['self_loops'])
    degree = np.array([len(v) for v in neighbors])
    transition = csr_matrix((np.repeat(1. / degree, degree),
                             (np.repeat(np.arange(len(x)), degree), np.concatenate(neighbors))),
                            shape=(len(x), len(x)))
    z = np.asarray(x, dtype=np.float64).copy()
    alpha = settings['root_weight']
    for _ in range(settings['depth']):
        z = alpha * z + (1 - alpha) * (transition @ z)
    return cdist(z[ids], z[ids])


def run_local_ce_comparison(result_dir, x, edge_index, fractions=(.01, .02, .05, .1),
                             ks=(1, 5, 10, 20), calibration_fraction=.25, split_seed=2026):
    result_dir = Path(result_dir)
    source = json.loads((result_dir / 'protocol.json').read_text())
    run = Path(source['source_dir'])
    ids = np.load(run / 'probe_ids.npy')
    graph = source['source_protocol']['distance_protocol']
    _, edges = _neighbors(edge_index, len(x), graph['self_loops'])
    digest = hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes() + edges.tobytes()).hexdigest()
    if digest != graph['input_sha256']:
        raise ValueError('Graph/features do not match the saved student outputs')
    count = int(len(ids) * calibration_fraction)
    if not 2 <= count <= len(ids) - 2:
        raise ValueError('Require at least two calibration and evaluation nodes')
    if not fractions or any(not 0 < f <= 1 for f in fractions):
        raise ValueError('Require fractions in (0, 1]')
    if not ks or any(int(k) != k or not 1 <= k < len(ids) - count for k in ks):
        raise ValueError('Invalid nearest-neighbor counts')
    order = np.random.default_rng(split_seed).permutation(len(ids))
    calibration, evaluation = order[:count], order[count:]
    with np.load(Path(source['distance_cache']) / 'distances.npz') as saved:
        matrices = {name: saved[name].copy() for name in saved.files}
    matrices['matched_mean'] = matched_mean_distance(x, edge_index, ids, source['distance_settings'])
    with np.load(run / 'teacher_predictions.npz') as saved:
        targets = saved['probabilities'][ids].astype(np.float64)
    targets /= targets.sum(1, keepdims=True)
    if array_digest(targets) != source['teacher_sha256']:
        raise ValueError('Teacher probabilities changed since the previous comparison')
    config = dict(version=1, source_result=str(result_dir), source_protocol=source,
                  fractions=list(fractions), ks=list(ks), calibration_fraction=calibration_fraction,
                  split_seed=split_seed, calibration_nodes=ids[calibration].tolist(),
                  evaluation_nodes=ids[evaluation].tolist(), slope_fit='all_calibration_ordered_pairs_l1',
                  methods=list(matrices))
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    output = result_dir / 'local_ce' / key
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / 'protocol.json', config)
    np.savez_compressed(output / 'distances.npz', **matrices)
    np.save(output / 'probe_ids.npy', ids)
    cal_off = ~np.eye(count, dtype=bool)
    geometry = {}
    for method, matrix in matrices.items():
        if matrix.shape != (len(ids), len(ids)) or not np.isfinite(matrix).all() or (matrix < 0).any() or not np.allclose(matrix, matrix.T) or not np.allclose(np.diag(matrix), 0):
            raise ValueError(f'Invalid distance matrix: {method}')
        cal = matrix[np.ix_(calibration, calibration)]
        scale = pair_scale(cal)
        geometry[method] = (cal / scale, matrix[np.ix_(evaluation, evaluation)] / scale, scale)
    rows, fits = [], []
    for filename, fingerprint in tqdm(source['source_logits_sha256'].items(), desc='Local CE preservation'):
        with np.load(run / filename) as saved:
            logits, train_ids = saved['trained'], saved['train_ids']
            if array_digest(logits, train_ids) != fingerprint:
                raise ValueError(f'Saved student outputs changed: {filename}')
            change = ce_change_matrix(logits, targets)
        np.save(output / f'ce_{Path(filename).stem}.npy', change)
        model, size, subset, seed = Path(filename).stem.split('_')
        metadata = dict(model=model, train_size=int(size[1:]), subset_seed=int(subset[6:]),
                        model_seed=int(seed[4:]), evaluation_train_overlap=int(np.isin(ids[evaluation], train_ids).sum()))
        cal_change = change[np.ix_(calibration, calibration)]
        eval_change = change[np.ix_(evaluation, evaluation)]
        for method, (cal_distance, eval_distance, scale) in geometry.items():
            slope = fit_l1_scale(cal_distance[cal_off], cal_change[cal_off])
            fits.append(dict(**metadata, method=method, slope=slope, distance_scale=scale,
                             calibration_mae=float(np.abs(slope * cal_distance[cal_off] - cal_change[cal_off]).mean())))
            rows.extend(dict(**metadata, method=method, **row) for row in
                        evaluate_local_pairs(eval_distance, eval_change, slope, fractions, ks))
    per_run = pd.DataFrame(rows)
    groups = ['train_size', 'model', 'method', 'selection', 'cutoff']
    metrics = ['ce_mean', 'ce_p95', 'relative_ce_mean', 'relative_ce_p95', 'fit_mae',
               'relative_fit_mae', 'pair_fraction', 'query_coverage', 'zero_distance_fraction']
    aggregation = {f'{metric}_{stat}': (metric, stat) for metric in metrics for stat in ('mean', 'std')}
    summary = per_run.groupby(groups, as_index=False).agg(**aggregation, runs=('ce_mean', 'count'))
    keys = ['train_size', 'model', 'subset_seed', 'model_seed', 'selection', 'cutoff']
    baseline = per_run[per_run.method == 'grip_S2X'][keys + ['relative_ce_mean']]
    paired = per_run.merge(baseline.rename(columns={'relative_ce_mean': 'baseline'}), on=keys, validate='many_to_one')
    paired['delta'] = paired.relative_ce_mean - paired.baseline
    paired_summary = paired.groupby(groups, as_index=False).agg(delta_mean=('delta', 'mean'), delta_std=('delta', 'std'))
    summary = summary.merge(paired_summary, on=groups, validate='one_to_one')
    report = dict(per_run=per_run, summary=summary, fits=pd.DataFrame(fits), paired=paired)
    for name, table in report.items():
        table.to_csv(output / f'{name}.csv', index=False)
    report.update(folder=str(output), models=source['models'], methods=list(matrices),
                  calibration_nodes=count, evaluation_nodes=len(evaluation))
    return report


def plot_local_ce_comparison(report):
    import matplotlib.pyplot as plt
    labels = dict(grip_S2X='S²X', multiscale='Multiscale', probability_ot='Probability OT',
                  neighborhood_mmd='Neighbor MMD', matched_mean='Matched mean')
    models, methods = report['models'], report['methods']
    metrics = [('relative_ce_mean', 'Mean CE change / all-pair mean'),
               ('relative_ce_p95', 'P95 CE change / all-pair mean'),
               ('relative_fit_mae', 'Calibrated MAE / all-pair mean')]
    figures = []
    for size in sorted(report['summary'].train_size.unique()):
        for selection in ('quantile', 'knn'):
            fig, axes = plt.subplots(3, len(models), figsize=(5 * len(models), 10), squeeze=False)
            for col, model in enumerate(models):
                for row, (metric, ylabel) in enumerate(metrics):
                    ax = axes[row, col]
                    for method in methods:
                        part = report['summary'].query('train_size == @size and model == @model and method == @method and selection == @selection').sort_values('cutoff')
                        x = part.cutoff.to_numpy() * (100 if selection == 'quantile' else 1)
                        mean, std = part[f'{metric}_mean'].to_numpy(), part[f'{metric}_std'].fillna(0).to_numpy()
                        ax.plot(x, mean, 'o-', label=labels.get(method, method))
                        ax.fill_between(x, mean - std, mean + std, alpha=.1)
                    ax.set(title=model.upper(), ylabel=ylabel,
                           xlabel='Closest pair percentile (%)' if selection == 'quantile' else 'Neighbors per node k')
                    if row == 0:
                        ax.axhline(1, color='gray', linestyle='--', linewidth=.8)
                    ax.legend(fontsize=8)
            fig.suptitle(f'Frozen students n={size} | disjoint calibration/evaluation nodes | lower is better\nBands: descriptive run SD')
            fig.tight_layout()
            fig.savefig(Path(report['folder']) / f'{selection}_n{size}.png', dpi=160, bbox_inches='tight')
            figures.append(fig)
    return figures
