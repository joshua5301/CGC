import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import softmax
from scipy.stats import spearmanr

from src.gnn_distance_probe import _probe_distances, distance_geometry
from src.tree_distance import _neighbors


def decompose_logits(logits):
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError('Require finite two-dimensional logits')
    z = z - z.mean(1, keepdims=True)
    radius = np.linalg.norm(z, axis=1)
    valid = radius > 1e-12
    unit = np.divide(z, radius[:, None], out=np.zeros_like(z), where=valid[:, None])
    angular = np.clip(1 - unit @ unit.T, 0, 2)
    radial_sq = (radius[:, None] - radius[None, :]) ** 2
    angular_sq = 2 * radius[:, None] * radius[None, :] * angular
    direction = angular.copy()
    direction[~(valid[:, None] & valid[None, :])] = np.nan
    probability = softmax(z, axis=1)
    gaps = dict(centered_logits=cdist(z, z), direction=direction,
                magnitude=np.sqrt(radial_sq),
                probability_tv=.5 * cdist(probability, probability, 'cityblock'))
    return gaps, radius, probability, radial_sq, angular_sq


def _corr(a, b):
    good = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[good], b[good]).statistic) if good.sum() > 2 and np.ptp(a[good]) and np.ptp(b[good]) else np.nan


def analyze_saved_probe(x, edge_index, propagated_features, raw_distance_dir,
                        normalized_distance_dir, run_dir, ks=(1, 5, 10, 20),
                        zero_tolerance=1e-8, output_dir=None):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'protocol.json').read_text())
    ids = np.load(run_dir / 'probe_ids.npy')
    x, edges, h = np.asarray(x), np.asarray(edge_index), np.asarray(propagated_features)
    matrices, _ = _probe_distances(x, edges, h, ids, raw_distance_dir,
                                  normalized_distance_dir, config['depths'])
    geometry, _ = distance_geometry(matrices)
    neighbors, _ = _neighbors(edges, len(x), False)
    degree = np.array([len(v) for v in neighbors])[ids]
    pairs = np.triu_indices(len(ids), 1)
    degree_gap = np.abs(degree[:, None] - degree[None, :])[pairs]
    local, health, zero_pairs, decomposition, correlations = [], [], [], [], []
    seen = set()
    if not ks or any(int(k) != k or not 1 <= k < len(ids) for k in ks):
        raise ValueError('Invalid neighborhood sizes')
    for size in config['train_sizes']:
        for subset in config['subset_seeds']:
            for model in config['models']:
                for seed in config['model_seeds']:
                    path = run_dir / f'{model}_n{size}_subset{subset}_seed{seed}.npz'
                    with np.load(path) as saved:
                        for phase in ('initial', 'trained'):
                            key = (size, model, seed)
                            if phase == 'initial' and key in seen:
                                continue
                            if phase == 'initial':
                                seen.add(key)
                            meta = dict(train_size=size, model=model, model_seed=seed,
                                        subset_seed=-1 if phase == 'initial' else subset, phase=phase)
                            gaps, radius, prob, radial_sq, angular_sq = decompose_logits(saved[phase])
                            entropy = -(prob * np.log(np.maximum(prob, 1e-300))).sum(1)
                            health.append(dict(**meta, norm_median=np.median(radius), norm_p99=np.quantile(radius, .99),
                                               confidence_mean=prob.max(1).mean(), fraction_confident_99=(prob.max(1) > .99).mean(),
                                               entropy_mean=entropy.mean(), degree_norm_spearman=_corr(degree, radius),
                                               train_ce=float(saved['train_ce']) if phase == 'trained' else np.nan,
                                               train_accuracy=100 * float(saved['train_accuracy']) if phase == 'trained' else np.nan))
                            for method, item in geometry.items():
                                for metric, gap in gaps.items():
                                    all_values = gap[pairs]
                                    finite = all_values[np.isfinite(all_values)]
                                    reference = finite.mean() if len(finite) else np.nan
                                    correlations.append(dict(**meta, method=method, metric=metric,
                                                             distance_output_spearman=_corr(item['distance'], all_values),
                                                             distance_degree_gap_spearman=_corr(item['distance'], degree_gap),
                                                             degree_gap_output_spearman=_corr(degree_gap, all_values)))
                                    for k in ks:
                                        index = (np.arange(len(ids))[:, None], item['neighbors'][:, :k])
                                        selected = gap[index]
                                        selected = selected[np.isfinite(selected)]
                                        local.append(dict(**meta, method=method, metric=metric, k=k,
                                                          relative_gap=selected.mean() / reference if len(selected) and reference > 1e-12 else np.nan,
                                                          gap_mean=selected.mean() if len(selected) else np.nan,
                                                          valid_pair_fraction=len(selected) / (len(ids) * k)))
                                for k in ks:
                                    index = (np.arange(len(ids))[:, None], item['neighbors'][:, :k])
                                    radial = radial_sq[index].sum()
                                    angular = angular_sq[index].sum()
                                    total = radial + angular
                                    all_total = (radial_sq[pairs] + angular_sq[pairs]).mean()
                                    decomposition.append(dict(**meta, method=method, k=k,
                                                              radial_fraction=radial / total if total > 1e-12 else np.nan,
                                                              radial_contribution=radial_sq[index].mean() / all_total if all_total > 1e-12 else np.nan,
                                                              angular_contribution=angular_sq[index].mean() / all_total if all_total > 1e-12 else np.nan))
                                selected = np.flatnonzero(item['distance'] <= zero_tolerance)
                                for pair in selected:
                                    i, j = pairs[0][pair], pairs[1][pair]
                                    zero_pairs.append(dict(**meta, method=method, node_i=ids[i], node_j=ids[j],
                                                           distance=matrices[method][i, j], normalized_distance=item['distance'][pair],
                                                           exact_zero=bool(item['distance'][pair] == 0),
                                                           logit_gap=gaps['centered_logits'][i, j], tv_gap=gaps['probability_tv'][i, j],
                                                           relative_logit_gap=gaps['centered_logits'][i, j] / max(np.median(gaps['centered_logits'][pairs]), 1e-12),
                                                           prediction_disagrees=bool(prob[i].argmax() != prob[j].argmax()),
                                                           degree_i=degree[i], degree_j=degree[j]))
    report = dict(local=pd.DataFrame(local), health=pd.DataFrame(health),
                  decomposition=pd.DataFrame(decomposition), correlations=pd.DataFrame(correlations),
                  zero_pairs=pd.DataFrame(zero_pairs))
    groups = ['train_size', 'model', 'method', 'metric', 'k']
    report['summary'] = report['local'].groupby(groups + ['phase'], as_index=False).agg(
        relative_gap_mean=('relative_gap', 'mean'), relative_gap_std=('relative_gap', 'std'),
        valid_pair_fraction=('valid_pair_fraction', 'mean'))
    initial = report['local'].query("phase == 'initial'").drop(columns=['subset_seed', 'phase'])
    trained = report['local'].query("phase == 'trained'")
    paired = trained.merge(initial[groups + ['model_seed', 'relative_gap']],
                           on=groups + ['model_seed'], suffixes=('_trained', '_initial'), validate='many_to_one')
    paired['delta'] = paired.relative_gap_trained - paired.relative_gap_initial
    report['paired_changes'] = paired
    report['change_summary'] = paired.groupby(groups, as_index=False).agg(
        delta_mean=('delta', 'mean'), delta_std=('delta', 'std'))
    output = Path(output_dir) if output_dir else run_dir / 'diagnostics_v1'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'settings.json').write_text(json.dumps(dict(run_dir=str(run_dir), ks=list(ks), zero_tolerance=zero_tolerance), indent=2))
    for name, table in report.items():
        table.to_csv(output / f'{name}.csv', index=False)
    report['folder'] = str(output)
    return report
