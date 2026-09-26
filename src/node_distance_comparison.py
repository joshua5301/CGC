import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from tqdm.auto import tqdm

from src.gnn_distance_probe import distance_geometry, output_stability
from src.node_distances import array_digest, build_node_distances, pair_scale
from src.tree_distance import _neighbors, _write_json


def assign_medoids(distance, medoids):
    assignment = np.argmin(distance[:, medoids], axis=1)
    assignment[medoids] = np.arange(len(medoids))
    return assignment


def select_medoids(distance, budget, seeds=range(10), max_steps=100):
    distance = np.asarray(distance, dtype=np.float64)
    if not 1 <= budget <= len(distance) or not seeds or max_steps < 1:
        raise ValueError('Invalid representative budget or optimization settings')
    distance = distance / pair_scale(distance)
    best, records = None, []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        medoids = [int(rng.integers(len(distance)))]
        nearest = distance[:, medoids[0]].copy()
        while len(medoids) < budget:
            nearest[medoids] = 0.
            selected = (int(rng.choice(len(distance), p=nearest / nearest.sum())) if nearest.sum() > 0 else
                        int(rng.choice(np.setdiff1d(np.arange(len(distance)), medoids))))
            medoids.append(selected)
            nearest = np.minimum(nearest, distance[:, selected])
        medoids = np.array(medoids)
        converged = False
        for step in range(max_steps):
            assignment = assign_medoids(distance, medoids)
            updated = medoids.copy()
            for cluster, current in enumerate(medoids):
                members = np.flatnonzero(assignment == cluster)
                costs = distance[np.ix_(members, members)].sum(0)
                candidate = members[np.argmin(costs)]
                if costs.min() < distance[members, current].sum() - 1e-12:
                    updated[cluster] = candidate
            if np.array_equal(updated, medoids):
                converged = True
                break
            medoids = updated
        assignment = assign_medoids(distance, medoids)
        objective = float(distance[np.arange(len(distance)), medoids[assignment]].mean())
        records.append(dict(seed=int(seed), objective=objective, converged=converged, steps=step + 1))
        if best is None or objective < best['objective']:
            best = dict(medoids=medoids.copy(), assignment=assignment.copy(), **records[-1])
    return best, records


def ce_preservation(logits, targets, representatives, mask=None):
    logits, targets = np.asarray(logits, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    representatives = np.asarray(representatives, dtype=np.int64)
    if logits.ndim != 2 or logits.shape != targets.shape or not np.isfinite(logits).all() or not np.isfinite(targets).all():
        raise ValueError('Require matching finite logits and soft targets')
    if (targets < 0).any() or not np.allclose(targets.sum(1), 1.) or representatives.shape != (len(logits),) or representatives.min() < 0 or representatives.max() >= len(logits):
        raise ValueError('Invalid probabilities or representative mapping')
    targets = targets / targets.sum(1, keepdims=True)
    mask = np.ones(len(logits), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != (len(logits),) or not mask.any():
        raise ValueError('Require a nonempty evaluation mask')
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    log_gap = (logp - logp[representatives])[mask]
    delta = (targets[mask] * log_gap).sum(1)
    z_gap = (logits - logits[representatives])[mask]
    centered = z_gap - z_gap.mean(1, keepdims=True)
    original_risk = float(-(targets[mask] * logp[mask]).sum(1).mean())
    return dict(evaluation_nodes=int(mask.sum()), original_risk=original_risk,
                representative_risk=original_risk + float(delta.mean()),
                signed_risk_change=float(delta.mean()), risk_gap=float(abs(delta.mean())),
                E_CE=float(np.abs(delta).mean()), E_robust=float(np.abs(log_gap).max(1).mean()),
                range_bound=float(np.ptp(z_gap, axis=1).mean()),
                centered_l2_bound=float(np.sqrt(2) * np.linalg.norm(centered, axis=1).mean()))


def _key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def _save_arrays(path, **arrays):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def compare_node_distances(x, edge_index, feature_stages, run_dir, output_dir,
                           models=('gcn', 'sage', 'gin'), budgets=(13, 70), depth=2,
                           root_weight=.5, self_loops=False, rff_width=512,
                           rff_seed=2026, medoid_seeds=range(10), medoid_steps=100,
                           ks=(1, 5, 10, 20), checkpoint_rows=16):
    x, edge_index = np.asarray(x), np.asarray(edge_index)
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    source = json.loads((run_dir / 'protocol.json').read_text())
    ids = np.load(run_dir / 'probe_ids.npy')
    if source.get('target_kind') != 'teacher_soft_labels' or source.get('graph') != 'full_original_graph':
        raise ValueError('Use the saved full-graph probe with teacher soft targets')
    if not models or set(models) - set(source['models']):
        raise ValueError('Requested architecture is absent from the saved probe')
    graph = source['distance_protocol']
    _, canonical = _neighbors(edge_index, len(x), graph['self_loops'])
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    digest.update(canonical.tobytes())
    if graph['input_sha256'] != digest.hexdigest() or graph['shape'] != list(x.shape):
        raise ValueError('Graph/features differ from those used for the saved student outputs')
    if ids.ndim != 1 or len(ids) < 2 or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(x):
        raise ValueError('Invalid saved probe node IDs')
    if not budgets or any(int(b) != b or not 1 <= b <= len(ids) for b in budgets):
        raise ValueError('Representative budgets must fit the saved probe pool')
    if not ks or any(int(k) != k or not 1 <= k < len(ids) for k in ks):
        raise ValueError('Invalid neighbor counts')
    if not medoid_seeds or medoid_steps < 1:
        raise ValueError('Require at least one representative restart and update step')
    budgets = tuple(int(budget) for budget in budgets)
    with np.load(run_dir / 'teacher_predictions.npz') as saved:
        targets = saved['probabilities'].astype(np.float64)
    if targets.ndim != 2 or len(targets) != len(x) or not np.isfinite(targets).all() or (targets < 0).any() or not np.allclose(targets.sum(1), 1):
        raise ValueError('Invalid saved all-node teacher probabilities')
    targets = targets[ids]
    targets /= targets.sum(1, keepdims=True)
    runs, fingerprints = [], {}
    for size in source['train_sizes']:
        for subset in source['subset_seeds']:
            for model in models:
                for seed in source['model_seeds']:
                    filename = f'{model}_n{size}_subset{subset}_seed{seed}.npz'
                    with np.load(run_dir / filename) as saved:
                        logits, train_ids = saved['trained'].copy(), saved['train_ids'].copy()
                    if logits.shape != targets.shape or not np.isfinite(logits).all():
                        raise ValueError(f'Saved logits do not match the probe/teacher: {filename}')
                    if train_ids.shape != (size,) or len(np.unique(train_ids)) != size or train_ids.min() < 0 or train_ids.max() >= len(x):
                        raise ValueError(f'Invalid saved training nodes: {filename}')
                    metadata = dict(train_size=size, subset_seed=subset, model=model, model_seed=seed)
                    runs.append((metadata, logits, train_ids))
                    fingerprints[filename] = array_digest(logits, train_ids)
    settings = dict(version=1, depth=depth, root_weight=root_weight, self_loops=self_loops,
                    rff_width=rff_width, rff_seed=rff_seed,
                    input_sha256=array_digest(x, canonical, ids, *feature_stages))
    cache = output_dir / 'distances' / _key(settings)
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / 'distances.npz').exists() and (cache / 'metadata.json').exists():
        with np.load(cache / 'distances.npz') as saved:
            matrices = {name: saved[name].copy() for name in saved.files}
        details = json.loads((cache / 'metadata.json').read_text())
    else:
        matrices, details = build_node_distances(x, edge_index, feature_stages, ids, cache,
                                                depth, root_weight, self_loops, rff_width,
                                                rff_seed, checkpoint_rows)
        _save_arrays(cache / 'distances.npz', **matrices)
        _write_json(cache / 'metadata.json', details)
    geometry, scales = distance_geometry(matrices)
    for matrix in matrices.values():
        if not np.allclose(np.diag(matrix), 0.):
            raise ValueError('Distance diagonal must be zero')
    config = dict(version=1, source_dir=str(run_dir), source_protocol=source,
                  distance_settings=settings, distance_cache=str(cache), models=list(models),
                  source_logits_sha256=fingerprints, teacher_sha256=array_digest(targets),
                  budgets=list(budgets), medoid_seeds=list(medoid_seeds), medoid_steps=medoid_steps,
                  ks=list(ks), evaluation='saved_trained_logits', selection='distance_objective_only',
                  weighting='cluster_mass_and_mean_teacher_target', normalization='common_pair_median')
    output = output_dir / _key(config)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / 'protocol.json', config)
    selected, selection_rows, restart_rows = {}, [], []
    with tqdm(total=len(matrices) * len(budgets), desc='Representative selection') as bar:
        for budget in budgets:
            for method, matrix in matrices.items():
                best, restarts = select_medoids(matrix, budget, medoid_seeds, medoid_steps)
                selected[budget, method] = best['medoids']
                selection_rows.append(dict(method=method, budget=budget,
                                           **{key: best[key] for key in ('seed', 'objective', 'converged', 'steps')}))
                restart_rows.extend(dict(method=method, budget=budget, **row) for row in restarts)
                bar.update()
    mappings = []
    for budget in budgets:
        for mode in ('own_representatives', 'S2X_representatives'):
            for method, matrix in matrices.items():
                selection = method if mode == 'own_representatives' else 'grip_S2X'
                medoids = selected[budget, selection]
                assignment = assign_medoids(matrix, medoids)
                representative = medoids[assignment]
                counts = np.bincount(assignment, minlength=budget)
                labels = np.zeros((budget, targets.shape[1]))
                np.add.at(labels, assignment, targets)
                labels /= counts[:, None]
                _save_arrays(output / f'{mode}_{method}_budget{budget}.npz', probe_node_ids=ids,
                             medoid_node_ids=ids[medoids], representative_node_ids=ids[representative],
                             assignment=assignment, cluster_sizes=counts, labels=labels, weights=counts / len(ids))
                mappings.append((dict(method=method, budget=budget, assignment_mode=mode), representative))
    ce_rows, local_parts, tail_parts, health_parts = [], [], [], []
    for metadata, logits, train_ids in tqdm(runs, desc='Frozen-output CE preservation'):
        masks = dict(all_probe=np.ones(len(ids), dtype=bool), outside_student_train=~np.isin(ids, train_ids))
        for mapping_meta, representative in mappings:
            for scope, mask in masks.items():
                if mask.any():
                    ce_rows.append(dict(**metadata, **mapping_meta, scope=scope,
                                        **ce_preservation(logits, targets, representative, mask)))
        stability = output_stability(logits, geometry, ks)
        local_parts.append(stability['local'].assign(**metadata))
        tail_parts.append(stability['tails'].assign(**metadata))
        health_parts.append(stability['health'].assign(**metadata))
    per_run = pd.DataFrame(ce_rows)
    groups = ['train_size', 'model', 'budget', 'assignment_mode', 'scope', 'method']
    summary = per_run.groupby(groups, as_index=False).agg(
        E_CE_mean=('E_CE', 'mean'), E_CE_std=('E_CE', 'std'),
        E_robust_mean=('E_robust', 'mean'), risk_gap_mean=('risk_gap', 'mean'),
        signed_risk_change_mean=('signed_risk_change', 'mean'), original_risk_mean=('original_risk', 'mean'),
        range_bound_mean=('range_bound', 'mean'), runs=('E_CE', 'count'))
    keys = ['train_size', 'model', 'subset_seed', 'model_seed', 'budget', 'assignment_mode', 'scope']
    baseline = per_run[per_run.method == 'grip_S2X'][keys + ['E_CE']].rename(columns={'E_CE': 'baseline_E_CE'})
    paired = per_run.merge(baseline, on=keys, validate='many_to_one')
    paired['delta_E_CE'] = paired.E_CE - paired.baseline_E_CE
    paired['win'] = paired.delta_E_CE < -1e-12
    paired_summary = paired.groupby(groups, as_index=False).agg(
        delta_E_CE_mean=('delta_E_CE', 'mean'), delta_E_CE_std=('delta_E_CE', 'std'),
        win_fraction=('win', 'mean'), runs=('win', 'count'))
    local = pd.concat(local_parts, ignore_index=True)
    local_summary = local.groupby(['train_size', 'model', 'method', 'metric', 'k'], as_index=False).agg(
        relative_gap_mean=('relative_gap', 'mean'), relative_gap_std=('relative_gap', 'std'))
    report = dict(per_run=per_run, summary=summary, paired=paired, paired_summary=paired_summary,
                  local=local, local_summary=local_summary, tails=pd.concat(tail_parts, ignore_index=True),
                  health=pd.concat(health_parts, ignore_index=True), selection=pd.DataFrame(selection_rows),
                  restarts=pd.DataFrame(restart_rows), distance_scales=scales, timing=pd.DataFrame(details['timing']))
    for name, table in report.items():
        table.to_csv(output / f'{name}.csv', index=False)
    report.update(folder=str(output), methods=list(matrices), models=list(models))
    return report


def plot_node_distance_comparison(report, scope='all_probe'):
    import matplotlib.pyplot as plt
    methods, models = report['methods'], report['models']
    labels = ['S²X', 'Multiscale', 'Prob. OT', 'Neighbor MMD']
    colors = ['#687b90', '#e58b37', '#318b71', '#8064b1']
    table = report['summary'].query('scope == @scope')
    figures = []
    for size in sorted(table.train_size.unique()):
        for budget in sorted(table.budget.unique()):
            fig, axes = plt.subplots(2, len(models), figsize=(5 * len(models), 7), squeeze=False)
            for row, mode in enumerate(('own_representatives', 'S2X_representatives')):
                for col, model in enumerate(models):
                    part = table.query('train_size == @size and budget == @budget and model == @model and assignment_mode == @mode').set_index('method').reindex(methods)
                    ax = axes[row, col]
                    ax.bar(labels, part.E_CE_mean, yerr=part.E_CE_std.fillna(0), color=colors, capsize=4)
                    ax.scatter(labels, part.risk_gap_mean, color='black', marker='_', s=120, label='Absolute risk gap')
                    ax.set(title=f'{model.upper()} | {mode.replace("_", " ")}', ylabel='CE change (lower is better)')
                    ax.tick_params(axis='x', labelrotation=15)
                    ax.legend(fontsize=8)
            fig.suptitle(f'Frozen outputs | student train n={size} | representatives={budget} | {scope}\nBars: mean absolute per-node CE change; error bars: run SD')
            fig.tight_layout()
            fig.savefig(Path(report['folder']) / f'ce_n{size}_budget{budget}_{scope}.png', dpi=170, bbox_inches='tight')
            figures.append(fig)
    paired = report['paired_summary'].query('scope == @scope and assignment_mode == "own_representatives"')
    for size in sorted(paired.train_size.unique()):
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.8), squeeze=False)
        for col, model in enumerate(models):
            ax = axes[0, col]
            budgets = sorted(paired.budget.unique())
            for index, budget in enumerate(budgets):
                part = paired.query('train_size == @size and model == @model and budget == @budget').set_index('method').reindex(methods[1:])
                positions = np.arange(3) + (index - (len(budgets) - 1) / 2) * .18
                ax.errorbar(positions, part.delta_E_CE_mean, yerr=part.delta_E_CE_std.fillna(0), fmt='o', capsize=4, label=f'm={budget}')
            ax.axhline(0, color='black', linewidth=.8)
            ax.set(xticks=np.arange(3), xticklabels=labels[1:], title=model.upper(), ylabel='Paired E_CE minus S²X')
            ax.legend()
        fig.suptitle(f'Student train n={size} | negative favors alternative | error bars: paired run SD')
        fig.tight_layout()
        fig.savefig(Path(report['folder']) / f'paired_ce_n{size}_{scope}.png', dpi=170, bbox_inches='tight')
        figures.append(fig)
    local = report['local_summary']
    for size in sorted(local.train_size.unique()):
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.8), squeeze=False)
        for col, model in enumerate(models):
            ax = axes[0, col]
            for method, label, color in zip(methods, labels, colors):
                part = local.query('train_size == @size and model == @model and method == @method and metric == "probability_tv"').sort_values('k')
                ax.plot(part.k, part.relative_gap_mean, 'o-', label=label, color=color)
            ax.axhline(1, color='black', linewidth=.8, linestyle='--')
            ax.set(title=model.upper(), xlabel='Nearest neighbors k', ylabel='Neighbor TV / all-pair TV')
            ax.legend(fontsize=8)
        fig.suptitle(f'Student train n={size} | fixed probe nodes | lower is better')
        fig.tight_layout()
        fig.savefig(Path(report['folder']) / f'neighbors_n{size}.png', dpi=170, bbox_inches='tight')
        figures.append(fig)
    return figures
