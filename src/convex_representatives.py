import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.dataloader import get_dataset
from src.node_distances import array_digest
from src.partition import EPS, geometric_medians
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student
from src.trained_teacher_kernel import recover_teacher
from src.tree_distance import _write_json


def identity_hidden(model, x):
    layer = model.layers[0]
    value = layer.lin(x)
    return (value if layer.bias is None else value + layer.bias).relu()


def convex_features(h, assignment, weights, clusters):
    return h.new_zeros(clusters, h.shape[1]).index_add_(0, assignment, weights[:, None] * h)


def cluster_softmax(scores, assignment, clusters):
    maxima = scores.new_full((clusters,), -torch.inf)
    maxima.scatter_reduce_(0, assignment, scores.detach(), reduce='amax', include_self=True)
    weights = (scores - maxima[assignment]).exp()
    total = scores.new_zeros(clusters).index_add_(0, assignment, weights)
    return weights / total[assignment]


@torch.no_grad()
def median_weights(h, assignment, clusters):
    counts = torch.bincount(assignment, minlength=clusters).to(h.dtype)
    weights = 1 / counts[assignment]
    centers = convex_features(h, assignment, weights, clusters)
    for _ in range(30):
        weights = 1 / (h - centers[assignment]).norm(dim=1).clamp_min(EPS)
        total = h.new_zeros(clusters).index_add_(0, assignment, weights)
        centers = h.new_zeros(clusters, h.shape[1]).index_add_(0, assignment, weights[:, None] * h) / total[:, None]
    return weights / total[assignment]


def fit_convex_representatives(model, h, assignment, targets, baseline, steps=1000, lr=.05, log_every=10,
                               progress=True):
    if steps < 1 or lr <= 0 or log_every < 1:
        raise ValueError('Require positive steps, learning rate and logging interval')
    model.eval().requires_grad_(False)
    h, assignment = h.double(), assignment.to(h.device)
    targets, baseline = targets.to(h.device), baseline.to(h.device)
    clusters = len(targets)
    weights = median_weights(h, assignment, clusters)
    initial = convex_features(h, assignment, weights, clusters).float()
    if not torch.allclose(initial, baseline, atol=1e-5, rtol=1e-4):
        raise ValueError('Convex initialization does not reproduce the saved geometric medians')
    scores = weights.log().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([scores], lr=lr)
    best_loss, best_weights, best_step = float('inf'), None, 0
    history = []
    for step in tqdm(range(steps + 1), desc='Within-cluster convex reconstruction', disable=not progress):
        weights = cluster_softmax(scores, assignment, clusters)
        representatives = convex_features(h, assignment, weights, clusters).float()
        loss = (identity_hidden(model, representatives) - targets).square().sum(1).mean()
        value = float(loss.detach())
        if not np.isfinite(value):
            raise FloatingPointError('Nonfinite hidden reconstruction loss')
        if value < best_loss:
            best_loss, best_weights, best_step = value, weights.detach().clone(), step
        if step % log_every == 0 or step == steps:
            history.append(dict(step=step, reconstruction_mse=value, best_mse=best_loss))
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    final = convex_features(h, assignment, best_weights, clusters).float()
    return dict(x=final.cpu(), weights=best_weights.cpu(), best_step=best_step,
                history=pd.DataFrame(history))


def reconstruct_raw_partition(model, raw, state, steps=1000, lr=.05):
    assignment = state['assignment'].to(raw.device)
    targets = state['metric_centers'].to(raw.device)
    initial = geometric_medians(raw.double(), assignment, state['nodes']).float()
    fitted = fit_convex_representatives(model, raw, assignment, targets, initial, steps, lr, progress=False)
    with torch.no_grad():
        before = float((identity_hidden(model, initial) - targets).square().sum(1).mean())
        after = float((identity_hidden(model, fitted['x'].to(raw.device)) - targets).square().sum(1).mean())
    return dict(x=fitted['x'], initial_raw_x=initial.cpu(), convex_weights=fitted['weights'],
                reconstruction_initial=before, reconstruction_final=after,
                reconstruction_best_step=fitted['best_step']), fitted['history']


def run_convex_representative_study(source_dir, teacher_run, output_dir, steps=1000, lr=.05,
                                    seeds=tuple(range(100, 110)), data_dir='/content/data/', device='cuda',
                                    feature_source='s2x', candidate_mode='cluster', candidate_k=256):
    if candidate_mode not in ('cluster', 'nearest_size', 'nearest_k'):
        raise ValueError('Require cluster, nearest_size or nearest_k candidates')
    if candidate_mode != 'cluster' and feature_source != 'raw':
        raise ValueError('Nearest-candidate comparison uses raw X mixtures')
    if feature_source not in ('s2x', 'raw'):
        raise ValueError('Require s2x or raw mixture features')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Require distinct nonempty student seeds')
    source = Path(source_dir)
    protocol = json.loads((source / 'protocol.json').read_text())
    if protocol['settings']['loss_weighting'] != 'uniform':
        raise ValueError('This paired experiment requires the uniform-CE source protocol')
    params = json.loads((source / 'hidden' / 'best.json').read_text())['params']
    key = _fingerprint(dict(mode='hidden', T=params['T'], kl_weight=params['kl_weight']))
    state = torch.load(source / f'partition_{key}.pt', map_location='cpu', weights_only=True)
    dataset = protocol['dataset']
    graph = get_dataset(SimpleNamespace(dataset_name=dataset, raw_data_dir=str(data_dir).rstrip('/') + '/'))
    digest = array_digest(graph.x.numpy(), graph.edge_index.numpy(), graph.y.numpy(),
                         graph.train_mask.numpy(), graph.val_mask.numpy(), graph.test_mask.numpy())
    if digest != protocol['graph']:
        raise ValueError('Data differs from the saved partition experiment')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(Path(teacher_run) / 'teacher_predictions.npz') as saved:
        expected = saved['probabilities'].copy()
    model, recovery = recover_teacher(teacher_run, protocol['teacher'], graph.x.to(device),
                                      graph.edge_index.to(device), graph.y.numpy(), graph.train_mask.numpy(),
                                      graph.val_mask.numpy(), expected, device=device)
    if array_digest(*[p.cpu().numpy() for p in model.state_dict().values()]) != protocol['teacher_state']:
        raise ValueError('Teacher weights differ from the partition teacher')
    train, _, validation, testing, h = _prepare_dataset(dataset, data_dir, device)
    targets, baseline = state['metric_centers'].to(device), state['x'].to(device)
    groups = state['assignment'].to(device)
    nodes = torch.arange(len(h), device=device)
    if feature_source == 'raw':
        h = train['x']
        if candidate_mode != 'cluster':
            from src.centroid_candidates import nearest_candidates
            from src.ntk_readout_study import readout_features

            features, _, _, _ = readout_features(model, graph.x.to(device), graph.edge_index.to(device), 'gcn')
            sizes = state['counts'] if candidate_mode == 'nearest_size' else [candidate_k] * state['nodes']
            nodes, groups = nearest_candidates(features, targets, sizes)
            h = h[nodes]
        baseline = geometric_medians(h.double(), groups, state['nodes']).float()
    from src.centroid_candidates import candidate_statistics

    candidate_info = candidate_statistics(nodes, groups, state['assignment'].to(device))
    config = dict(version=1, source=str(source), source_protocol=protocol, params=params,
                  partition=array_digest(state['assignment'].numpy(), state['x'].numpy(),
                                         state['y'].numpy(), state['metric_centers'].numpy()),
                  steps=steps, lr=lr, seeds=list(seeds), objective='uniform_hidden_squared_error',
                  initialization='geometric_median_coefficients', torch=str(torch.__version__), device=str(device))
    if feature_source != 's2x':
        config['feature_source'] = feature_source
    if candidate_mode != 'cluster':
        config['candidates'] = dict(mode=candidate_mode, k=candidate_k if candidate_mode == 'nearest_k' else None,
                                    ids=array_digest(nodes.cpu().numpy(), groups.cpu().numpy()), metric='teacher_hidden_euclidean')
    folder = Path(output_dir) / _fingerprint(config)
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(folder / 'protocol.json', config)
    _write_json(folder / 'teacher_recovery.json', recovery)
    _write_json(folder / 'candidates.json', candidate_info)
    path = folder / 'convex.pt'
    if path.exists():
        fitted = torch.load(path, map_location='cpu', weights_only=True)
    else:
        fitted = fit_convex_representatives(model, h, groups, targets, baseline, steps, lr)
        if candidate_mode != 'cluster':
            fitted.update(candidate_nodes=nodes.cpu(), candidate_groups=groups.cpu())
        fitted.pop('history').to_csv(folder / 'reconstruction_history.csv', index=False)
        temporary = path.with_suffix('.tmp')
        torch.save(fitted, temporary)
        temporary.replace(path)
    representatives = dict(median=baseline, convex=fitted['x'].to(device))
    summaries, detail, errors = [], [], []
    for method, cx in representatives.items():
        with torch.no_grad():
            per_cluster = (identity_hidden(model, cx) - targets).square().sum(1).cpu().numpy()
        errors.extend(dict(method=method, cluster=i, squared_error=float(e), count=int(state['counts'][i]))
                      for i, e in enumerate(per_cluster))
        rows = []
        for seed in tqdm(seeds, desc=f'{method}: paired students'):
            result_path = folder / f'{method}_seed{seed}.json'
            if result_path.exists():
                record = json.loads(result_path.read_text())
            else:
                val, test, epoch = _train_student(cx, state['y'].to(device), validation, params, seed,
                                                  protocol['settings'], testing=testing)
                record = dict(method=method, seed=seed, validation=100 * val, test=100 * test, epoch=epoch)
                _write_json(result_path, record)
            rows.append(record)
        table = pd.DataFrame(rows)
        detail.extend(rows)
        summaries.append(dict(method=method, feature_source=feature_source, candidate_mode=candidate_mode,
                              candidate_k=candidate_k if candidate_mode == 'nearest_k' else None,
                              **candidate_info, dataset=dataset, ratio=protocol['ratio'], nodes=len(cx),
                              reconstruction_mse=float(per_cluster.mean()), final_val=table.validation.mean(),
                              final_val_std=table.validation.std(ddof=0), test_mean=table.test.mean(),
                              test_std=table.test.std(ddof=0)))
    summary, repeats, cluster_errors = pd.DataFrame(summaries), pd.DataFrame(detail), pd.DataFrame(errors)
    paired = repeats.pivot(index='seed', columns='method', values=['validation', 'test'])
    delta = pd.DataFrame({metric + '_delta': paired[metric]['convex'] - paired[metric]['median']
                          for metric in ('validation', 'test')}).reset_index()
    for name, table in [('summary', summary), ('final_seeds', repeats), ('paired_deltas', delta),
                         ('cluster_errors', cluster_errors)]:
        table.to_csv(folder / f'{name}.csv', index=False)
    weights = fitted['weights']
    sums = torch.zeros(len(targets), dtype=weights.dtype).index_add_(0, groups.cpu(), weights)
    diagnostics = dict(best_step=fitted['best_step'], min_weight=float(weights.min()),
                       coefficient_sum_error=float((sums - 1).abs().max()))
    _write_json(folder / 'diagnostics.json', diagnostics)
    return dict(summary=summary, paired=delta, errors=cluster_errors, diagnostics=diagnostics,
                history=pd.read_csv(folder / 'reconstruction_history.csv'), folder=str(folder))


def plot_convex_representatives(report):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    history = report['history']
    axes[0].plot(history.step, history.reconstruction_mse, label='Current')
    axes[0].plot(history.step, history.best_mse, label='Best')
    axes[0].set(xlabel='Optimization step', ylabel='Mean squared hidden distance', title='Reconstruction')
    axes[0].legend()
    for ax, metric, title in zip(axes[1:], ('validation_delta', 'test_delta'), ('Validation', 'Test')):
        ax.scatter(report['paired'].seed, report['paired'][metric])
        ax.axhline(0, color='gray', linestyle='--')
        ax.set(xlabel='Student seed', ylabel='Convex minus median (pp)', title=title)
    fig.savefig(Path(report['folder']) / 'comparison.png', dpi=180)
    return fig
