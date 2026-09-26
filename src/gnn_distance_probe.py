import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from scipy.spatial.distance import cdist
from torch import nn
from torch_geometric import seed_everything
from torch_geometric.nn import GCNConv, GINConv, SAGEConv
from tqdm.auto import tqdm

from src.tree_distance import _neighbors
from src.tree_distance_analysis import load_distance_blocks
from src.tree_distance_factorial import mean_depth_one, normalize_rows


class ProbeGNN(nn.Module):
    def __init__(self, model, features, hidden, classes, dropout=.5):
        super().__init__()
        self.dropout = dropout
        layers = []
        for inputs, outputs in ((features, hidden), (hidden, classes)):
            if model == 'gcn':
                layer = GCNConv(inputs, outputs, cached=True)
            elif model == 'sage':
                layer = SAGEConv(inputs, outputs, aggr='mean')
            elif model == 'gin':
                layer = GINConv(nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(),
                                             nn.Linear(hidden, outputs)), train_eps=True)
            else:
                raise ValueError('Require gcn, sage or gin')
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edges):
        x = self.layers[0](x, edges)
        x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return self.layers[1](x, edges)


def fit_probe(x, edges, train_ids, train_labels, probe_ids, model, classes, seed,
              hidden=128, dropout=.5, epochs=300, lr=.01, weight_decay=5e-4):
    seed_everything(seed)
    network = ProbeGNN(model, x.shape[1], hidden, classes, dropout).to(x.device)
    ids = torch.as_tensor(train_ids, device=x.device, dtype=torch.long)
    labels = torch.as_tensor(train_labels, device=x.device,
                             dtype=x.dtype if np.asarray(train_labels).ndim == 2 else torch.long)
    queries = torch.as_tensor(probe_ids, device=x.device, dtype=torch.long)
    network.eval()
    with torch.no_grad():
        initial = network(x, edges)[queries].cpu().numpy()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        network.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(network(x, edges)[ids], labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite training loss for {model}')
        loss.backward()
        optimizer.step()
    network.eval()
    with torch.no_grad():
        output = network(x, edges)
        trained = output[queries].cpu().numpy()
        targets = labels.argmax(1) if labels.ndim == 2 else labels
        train_accuracy = float((output[ids].argmax(1) == targets).float().mean())
        train_ce = float(F.cross_entropy(output[ids], labels))
    if not np.isfinite(initial).all() or not np.isfinite(trained).all():
        raise FloatingPointError('Nonfinite probe logits')
    return dict(initial=initial, trained=trained, train_accuracy=train_accuracy, train_ce=train_ce)


def distance_geometry(matrices):
    geometry, scales = {}, []
    for method, matrix in matrices.items():
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 2:
            raise ValueError('Require square distances for at least two probe nodes')
        if not np.isfinite(matrix).all() or (matrix < 0).any() or not np.allclose(matrix, matrix.T):
            raise ValueError('Require finite nonnegative symmetric distances')
        pairs = np.triu_indices(len(matrix), 1)
        values = matrix[pairs]
        scale = float(np.median(values))
        if scale <= 0:
            scale = float(np.median(values[values > 0])) if (values > 0).any() else 1.
        ranked = matrix.copy()
        np.fill_diagonal(ranked, np.inf)
        geometry[method] = dict(distance=values / scale, pairs=pairs,
                                neighbors=np.argsort(ranked, axis=1, kind='stable')[:, :-1])
        scales.append(dict(method=method, distance_scale=scale,
                           zero_distance_fraction=float((values == 0).mean()),
                           normalized_distance_p95=float(np.quantile(values / scale, .95))))
    return geometry, pd.DataFrame(scales)


def output_stability(logits, geometry, ks=(1, 5, 10, 20)):
    logits = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValueError('Require finite logits')
    centered = logits - logits.mean(1, keepdims=True)
    probability = np.exp(logits - logits.max(1, keepdims=True))
    probability /= probability.sum(1, keepdims=True)
    gaps = dict(centered_logits=cdist(centered, centered),
                probability_tv=.5 * cdist(probability, probability, metric='cityblock'))
    local, tails, health = [], [], []
    for metric, gap in gaps.items():
        pair_indices = np.triu_indices(len(logits), 1)
        values = gap[pair_indices]
        positive = values[values > 0]
        scale = float(np.median(positive)) if len(positive) else 0.
        degenerate = float(values.max()) <= 1e-12
        normalized = gap / scale if not degenerate else np.full_like(gap, np.nan)
        health.append(dict(metric=metric, output_scale=scale, degenerate=degenerate,
                           random_gap_mean=float(values.mean()), random_gap_p95=float(np.quantile(values, .95))))
        for method, item in geometry.items():
            if item['neighbors'].shape[0] != len(logits):
                raise ValueError('Distance and output nodes do not match')
            distance = item['distance']
            output_gap = normalized[item['pairs']]
            nonzero = distance > 0
            ratios = output_gap[nonzero] / distance[nonzero]
            violations = (distance == 0) & (output_gap > 1e-6)
            tails.append(dict(method=method, metric=metric,
                              ratio_p95=float(np.quantile(ratios, .95)) if len(ratios) else np.nan,
                              ratio_p99=float(np.quantile(ratios, .99)) if len(ratios) else np.nan,
                              ratio_max=float('inf') if violations.any() else (float(ratios.max()) if len(ratios) else np.nan),
                              zero_distance_pairs=int((distance == 0).sum()),
                              zero_distance_violations=int(violations.sum())))
            for k in ks:
                if int(k) != k or not 1 <= k < len(logits):
                    raise ValueError('Require 1 <= k < number of probe nodes')
                columns = item['neighbors'][:, :k]
                rows = np.arange(len(logits))[:, None]
                actual = gap[rows, columns]
                selected = normalized[rows, columns]
                local.append(dict(method=method, metric=metric, k=k,
                                  gap_mean=float(actual.mean()), gap_p95=float(np.quantile(actual, .95)),
                                  normalized_gap_mean=float(selected.mean()),
                                  normalized_gap_p95=float(np.quantile(selected, .95)),
                                  relative_gap=float(actual.mean() / values.mean()) if not degenerate else np.nan))
    return dict(local=pd.DataFrame(local), tails=pd.DataFrame(tails), health=pd.DataFrame(health))


def _probe_distances(x, edges, h, probe_ids, raw_dir, normalized_dir, depths):
    protocol, raw = load_distance_blocks(x, edges, raw_dir, probe_ids, probe_ids, depths)
    normalized_protocol, normalized = load_distance_blocks(normalize_rows(x), edges, normalized_dir, probe_ids, probe_ids, [1])
    if any(protocol[key] != normalized_protocol[key] for key in ('weight', 'self_loops', 'dtype')):
        raise ValueError('Raw and normalized tree caches must use the same weight, loops and dtype')
    neighbors, _ = _neighbors(edges, len(x), protocol['self_loops'])
    degrees = np.array([len(v) for v in neighbors])[probe_ids]
    matrices = dict(raw_euclidean=raw['raw'], l2_euclidean=normalized['raw'],
                    grip_S2X=cdist(h[probe_ids], h[probe_ids]),
                    l2_tree_1_sum=normalized['tree_1'],
                    l2_tree_1_mean=mean_depth_one(normalized['raw'], normalized['tree_1'], degrees, degrees))
    for depth in depths:
        matrices[f'raw_tree_{depth}_sum'] = raw[f'tree_{depth}']
    if 1 in depths:
        matrices['raw_tree_1_mean'] = mean_depth_one(raw['raw'], raw['tree_1'], degrees, degrees)
    return matrices, protocol


def run_gnn_distance_probe(x, edge_index, y, train_mask, propagated_features, raw_distance_dir,
                           normalized_distance_dir, output_dir, train_sizes=(70,),
                           models=('gcn', 'sage', 'gin'), subset_seeds=(0, 1, 2, 3, 4),
                           model_seeds=(100, 101), probe_nodes=512, probe_seed=2026,
                           depths=(1, 2, 3), ks=(1, 5, 10, 20), hidden=128, dropout=.5,
                           epochs=300, lr=.01, weight_decay=5e-4, device='cuda',
                           pseudo_labels=None, teacher_config=None):
    x, edges, y = np.asarray(x), np.asarray(edge_index), np.asarray(y)
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_mask.shape != (len(x),):
        raise ValueError('Training mask shape mismatch')
    pool, unlabeled = np.flatnonzero(train_mask), np.flatnonzero(~train_mask)
    if pseudo_labels is not None:
        pseudo_labels = np.asarray(pseudo_labels, dtype=np.float32)
        if pseudo_labels.ndim != 2 or len(pseudo_labels) != len(x) or not np.isfinite(pseudo_labels).all() or (pseudo_labels < 0).any() or not np.allclose(pseudo_labels.sum(1), 1):
            raise ValueError('Require one probability vector per graph node')
        pool = np.arange(len(x))
    if not train_sizes or any(int(n) != n or n < 1 or n > len(pool) for n in train_sizes):
        raise ValueError('Training sizes must be between 1 and the original training-pool size')
    if not 2 <= probe_nodes <= len(unlabeled) or epochs < 1 or hidden < 1 or not 0 <= dropout < 1 or lr <= 0 or weight_decay < 0:
        raise ValueError('Invalid probe size or training settings')
    if not models or set(models) - {'gcn', 'sage', 'gin'} or not subset_seeds or not model_seeds:
        raise ValueError('Require gcn/sage/gin and nonempty seed lists')
    if not ks or any(int(k) != k or not 1 <= k < probe_nodes for k in ks):
        raise ValueError('Invalid neighborhood k')
    classes = np.unique(y[pool]) if pseudo_labels is None else np.arange(pseudo_labels.shape[1])
    pool_labels = np.searchsorted(classes, y[pool]) if pseudo_labels is None else pseudo_labels
    probe_ids = np.sort(np.random.default_rng(probe_seed).choice(unlabeled, probe_nodes, replace=False))
    h = np.asarray(propagated_features)
    if h.ndim != 2 or len(h) != len(x) or not np.isfinite(h).all():
        raise ValueError('Invalid GRIP baseline features')
    matrices, distance_protocol = _probe_distances(x, edges, h, probe_ids, raw_distance_dir,
                                                   normalized_distance_dir, depths)
    geometry, scales = distance_geometry(matrices)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    digest = hashlib.sha256()
    for value in (pool, pool_labels, probe_ids, h):
        digest.update(np.ascontiguousarray(value).tobytes())
    config = dict(version=1, revision=revision, torch=str(torch.__version__),
                  torch_geometric=str(torch_geometric.__version__), device=str(device), distance_protocol=distance_protocol,
                  input_sha256=digest.hexdigest(), train_sizes=list(train_sizes), models=list(models),
                  subset_seeds=list(subset_seeds), model_seeds=list(model_seeds), probe_nodes=probe_nodes,
                  probe_seed=probe_seed, depths=list(depths), ks=list(ks), hidden=hidden, dropout=dropout,
                  epochs=epochs, lr=lr, weight_decay=weight_decay, layers=2,
                  graph='full_original_graph', supervision='uniform_ce_on_sampled_training_nodes',
                  checkpoint='fixed_final_epoch_without_validation', normalization='fixed_pair_median',
                  probe_pool='outside_entire_original_training_pool', labels_outside_training_used=False)
    config.update(sampling_pool='all_nodes' if pseudo_labels is not None else 'original_training_nodes',
                  target_kind='teacher_soft_labels' if pseudo_labels is not None else 'ground_truth',
                  teacher=teacher_config)
    if pseudo_labels is not None:
        config['supervision'] = 'uniform_soft_ce_on_sampled_graph_nodes'
        config['labels_outside_training_used'] = bool((teacher_config or {}).get('validation_selection', False))
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    output = Path(output_dir) / key
    output.mkdir(parents=True, exist_ok=True)
    (output / 'protocol.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    np.save(output / 'probe_ids.npy', probe_ids)
    scales.to_csv(output / 'distance_scales.csv', index=False)
    settings = dict(hidden=hidden, dropout=dropout, epochs=epochs, lr=lr, weight_decay=weight_decay)
    tx = torch.as_tensor(x, device=device, dtype=torch.float32)
    te = torch.as_tensor(edges, device=device, dtype=torch.long)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tables = {name: [] for name in ('local', 'tails', 'health')}
    runs = []
    seen_initial = set()
    total = len(train_sizes) * len(subset_seeds) * len(models) * len(model_seeds)
    with tqdm(total=total, desc='GNN output preservation') as progress:
        for size in train_sizes:
            for subset_seed in subset_seeds:
                selected = np.random.default_rng(subset_seed).permutation(len(pool))[:size]
                selected = selected[np.argsort(pool[selected])]
                train_ids, labels = pool[selected], pool_labels[selected]
                hard_labels = labels.argmax(1) if labels.ndim == 2 else labels
                for model in models:
                    for model_seed in model_seeds:
                        metadata = dict(train_size=size, subset_seed=subset_seed, model=model, model_seed=model_seed)
                        path = output / f'{model}_n{size}_subset{subset_seed}_seed{model_seed}.npz'
                        if path.exists():
                            with np.load(path) as saved:
                                result = {name: saved[name].copy() for name in ('initial', 'trained', 'train_accuracy', 'train_ce')}
                        else:
                            result = fit_probe(tx, te, train_ids, labels, probe_ids, model, len(classes), model_seed, **settings)
                            temporary = path.with_suffix('.tmp')
                            with temporary.open('wb') as handle:
                                np.savez_compressed(handle, **result, train_ids=train_ids,
                                                    class_counts=np.bincount(hard_labels, minlength=len(classes)))
                            temporary.replace(path)
                        runs.append(dict(**metadata, train_accuracy=100 * float(result['train_accuracy']),
                                         train_ce=float(result['train_ce']),
                                         represented_classes=int(len(np.unique(hard_labels))),
                                         probe_training_overlap=int(np.isin(probe_ids, train_ids).sum())))
                        for phase in ('initial', 'trained'):
                            initial_key = (size, model, model_seed)
                            if phase == 'initial':
                                if initial_key in seen_initial:
                                    continue
                                seen_initial.add(initial_key)
                            evaluation = output_stability(result[phase], geometry, ks)
                            for name, table in evaluation.items():
                                phase_metadata = dict(metadata, subset_seed=-1) if phase == 'initial' else metadata
                                tables[name].append(table.assign(**phase_metadata, phase=phase))
                        progress.update()
    report = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    report['runs'] = pd.DataFrame(runs)
    groups = ['train_size', 'model', 'phase', 'method', 'metric', 'k']
    report['summary'] = report['local'].groupby(groups, as_index=False).agg(
        relative_gap_mean=('relative_gap', 'mean'), relative_gap_std=('relative_gap', 'std'),
        normalized_gap_p95_mean=('normalized_gap_p95', 'mean'),
        gap_mean=('gap_mean', 'mean'), gap_p95_mean=('gap_p95', 'mean'),
        runs=('relative_gap', 'count'))
    report['tail_summary'] = report['tails'].groupby(groups[:-1], as_index=False).agg(
        ratio_p99_mean=('ratio_p99', 'mean'),
        zero_distance_violations_mean=('zero_distance_violations', 'mean'))
    for name, table in report.items():
        table.to_csv(output / f'{name}.csv', index=False)
    report['folder'] = str(output)
    return report
