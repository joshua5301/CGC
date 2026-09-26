import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from torch.func import functional_call
from tqdm.auto import tqdm

from src.gnn_distance_probe import ProbeGNN
from src.local_ce_distance import run_local_ce_comparison, saved_student_fingerprints
from src.node_distances import array_digest
from src.tree_distance import _neighbors, _write_json


def _save(path, **values):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def gram_distance(gram):
    gram = np.asarray(gram, dtype=np.float64)
    squared = gram.diagonal()[:, None] + gram.diagonal()[None, :] - 2 * gram
    distance = np.sqrt(np.maximum((squared + squared.T) / 2, 0))
    np.fill_diagonal(distance, 0)
    return distance


def make_network(x, edges, architecture, width, outputs, seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = ProbeGNN(architecture, x.shape[1], width, outputs, dropout=0.).to(x.device).eval()
        with torch.no_grad():
            model(x, edges)
    return model


def sketch_grams(model, x, edges, ids, projections, seed, callback=None, exclude_names=()):
    names, parameters = zip(*model.named_parameters())
    generator = torch.Generator(device=x.device).manual_seed(seed)

    def output(*values):
        return functional_call(model, dict(zip(names, values)), (x, edges))[ids]

    blocks, result = [], {}
    for count in tqdm(range(1, max(projections) + 1), desc='Jacobian projections', leave=False):
        direction = tuple(torch.randint(0, 2, p.shape, device=p.device, generator=generator).to(p.dtype) * 2 - 1
                          for p in parameters)
        direction = tuple(torch.zeros_like(v) if name in exclude_names else v
                          for name, v in zip(names, direction))
        _, product = torch.autograd.functional.jvp(output, parameters, direction, create_graph=False)
        blocks.append(product.detach())
        if count in projections:
            features = torch.cat(blocks, dim=1).double() / np.sqrt(count * blocks[0].shape[1])
            gram = (features @ features.T).cpu().numpy()
            result[count] = gram
            if callback is not None:
                callback(count, gram)
    return result


def exact_trace_gram(model, x, edges, ids):
    parameters = tuple(model.parameters())
    outputs = model(x, edges)[ids]
    gram = torch.zeros((len(ids), len(ids)), device=x.device, dtype=torch.float64)
    for channel in tqdm(range(outputs.shape[1]), desc='Exact Jacobian audit', leave=False):
        rows = []
        for node in range(len(ids)):
            gradients = torch.autograd.grad(outputs[node, channel], parameters, retain_graph=True, allow_unused=True)
            rows.append(torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                                   for p, g in zip(parameters, gradients)]))
        jacobian = torch.stack(rows).double()
        gram += jacobian @ jacobian.T / outputs.shape[1]
    return gram.cpu().numpy()


def distance_agreement(distance, reference, k=10):
    pairs = np.triu_indices(len(distance), 1)
    a, b = distance[pairs], reference[pairs]
    denominator = np.linalg.norm(b)
    relative_error = float(np.linalg.norm(a - b) / denominator) if denominator > 0 else np.nan
    correlation = float(spearmanr(a, b).statistic) if np.ptp(a) > 0 and np.ptp(b) > 0 else np.nan
    count = min(k, len(distance) - 1)
    left, right = distance.copy(), reference.copy()
    np.fill_diagonal(left, np.inf)
    np.fill_diagonal(right, np.inf)
    left = np.argsort(left, axis=1, kind='stable')[:, :count]
    right = np.argsort(right, axis=1, kind='stable')[:, :count]
    overlap = (left[:, :, None] == right[:, None, :]).any(axis=2).mean()
    return dict(distance_relative_error=relative_error, distance_spearman=correlation,
                neighbor_overlap=float(overlap), k=count)


def run_empirical_ntk_study(previous_local_dir, x, edge_index,
                            architectures=('gcn', 'sage', 'gin'), widths=(64, 256),
                            network_seeds=tuple(range(5000, 5008)), ensemble_sizes=(1, 4, 8),
                            projections=(64, 256, 512), sketch_seeds=(6000, 7000),
                            outputs=16, audit_nodes=16, audit_seed=9000, device='cuda'):
    widths, ensemble_sizes, projections = [sorted(set(values)) for values in (widths, ensemble_sizes, projections)]
    if (not widths or not ensemble_sizes or not projections or min(widths + ensemble_sizes + projections) < 1
            or max(ensemble_sizes) > len(network_seeds) or outputs < 1 or audit_nodes < 2
            or len(sketch_seeds) != 2 or len(set(sketch_seeds)) != 2
            or len(set(network_seeds)) != len(network_seeds)
            or not architectures or set(architectures) - {'gcn', 'sage', 'gin'}):
        raise ValueError('Require positive sizes, distinct seeds, two sketch replicas and valid architectures')
    previous_local_dir = Path(previous_local_dir)
    previous = json.loads((previous_local_dir / 'protocol.json').read_text())
    result_dir = Path(previous['source_result'])
    source = json.loads((result_dir / 'protocol.json').read_text())
    models, fingerprints = saved_student_fingerprints(source, previous['models'])
    if fingerprints != previous['source_logits_sha256']:
        raise ValueError('Frozen student outputs differ from the previous experiment')
    student_seeds = {int(Path(name).stem.split('_')[-1][4:]) for name in fingerprints}
    if student_seeds & set(network_seeds):
        raise ValueError('Distance networks and evaluated students must have different initialization seeds')
    ids = np.load(Path(source['source_dir']) / 'probe_ids.npy')
    graph = source['source_protocol']['distance_protocol']
    _, canonical = _neighbors(edge_index, len(x), graph['self_loops'])
    digest = hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes() + canonical.tobytes()).hexdigest()
    if digest != graph['input_sha256']:
        raise ValueError('Original graph/features changed')
    position = {int(node): i for i, node in enumerate(ids)}
    evaluation = np.array([position[node] for node in previous['evaluation_nodes']])
    if audit_nodes > len(evaluation):
        raise ValueError('Audit subset exceeds evaluation nodes')
    audit = np.random.default_rng(audit_seed).choice(evaluation, audit_nodes, replace=False)
    config = dict(version=1, input_sha256=array_digest(x, edge_index, ids),
                  architectures=list(architectures), widths=widths, network_seeds=list(network_seeds),
                  ensemble_sizes=ensemble_sizes, projections=projections, sketch_seeds=list(sketch_seeds),
                  outputs=outputs, audit_positions=audit.tolist(), audit_seed=audit_seed,
                  device=str(device), torch=str(torch.__version__), pyg=str(torch_geometric.__version__),
                  dtype='float32_derivatives_float64_grams', parameterization='native_PyG_defaults',
                  kernel='output_channel_averaged_trace_NTK', tf32=False)
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    folder = result_dir / 'empirical_ntk_study' / key
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(folder / 'protocol.json', config)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tx = torch.as_tensor(x, dtype=torch.float32, device=device)
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    queries = torch.as_tensor(ids, dtype=torch.long, device=device)
    distances, metadata, audits, stability, ensembles = {}, [], [], [], []
    with np.load(previous_local_dir / 'distances.npz') as saved:
        for name, digest in previous.get('extra_sha256', {}).items():
            if name.startswith('deep_'):
                continue
            distance = saved[name].copy()
            if array_digest(distance) != digest:
                raise ValueError(f'Previous distance changed: {name}')
            distances[name] = distance
    started = time.perf_counter()
    for architecture in architectures:
        for width in widths:
            sums = {(replica, r): np.zeros((len(ids), len(ids))) for replica in sketch_seeds for r in projections}
            forward_blocks = []
            half = max(ensemble_sizes) // 2
            first_half = None
            for index, seed in enumerate(tqdm(network_seeds[:max(ensemble_sizes)], desc=f'{architecture} width {width}')):
                stem = f'{architecture}_w{width}_seed{seed}'
                model = None

                def network():
                    nonlocal model
                    if model is None:
                        model = make_network(tx, edges, architecture, width, outputs, seed)
                    return model

                output_path = folder / f'{stem}_outputs.npz'
                if not output_path.exists():
                    with torch.no_grad():
                        forward = network()(tx, edges)[queries].cpu().numpy()
                    _save(output_path, output=forward)
                with np.load(output_path) as saved:
                    forward_blocks.append(saved['output'].copy())
                for replica in sketch_seeds:
                    paths = {r: folder / f'{stem}_sketch{replica}_r{r}.npz' for r in projections}
                    if not all(path.exists() for path in paths.values()):
                        def checkpoint(r, gram):
                            if not np.isfinite(gram).all():
                                raise FloatingPointError('Nonfinite Jacobian sketch')
                            _save(paths[r], gram=gram)
                        sketch_grams(network(), tx, edges, queries, projections, replica + 100003 * seed, checkpoint)
                    for r, path in paths.items():
                        with np.load(path) as saved:
                            sums[replica, r] += saved['gram']
                if index == 0:
                    path = folder / f'{stem}_exact_audit.npz'
                    if not path.exists():
                        _save(path, gram=exact_trace_gram(network(), tx, edges, queries[audit]))
                    with np.load(path) as saved:
                        exact = gram_distance(saved['gram'])
                    for replica in sketch_seeds:
                        for r in projections:
                            approximate = gram_distance(sums[replica, r][np.ix_(audit, audit)])
                            audits.append(dict(architecture=architecture, width=width, network_seed=seed,
                                               projections=r, sketch_seed=replica,
                                               **distance_agreement(approximate, exact, k=min(5, audit_nodes - 1))))
                count = index + 1
                if count == half and max(ensemble_sizes) % 2 == 0:
                    first_half = sum(sums[replica, max(projections)] for replica in sketch_seeds) / len(sketch_seeds)
                if count in ensemble_sizes:
                    base = dict(architecture=architecture, width=width, ensembles=count, outputs=outputs)
                    name = f'deep_rf_{architecture}_w{width}_m{count}'
                    forward = np.concatenate(forward_blocks, axis=1).astype(np.float64) / np.sqrt(count * outputs)
                    distances[name] = cdist(forward, forward)
                    metadata.append(dict(method=name, family='rf', projections=0, sketch_seed=-1, **base))
                    for replica in sketch_seeds:
                        for r in projections:
                            name = f'deep_ntk_{architecture}_w{width}_m{count}_r{r}_s{replica}'
                            distances[name] = gram_distance(sums[replica, r] / count)
                            metadata.append(dict(method=name, family='ntk', projections=r, sketch_seed=replica, **base))
                del model
            if first_half is not None:
                total = sum(sums[replica, max(projections)] for replica in sketch_seeds) / len(sketch_seeds)
                pair = [gram_distance(first_half / half), gram_distance((total - first_half) / half)]
                forward_pair = [cdist(block, block) for block in
                                (np.concatenate(forward_blocks[:half], axis=1).astype(np.float64) / np.sqrt(half * outputs),
                                 np.concatenate(forward_blocks[half:], axis=1).astype(np.float64) / np.sqrt(half * outputs))]
                for family, values in (('ntk', pair), ('rf', forward_pair)):
                    ensembles.append(dict(architecture=architecture, width=width, family=family,
                                          networks_per_half=half,
                                          **distance_agreement(values[0][np.ix_(evaluation, evaluation)],
                                                               values[1][np.ix_(evaluation, evaluation)])))
    metadata = pd.DataFrame(metadata)
    for row in metadata.itertuples():
        if row.family == 'ntk':
            other = next(seed for seed in sketch_seeds if seed != row.sketch_seed)
            reference = f'deep_ntk_{row.architecture}_w{row.width}_m{row.ensembles}_r{max(projections)}_s{other}'
            kind = 'independent_sketch_same_networks'
        else:
            reference = f'deep_rf_{row.architecture}_w{row.width}_m{max(ensemble_sizes)}'
            kind = 'nested_network_ensemble'
        stability.append(dict(method=row.method, reference=reference, kind=kind,
                              **distance_agreement(distances[row.method][np.ix_(evaluation, evaluation)],
                                                   distances[reference][np.ix_(evaluation, evaluation)])))
        if row.family == 'ntk':
            reference = f'deep_ntk_{row.architecture}_w{row.width}_m{max(ensemble_sizes)}_r{row.projections}_s{row.sketch_seed}'
            stability.append(dict(method=row.method, reference=reference, kind='nested_network_ensemble',
                                  **distance_agreement(distances[row.method][np.ix_(evaluation, evaluation)],
                                                       distances[reference][np.ix_(evaluation, evaluation)])))
    metadata.to_csv(folder / 'metadata.csv', index=False)
    pd.DataFrame(audits).to_csv(folder / 'exact_audit.csv', index=False)
    pd.DataFrame(stability).to_csv(folder / 'stability.csv', index=False)
    pd.DataFrame(ensembles).to_csv(folder / 'ensemble_agreement.csv', index=False)
    _write_json(folder / 'timing.json', dict(preparation_seconds=time.perf_counter() - started))
    report = run_local_ce_comparison(
        result_dir, x, edge_index, previous['fractions'], previous['ks'],
        previous['calibration_fraction'], previous['split_seed'], models=models,
        extra_distances=distances, extra_protocol=dict(study=config, folder=str(folder), previous=previous))
    report.update(metadata=metadata, exact_audit=pd.DataFrame(audits), stability=pd.DataFrame(stability),
                  ensemble_agreement=pd.DataFrame(ensembles), study_folder=str(folder))
    for name in ('metadata', 'exact_audit', 'stability', 'ensemble_agreement'):
        report[name].to_csv(Path(report['folder']) / f'{name}.csv', index=False)
    report['detailed_summary'] = report['summary'].merge(metadata, on='method', how='inner')
    report['detailed_summary'].to_csv(Path(report['folder']) / 'detailed_summary.csv', index=False)
    runs = report['per_run'].merge(metadata, on='method', how='inner')
    keys = ['model', 'train_size', 'subset_seed', 'model_seed', 'selection', 'cutoff',
            'architecture', 'width', 'ensembles']
    metrics = ['relative_ce_mean', 'relative_ce_p95']
    forward = runs[runs.family == 'rf'][keys + metrics].rename(columns={m: f'rf_{m}' for m in metrics})
    paired = runs[runs.family == 'ntk'].merge(forward, on=keys, validate='many_to_one')
    for metric in metrics:
        paired[f'delta_{metric}'] = paired[metric] - paired[f'rf_{metric}']
    groups = [key for key in keys if key not in ('subset_seed', 'model_seed')] + ['projections', 'sketch_seed']
    report['paired_rf_summary'] = paired.groupby(groups, as_index=False).agg(
        delta_mean=('delta_relative_ce_mean', 'mean'), delta_std=('delta_relative_ce_mean', 'std'),
        delta_p95_mean=('delta_relative_ce_p95', 'mean'), runs=('delta_relative_ce_mean', 'count'))
    paired.to_csv(Path(report['folder']) / 'paired_rf_runs.csv', index=False)
    report['paired_rf_summary'].to_csv(Path(report['folder']) / 'paired_rf_summary.csv', index=False)
    return report


def plot_empirical_ntk_study(report, k=10):
    import matplotlib.pyplot as plt
    table = report['detailed_summary']
    matched = table[(table.model == table.architecture) & (table.selection == 'knn') & (table.cutoff == k)]
    architectures = list(report['metadata'].architecture.unique())
    figures = []
    for size in sorted(matched.train_size.unique()):
        fig, axes = plt.subplots(2, len(architectures), figsize=(5 * len(architectures), 8), squeeze=False,
                                 constrained_layout=True)
        for col, architecture in enumerate(architectures):
            part = matched[(matched.architecture == architecture) & (matched.train_size == size)]
            for row, metric in enumerate(('relative_ce_mean_mean', 'relative_ce_p95_mean')):
                ax = axes[row, col]
                for (width, count), group in part[part.family == 'ntk'].groupby(['width', 'ensembles']):
                    stats = group.groupby('projections')[metric].agg(['mean', 'min', 'max'])
                    line, = ax.plot(stats.index, stats['mean'], 'o-', label=f'NTK w{width} m{count}')
                    ax.fill_between(stats.index, stats['min'], stats['max'], color=line.get_color(), alpha=.12)
                for width, group in part[part.family == 'rf'].groupby('width'):
                    best_count = group.ensembles.max()
                    value = group[group.ensembles == best_count][metric].iloc[0]
                    ax.axhline(value, linestyle='--', label=f'RF w{width} m{best_count}')
                baseline = report['summary'].query('model == @architecture and train_size == @size and method == "grip_S2X" and selection == "knn" and cutoff == @k')
                if len(baseline):
                    ax.axhline(baseline[metric].iloc[0], color='black', linestyle=':', label='S²X')
                ax.set(title=f'{architecture.upper()} | {metric}', xlabel='Jacobian projections', xscale='log')
                ax.legend(fontsize=7)
        fig.suptitle(f'Matched architecture | n={size}, k={k} | bands: two sketch replicas, not CI')
        fig.savefig(Path(report['folder']) / f'ntk_ce_n{size}_k{k}.png', dpi=160)
        figures.append(fig)
    fig, axes = plt.subplots(2, len(architectures), figsize=(5 * len(architectures), 8), squeeze=False,
                             constrained_layout=True)
    for col, architecture in enumerate(architectures):
        audit = report['exact_audit'].query('architecture == @architecture')
        for width, group in audit.groupby('width'):
            for row, metric in enumerate(('distance_relative_error', 'neighbor_overlap')):
                stats = group.groupby('projections')[metric].agg(['mean', 'min', 'max'])
                axes[row, col].plot(stats.index, stats['mean'], 'o-', label=f'width {width}')
                axes[row, col].fill_between(stats.index, stats['min'], stats['max'], alpha=.15)
                axes[row, col].set(title=f'{architecture.upper()} | {metric}', xlabel='Projections', xscale='log')
                axes[row, col].legend()
    fig.suptitle('Small-node exact Jacobian audit | first network seed | no projection in reference')
    fig.savefig(Path(report['folder']) / 'ntk_exact_audit.png', dpi=160)
    figures.append(fig)
    return figures
