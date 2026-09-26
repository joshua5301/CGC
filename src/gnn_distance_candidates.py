import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from torch.func import functional_call
from tqdm.auto import tqdm

from src.gnn_distance_probe import ProbeGNN
from src.local_ce_distance import run_local_ce_comparison, saved_student_fingerprints
from src.node_distances import array_digest, neighborhood_mmd_distance, pair_scale, transition_neighbors
from src.tree_distance import _neighbors, _write_json


def projected_jacobian(function, parameters, projections=64, seed=4000, progress=None):
    generator = torch.Generator(device=parameters[0].device).manual_seed(seed)
    columns = []
    for _ in range(projections):
        directions = tuple(torch.randint(0, 2, p.shape, device=p.device, generator=generator).to(p.dtype) * 2 - 1
                           for p in parameters)
        _, product = torch.autograd.functional.jvp(function, parameters, directions, create_graph=False)
        columns.append(product.detach().cpu().numpy())
        if progress is not None:
            progress.update()
    return np.stack(columns, axis=1) / np.sqrt(projections)


def random_gnn_features(x, edge_index, ids, architecture, width, seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        network = ProbeGNN(architecture, x.shape[1], width, width, dropout=0.).to(x.device).eval()
        with torch.no_grad():
            return network(x, edge_index)[ids].cpu().numpy() / np.sqrt(width)


def empirical_ntk_features(x, edge_index, ids, architecture, width, seed, projections, projection_seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        network = ProbeGNN(architecture, x.shape[1], width, 1, dropout=0.).to(x.device).eval()
        with torch.no_grad():
            network(x, edge_index)
        names, parameters = zip(*network.named_parameters())

        def output(*values):
            return functional_call(network, dict(zip(names, values)), (x, edge_index))[ids, 0]

        with tqdm(total=projections, desc=f'NTK {architecture} seed {seed}', leave=False) as bar:
            return projected_jacobian(output, parameters, projections, projection_seed, bar)


def scattering_features(x, edge_index, ids, self_loops=False):
    x = np.asarray(x, dtype=np.float64)
    neighbors = transition_neighbors(edge_index, len(x), self_loops)
    degree = np.array([len(v) for v in neighbors])
    transition = csr_matrix((np.repeat(1. / degree, degree),
                             (np.repeat(np.arange(len(x)), degree), np.concatenate(neighbors))),
                            shape=(len(x), len(x)))
    first = transition @ x
    second = transition @ first
    blocks = [second[ids], (transition @ np.abs(x - first))[ids], np.abs(first - second)[ids]]
    scales = [pair_scale(cdist(block, block)) for block in blocks]
    return np.concatenate([block / scale for block, scale in zip(blocks, scales)], axis=1) / np.sqrt(3), scales


def _save(path, **values):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def build_candidate_distances(x, edge_index, ids, output_dir, base_settings,
                              architectures=('gcn', 'sage', 'gin'), rf_width=128,
                              rf_seeds=tuple(range(2000, 2008)), ntk_width=64,
                              ntk_seeds=(3000, 3001), ntk_projections=64,
                              projection_seed=4000, device='cuda'):
    if not architectures or set(architectures) - {'gcn', 'sage', 'gin'} or len(set(architectures)) != len(architectures):
        raise ValueError('Require distinct GCN, SAGE or GIN candidate architectures')
    if min(rf_width, ntk_width, ntk_projections) < 1 or not rf_seeds or not ntk_seeds:
        raise ValueError('Require positive widths/projections and nonempty initialization seeds')
    if len(set(rf_seeds)) != len(rf_seeds) or len(set(ntk_seeds)) != len(ntk_seeds):
        raise ValueError('Initialization seeds must be distinct within each family')
    x, edge_index, ids = np.asarray(x), np.asarray(edge_index), np.asarray(ids)
    config = dict(version=1, input_sha256=array_digest(x, edge_index, ids),
                  base_settings=base_settings, architectures=list(architectures),
                  rf_width=rf_width, rf_seeds=list(rf_seeds), ntk_width=ntk_width,
                  ntk_seeds=list(ntk_seeds), ntk_projections=ntk_projections,
                  projection_seed=projection_seed, device=str(device),
                  torch=str(torch.__version__), torch_geometric=str(torch_geometric.__version__),
                  tf32=False, neural_depth=2, ntk='finite_width_scalar_output_rademacher_jvp')
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    output = Path(output_dir) / key
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / 'protocol.json', config)
    tx = torch.as_tensor(x, device=device, dtype=torch.float32)
    edges = torch.as_tensor(edge_index, device=device, dtype=torch.long)
    queries = torch.as_tensor(ids, device=device, dtype=torch.long)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    matrices, timing = {}, []
    for family, seeds, width in (('rf', rf_seeds, rf_width), ('entk', ntk_seeds, ntk_width)):
        for architecture in tqdm(architectures, desc=f'{family.upper()} candidate distances'):
            blocks, seconds = [], 0.
            for seed in seeds:
                path = output / f'{family}_{architecture}_seed{seed}.npz'
                if path.exists():
                    with np.load(path) as saved:
                        block, elapsed = saved['embedding'].copy(), float(saved['seconds'])
                else:
                    if tx.is_cuda:
                        torch.cuda.synchronize(tx.device)
                    started = time.perf_counter()
                    if family == 'rf':
                        block = random_gnn_features(tx, edges, queries, architecture, width, seed)
                    else:
                        block = empirical_ntk_features(tx, edges, queries, architecture, width, seed,
                                                       ntk_projections, projection_seed + seed)
                    if tx.is_cuda:
                        torch.cuda.synchronize(tx.device)
                    elapsed = time.perf_counter() - started
                expected_width = width if family == 'rf' else ntk_projections
                if block.shape != (len(ids), expected_width) or not np.isfinite(block).all():
                    raise ValueError(f'Invalid candidate embedding: {path}')
                if not path.exists():
                    _save(path, embedding=block, seconds=elapsed)
                blocks.append(block)
                seconds += elapsed
            embedding = np.concatenate(blocks, axis=1) / np.sqrt(len(blocks))
            method = f'{family}_{architecture}'
            matrices[method] = cdist(embedding, embedding)
            timing.append(dict(method=method, seconds=seconds, embedding_width=embedding.shape[1]))
    for method in ('sum_mmd', 'scattering'):
        path = output / f'{method}.npz'
        if path.exists():
            with np.load(path) as saved:
                distance, elapsed = saved['distance'].copy(), float(saved['seconds'])
        else:
            started = time.perf_counter()
            if method == 'sum_mmd':
                distance, details = neighborhood_mmd_distance(
                    x, edge_index, ids, depth=base_settings['depth'], width=base_settings['rff_width'],
                    root_weight=base_settings['root_weight'], seed=base_settings['rff_seed'],
                    self_loops=base_settings['self_loops'], aggregation='sum')
            else:
                embedding, details = scattering_features(x, edge_index, ids, base_settings['self_loops'])
                distance = cdist(embedding, embedding)
            elapsed = time.perf_counter() - started
            _write_json(output / f'{method}_details.json', dict(details=details))
        if distance.shape != (len(ids), len(ids)) or not np.isfinite(distance).all():
            raise ValueError(f'Invalid candidate distance: {method}')
        if not path.exists():
            _save(path, distance=distance, seconds=elapsed)
        matrices[method] = distance
        timing.append(dict(method=method, seconds=elapsed, embedding_width=np.nan))
    _save(output / 'distances.npz', **matrices)
    pd.DataFrame(timing).to_csv(output / 'timing.csv', index=False)
    return matrices, dict(config=config, folder=str(output)), pd.DataFrame(timing)


def run_candidate_distance_study(result_dir, x, edge_index, models=('gcn', 'sage', 'gin'),
                                  fractions=(.01, .02, .05, .1), ks=(1, 5, 10, 20),
                                  calibration_fraction=.25, split_seed=2026, **candidate_options):
    result_dir = Path(result_dir)
    source = json.loads((result_dir / 'protocol.json').read_text())
    models, fingerprints = saved_student_fingerprints(source, models)
    graph = source['source_protocol']['distance_protocol']
    _, canonical = _neighbors(edge_index, len(x), graph['self_loops'])
    digest = hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes() + canonical.tobytes()).hexdigest()
    if digest != graph['input_sha256']:
        raise ValueError('Graph/features do not match the original student cache')
    student_seeds = {int(Path(name).stem.split('_')[-1][4:]) for name in fingerprints}
    feature_seeds = set(candidate_options.get('rf_seeds', range(2000, 2008))) | set(candidate_options.get('ntk_seeds', (3000, 3001)))
    if student_seeds & feature_seeds:
        raise ValueError('Distance initialization seeds must differ from evaluation student seeds')
    ids = np.load(Path(source['source_dir']) / 'probe_ids.npy')
    distances, protocol, timing = build_candidate_distances(
        x, edge_index, ids, result_dir / 'distance_candidates', source['distance_settings'], **candidate_options)
    report = run_local_ce_comparison(result_dir, x, edge_index, fractions, ks,
                                     calibration_fraction, split_seed, models=models,
                                     extra_distances=distances, extra_protocol=protocol)
    timing.to_csv(Path(report['folder']) / 'candidate_timing.csv', index=False)
    report['candidate_timing'] = timing
    return report


def plot_candidate_distances(report):
    import matplotlib.pyplot as plt
    labels = dict(grip_S2X='S²X', probability_ot='Probability OT', matched_mean='B²X',
                  multiscale='Multiscale', neighborhood_mmd='Mean MMD', sum_mmd='Sum MMD',
                  scattering='Scattering', gcn2_ntk='GCN2 analytic NTK', gcn2_nngp='GCN2 analytic NNGP',
                  **{f'{family}_{model}': f'{family.upper()} {model.upper()}'
                                              for family in ('rf', 'entk') for model in ('gcn', 'sage', 'gin')})
    methods, models = report['methods'], report['models']
    metrics = [('relative_ce_mean_mean', 'Mean CE change / all-pair mean'),
               ('relative_ce_p95_mean', 'P95 CE change / all-pair mean'),
               ('relative_fit_mae_mean', 'Calibrated MAE / all-pair mean')]
    figures = []
    for size in sorted(report['summary'].train_size.unique()):
        for selection in ('quantile', 'knn'):
            table = report['summary'].query('train_size == @size and selection == @selection')
            cutoffs = sorted(table.cutoff.unique())
            fig, axes = plt.subplots(3, len(models), figsize=(5.5 * len(models), 15), squeeze=False,
                                     constrained_layout=True)
            for row, (metric, title) in enumerate(metrics):
                finite = table[metric].to_numpy()
                finite = finite[np.isfinite(finite)]
                maximum = float(finite.max()) if len(finite) and finite.max() > 0 else 1.
                for col, model in enumerate(models):
                    values = table.query('model == @model').pivot(index='method', columns='cutoff', values=metric).reindex(index=methods, columns=cutoffs).to_numpy()
                    ax = axes[row, col]
                    display = ax.imshow(values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=maximum)
                    ax.set(xticks=np.arange(len(cutoffs)),
                           xticklabels=[f'{v * 100:g}%' if selection == 'quantile' else f'{v:g}' for v in cutoffs],
                           yticks=np.arange(len(methods)), yticklabels=[labels.get(m, m) for m in methods],
                           title=f'{model.upper()}\n{title}')
                    ax.tick_params(axis='y', labelsize=8)
                    for i in range(len(methods)):
                        for j in range(len(cutoffs)):
                            value = values[i, j]
                            ax.text(j, i, f'{value:.2f}' if np.isfinite(value) else 'NA', ha='center', va='center',
                                    fontsize=8, color='white' if np.isfinite(value) and value > .65 * maximum else 'black')
                fig.colorbar(display, ax=list(axes[row]), fraction=.02, pad=.02)
            fig.suptitle(f'Frozen students n={size} | {selection} | lower is better')
            fig.savefig(Path(report['folder']) / f'candidates_{selection}_n{size}.png', dpi=160, bbox_inches='tight')
            figures.append(fig)
    return figures
