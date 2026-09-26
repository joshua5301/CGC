import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.empirical_ntk_study import _save, gram_distance, make_network, sketch_grams
from src.local_ce_distance import run_local_ce_comparison, saved_student_fingerprints
from src.node_distances import array_digest
from src.tree_distance import _neighbors, _write_json


def readout_features(model, x, edges, architecture):
    last = model.layers[1]
    modules = [last.lin] if architecture == 'gcn' else (
        [last.lin_l, last.lin_r] if architecture == 'sage' else [last.nn[-1]])
    captured = {}
    handles = [module.register_forward_pre_hook(
        lambda module, inputs, index=i: captured.__setitem__(index, inputs[0]))
        for i, module in enumerate(modules)]
    try:
        with torch.no_grad():
            output = model(x, edges)
    finally:
        for handle in handles:
            handle.remove()
    if architecture == 'gcn':
        indices, weights = last._cached_edge_index
        propagation = torch.sparse_coo_tensor(indices.flip(0), weights, (len(x), len(x))).coalesce()
        features = torch.sparse.mm(propagation, captured[0])
        readout = list(last.lin.parameters()) + [last.bias]
        biases = int(last.bias is not None)
    else:
        features = torch.cat([captured[i] for i in range(len(modules))], dim=1)
        readout = [p for module in modules for p in module.parameters()]
        biases = sum(module.bias is not None for module in modules)
    parameter_ids = {id(p) for p in readout if p is not None}
    names = [name for name, p in model.named_parameters() if id(p) in parameter_ids]
    return features, output, names, biases


def run_readout_study(previous_local_dir, x, edge_index, device='cuda'):
    previous_local_dir = Path(previous_local_dir)
    previous = json.loads((previous_local_dir / 'protocol.json').read_text())
    old_config = previous['extra_protocol']['study']
    old_cache = Path(previous['extra_protocol']['folder'])
    result_dir = Path(previous['source_result'])
    source = json.loads((result_dir / 'protocol.json').read_text())
    models, fingerprints = saved_student_fingerprints(source, previous['models'])
    if fingerprints != previous['source_logits_sha256']:
        raise ValueError('Original student outputs changed')
    ids = np.load(Path(source['source_dir']) / 'probe_ids.npy')
    if array_digest(x, edge_index, ids) != old_config['input_sha256']:
        raise ValueError('Graph/features/probe nodes differ from the preceding NTK study')
    graph = source['source_protocol']['distance_protocol']
    _, canonical = _neighbors(edge_index, len(x), graph['self_loops'])
    digest = hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes() + canonical.tobytes()).hexdigest()
    if digest != graph['input_sha256']:
        raise ValueError('Graph/features differ from the frozen students')
    if str(torch.__version__) != old_config['torch']:
        raise ValueError('Use the same PyTorch version as the preceding NTK study')
    config = dict(version=1, original=old_config, projections=max(old_config['projections']),
                  ensembles=max(old_config['ensemble_sizes']), device=str(device),
                  decomposition='exact_readout_plus_independently_sketched_internal_parameters')
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    folder = result_dir / 'ntk_readout_study' / key
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(folder / 'protocol.json', config)
    tx = torch.as_tensor(x, device=device, dtype=torch.float32)
    edges = torch.as_tensor(edge_index, device=device, dtype=torch.long)
    queries = torch.as_tensor(ids, device=device, dtype=torch.long)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    distances, metadata, checks = {}, [], []
    with np.load(previous_local_dir / 'distances.npz') as saved:
        for name, digest in previous['extra_sha256'].items():
            if name.startswith('deep_'):
                continue
            value = saved[name].copy()
            if array_digest(value) != digest:
                raise ValueError(f'Previous distance changed: {name}')
            distances[name] = value
    for architecture in old_config['architectures']:
        for width in old_config['widths']:
            totals = {name: np.zeros((len(ids), len(ids))) for name in ('hidden', 'readout', 'internal', 'full', 'logits')}
            for seed in tqdm(old_config['network_seeds'][:config['ensembles']], desc=f'Readout {architecture} w{width}'):
                path = folder / f'{architecture}_w{width}_seed{seed}.npz'
                if not path.exists():
                    model = make_network(tx, edges, architecture, width, old_config['outputs'], seed)
                    features, output, names, biases = readout_features(model, tx, edges, architecture)
                    features, output = features[queries].double(), output[queries].double()
                    cached = old_cache / f'{architecture}_w{width}_seed{seed}_outputs.npz'
                    with np.load(cached) as saved:
                        difference = float(np.max(np.abs(output.cpu().numpy() - saved['output'])))
                        if not np.allclose(output.cpu().numpy(), saved['output'], rtol=1e-5, atol=1e-6):
                            raise ValueError('Reconstructed network does not match preceding cached logits')
                    hidden = (features @ features.T).cpu().numpy()
                    readout = hidden + biases
                    internal = np.zeros_like(hidden)
                    for replica in old_config['sketch_seeds']:
                        kernel = sketch_grams(model, tx, edges, queries, [config['projections']],
                                              replica + 100003 * seed, exclude_names=names)
                        internal += kernel[config['projections']] / len(old_config['sketch_seeds'])
                    _save(path, hidden=hidden, readout=readout, internal=internal, full=readout + internal,
                          logits=(output @ output.T / output.shape[1]).cpu().numpy(),
                          feature_width=features.shape[1], reconstruction_max_error=difference)
                with np.load(path) as saved:
                    for name in totals:
                        totals[name] += saved[name] / config['ensembles']
                    checks.append(dict(architecture=architecture, width=width, seed=seed,
                                       feature_width=int(saved['feature_width']),
                                       reconstruction_max_error=float(saved['reconstruction_max_error'])))
            for family, gram in totals.items():
                name = f'readout_{architecture}_w{width}_{family}'
                distances[name] = gram_distance(gram)
                metadata.append(dict(method=name, architecture=architecture, width=width, family=family))
    report = run_local_ce_comparison(result_dir, x, edge_index, previous['fractions'], previous['ks'],
                                     previous['calibration_fraction'], previous['split_seed'], models=models,
                                     extra_distances=distances, extra_protocol=dict(readout=config, folder=str(folder)))
    report['metadata'] = pd.DataFrame(metadata)
    report['checks'] = pd.DataFrame(checks)
    report['detail'] = report['summary'].merge(report['metadata'], on='method', how='inner')
    for name in ('metadata', 'checks', 'detail'):
        report[name].to_csv(Path(report['folder']) / f'{name}.csv', index=False)
    return report


def plot_readout_study(report, k=10):
    import matplotlib.pyplot as plt
    table = report['detail']
    table = table[(table.model == table.architecture) & (table.selection == 'knn') & (table.cutoff == k)]
    models = list(table.model.unique())
    figures = []
    for size in sorted(table.train_size.unique()):
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False, constrained_layout=True)
        for ax, model in zip(axes[0], models):
            part = table[(table.model == model) & (table.train_size == size)]
            part.pivot(index='family', columns='width', values='relative_ce_mean_mean').reindex(
                ['logits', 'hidden', 'readout', 'internal', 'full']).plot.bar(ax=ax)
            baseline = report['summary'].query('model == @model and train_size == @size and method == "grip_S2X" and selection == "knn" and cutoff == @k')
            ax.axhline(baseline.relative_ce_mean_mean.iloc[0], color='black', linestyle=':', label='S²X')
            ax.set(title=model.upper(), ylabel='Relative CE change (lower is better)', xlabel='')
            ax.tick_params(axis='x', rotation=30)
            ax.legend()
        fig.suptitle(f'Same networks, readout decomposition | n={size}, k={k}')
        fig.savefig(Path(report['folder']) / f'readout_n{size}_k{k}.png', dpi=160)
        figures.append(fig)
    return figures
