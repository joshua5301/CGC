import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric

from src.empirical_ntk_study import _save, gram_distance, make_network, sketch_grams
from src.local_ce_distance import run_local_ce_comparison, saved_student_fingerprints
from src.node_distances import array_digest
from src.ntk_readout_study import readout_features
from src.probe_teacher import fit_gcn_probe_teacher
from src.tree_distance import _neighbors, _write_json


def probability_error(model, x, edges, expected):
    with torch.no_grad():
        actual = model(x, edges).softmax(1).cpu().numpy()
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError('Teacher probability shape or finiteness mismatch')
    return float(np.abs(actual - expected).max())


def recover_teacher(run, config, x, edges, y, train_mask, val_mask, expected,
                     checkpoint=None, device='cuda', tolerance=1e-5):
    run = Path(run)
    model = make_network(x, edges, 'gcn', config['hidden'], expected.shape[1], config['seed'], config['dropout'])
    candidates = [Path(checkpoint)] if checkpoint else sorted(set(
        path for directory in (run, run.parent) for pattern in ('*teacher*.pt', '*teacher*.pth')
        for path in directory.glob(pattern)))
    for path in candidates:
        try:
            saved = torch.load(path, map_location='cpu', weights_only=True)
            state = saved.get('state_dict', saved) if isinstance(saved, dict) else saved
            model.load_state_dict(state, strict=True)
            error = probability_error(model, x, edges, expected)
        except (RuntimeError, ValueError, TypeError, KeyError):
            if checkpoint:
                raise
            continue
        if error <= tolerance:
            return model, dict(source='checkpoint', checkpoint=str(path), probability_max_error=error)
        if checkpoint:
            raise ValueError(f'Checkpoint predictions differ from original teacher: max error {error:g}')
    if not np.array_equal(np.unique(y[train_mask]), np.asarray(config['classes'])):
        raise ValueError('Training label classes differ from original teacher')
    settings = {key: config[key] for key in ('hidden', 'dropout', 'lr', 'weight_decay', 'epochs', 'eval_every', 'seed')}
    teacher = fit_gcn_probe_teacher(x.cpu().numpy(), edges.cpu().numpy(),
                                    np.flatnonzero(train_mask), y[train_mask],
                                    np.flatnonzero(val_mask), y[val_mask], device=device, **settings)
    model.load_state_dict(teacher['state_dict'])
    model.eval()
    error = probability_error(model, x, edges, expected)
    if error > tolerance or teacher['config']['best_epoch'] != config['best_epoch']:
        raise ValueError(f'Teacher replay differs from saved teacher (max probability error {error:g}, '
                         f'epoch {teacher["config"]["best_epoch"]} versus {config["best_epoch"]}); '
                         'provide the original checkpoint. Frozen labels/students were not changed.')
    path = run / 'teacher_recovered_state.pt'
    temporary = path.with_suffix('.tmp')
    torch.save(teacher['state_dict'], temporary)
    temporary.replace(path)
    return model, dict(source='verified_replay', checkpoint=str(path), probability_max_error=error,
                       best_epoch=teacher['config']['best_epoch'])


def run_teacher_kernel_study(previous_local_dir, x, edge_index, y, train_mask, val_mask,
                              projections=512, sketch_seeds=(6000, 7000), checkpoint=None, device='cuda'):
    if projections < 1 or not sketch_seeds or len(set(sketch_seeds)) != len(sketch_seeds):
        raise ValueError('Require positive projections and distinct nonempty sketch seeds')
    previous = json.loads((Path(previous_local_dir) / 'protocol.json').read_text())
    result_dir = Path(previous['source_result'])
    source = json.loads((result_dir / 'protocol.json').read_text())
    models, fingerprints = saved_student_fingerprints(source, previous['models'])
    if fingerprints != previous['source_logits_sha256']:
        raise ValueError('Original frozen students changed')
    original = source['source_protocol']
    teacher = original['teacher']
    if teacher.get('model') != 'gcn' or teacher.get('layers') != 2 or teacher.get('T') != 1.:
        raise ValueError('This experiment requires the original two-layer GCN teacher with T=1')
    run = Path(source['source_dir'])
    ids = np.load(run / 'probe_ids.npy')
    with np.load(run / 'teacher_predictions.npz') as saved:
        targets = saved['probabilities'].copy()
    normalized = targets[ids].astype(np.float64)
    normalized /= normalized.sum(1, keepdims=True)
    if array_digest(normalized) != source['teacher_sha256']:
        raise ValueError('Original teacher targets changed')
    graph = original['distance_protocol']
    _, canonical = _neighbors(edge_index, len(x), graph['self_loops'])
    digest = hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes() + canonical.tobytes()).hexdigest()
    if digest != graph['input_sha256']:
        raise ValueError('Graph/features differ from original teacher/student experiment')
    tx = torch.as_tensor(x, dtype=torch.float32, device=device)
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    queries = torch.as_tensor(ids, dtype=torch.long, device=device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    trained, recovery = recover_teacher(run, teacher, tx, edges, np.asarray(y),
                                        np.asarray(train_mask, dtype=bool), np.asarray(val_mask, dtype=bool),
                                        targets, checkpoint, device)
    initial = make_network(tx, edges, 'gcn', teacher['hidden'], targets.shape[1], teacher['seed'], teacher['dropout'])
    state_hash = array_digest(*[p.detach().cpu().numpy() for p in trained.state_dict().values()])
    config = dict(version=1, teacher=teacher, teacher_state_sha256=state_hash,
                  input_sha256=array_digest(x, edge_index, ids), projections=projections,
                  sketch_seeds=list(sketch_seeds), torch=str(torch.__version__), pyg=str(torch_geometric.__version__),
                  device=str(device), forward_mode='eval', parameter_metric='native_euclidean',
                  tf32=False, dtype='float32_derivatives_float64_grams', teacher_probabilities_checked=True)
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    folder = result_dir / 'teacher_kernel_study' / key
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(folder / 'protocol.json', config)
    _write_json(folder / 'recovery.json', recovery)
    distances, metadata = {}, []
    with np.load(Path(previous_local_dir) / 'distances.npz') as saved:
        for name, digest in previous.get('extra_sha256', {}).items():
            if name.startswith('teacherstate_'):
                continue
            distance = saved[name].copy()
            if array_digest(distance) != digest:
                raise ValueError(f'Previous distance changed: {name}')
            distances[name] = distance
    for phase, model in (('initial', initial), ('trained', trained)):
        path = folder / f'{phase}.npz'
        if not path.exists():
            features, logits, names, biases = readout_features(model, tx, edges, 'gcn')
            features, logits = features[queries].double(), logits[queries].double()
            hidden = (features @ features.T).cpu().numpy()
            internal = np.zeros_like(hidden)
            for seed in sketch_seeds:
                gram = sketch_grams(model, tx, edges, queries, [projections], seed, exclude_names=names)
                internal += gram[projections] / len(sketch_seeds)
            _save(path, hidden=hidden, readout=hidden + biases, internal=internal,
                  full=hidden + biases + internal,
                  logits=(logits @ logits.T / logits.shape[1]).cpu().numpy())
        with np.load(path) as saved:
            for family in saved.files:
                name = f'teacherstate_{phase}_{family}'
                distances[name] = gram_distance(saved[family])
                metadata.append(dict(method=name, phase=phase, family=family))
    report = run_local_ce_comparison(result_dir, x, edge_index, previous['fractions'], previous['ks'],
                                     previous['calibration_fraction'], previous['split_seed'], models=models,
                                     extra_distances=distances, extra_protocol=dict(teacher_kernel=config, folder=str(folder)))
    report.update(metadata=pd.DataFrame(metadata), recovery=recovery, teacher_config=teacher)
    report['detail'] = report['summary'].merge(report['metadata'], on='method', how='inner')
    report['metadata'].to_csv(Path(report['folder']) / 'metadata.csv', index=False)
    report['detail'].to_csv(Path(report['folder']) / 'detail.csv', index=False)
    return report


def plot_teacher_kernel_study(report, k=10):
    import matplotlib.pyplot as plt
    table = report['detail'].query('selection == "knn" and cutoff == @k')
    figures = []
    for size in sorted(table.train_size.unique()):
        fig, axes = plt.subplots(2, len(report['models']), figsize=(5 * len(report['models']), 8),
                                 squeeze=False, constrained_layout=True)
        for col, model in enumerate(report['models']):
            part = table.query('model == @model and train_size == @size')
            for row, metric in enumerate(('relative_ce_mean_mean', 'relative_ce_p95_mean')):
                part.pivot(index='family', columns='phase', values=metric).reindex(
                    ['hidden', 'readout', 'internal', 'full', 'logits']).plot.bar(ax=axes[row, col])
                baseline = report['summary'].query('model == @model and train_size == @size and method == "grip_S2X" and selection == "knn" and cutoff == @k')
                axes[row, col].axhline(baseline[metric].iloc[0], color='black', linestyle=':', label='S²X')
                axes[row, col].set(title=f'{model.upper()} | {metric}', xlabel='')
                axes[row, col].tick_params(axis='x', rotation=30)
                axes[row, col].legend()
        fig.suptitle(f'One original GCN teacher before/after training | n={size}, k={k}')
        fig.savefig(Path(report['folder']) / f'teacher_kernel_n{size}_k{k}.png', dpi=160)
        figures.append(fig)
    return figures
