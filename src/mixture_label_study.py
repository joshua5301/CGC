import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.convex_representatives import convex_features
from src.dataloader import get_dataset
from src.node_distances import array_digest
from src.partition import cell_means, EPS
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student
from src.trained_teacher_kernel import recover_teacher
from src.tree_distance import _write_json


def mixture_labels(q, assignment, weights, clusters):
    assignment = assignment.to(q.device)
    weights = weights.to(device=q.device, dtype=torch.float64)
    if weights.shape != (len(q),) or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError('Require a nonnegative finite coefficient for every node')
    sums = weights.new_zeros(clusters).index_add_(0, assignment, weights)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
        raise ValueError('Mixture coefficients must sum to one in every cluster')
    labels = convex_features(q.double(), assignment, weights / sums[assignment], clusters)
    return (labels / labels.sum(1, keepdim=True)).float()


def run_mixture_label_study(source_dir, teacher_run, output_dir, seeds=tuple(range(100, 110)),
                            data_dir='/content/data/', device='cuda'):
    source = Path(source_dir)
    protocol = json.loads((source / 'protocol.json').read_text())
    if protocol['realization'] != 'within_cluster_raw_convex' or protocol['settings']['loss_weighting'] != 'uniform':
        raise ValueError('Require a raw-convex sweep with uniform student CE')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Require distinct nonempty student seeds')
    params = json.loads((source / 'hidden' / 'best.json').read_text())['params']
    key = _fingerprint(dict(mode='hidden', T=params['T'], kl_weight=params['kl_weight']))
    state = torch.load(source / f'partition_{key}.pt', map_location='cpu', weights_only=True)
    graph = get_dataset(SimpleNamespace(dataset_name=protocol['dataset'], raw_data_dir=str(data_dir).rstrip('/') + '/'))
    if array_digest(graph.x.numpy(), graph.edge_index.numpy(), graph.y.numpy(), graph.train_mask.numpy(),
                    graph.val_mask.numpy(), graph.test_mask.numpy()) != protocol['graph']:
        raise ValueError('Source graph changed')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(Path(teacher_run) / 'teacher_predictions.npz') as saved:
        expected = saved['probabilities'].copy()
    tx, edges = graph.x.to(device), graph.edge_index.to(device)
    model, recovery = recover_teacher(teacher_run, protocol['teacher'], tx, edges, graph.y.numpy(),
                                      graph.train_mask.numpy(), graph.val_mask.numpy(), expected, device=device)
    if array_digest(*[p.cpu().numpy() for p in model.state_dict().values()]) != protocol['teacher_state']:
        raise ValueError('Source teacher changed')
    with torch.no_grad():
        q = (model(tx, edges) / params['T']).softmax(1).double().clamp_min(EPS)
        mean = cell_means(q, state['assignment'].to(device), state['nodes']).float()
    if not torch.allclose(mean.cpu(), state['y'], atol=1e-5, rtol=1e-4):
        raise ValueError('Recomputed teacher targets differ from saved cluster labels')
    weighted = mixture_labels(q, state['assignment'], state['convex_weights'], state['nodes'])
    with torch.no_grad():
        reconstructed = convex_features(tx.double(), state['assignment'].to(device),
                                         state['convex_weights'].to(device), state['nodes']).float()
    if not torch.allclose(reconstructed.cpu(), state['x'], atol=1e-5, rtol=1e-4):
        raise ValueError('Saved coefficients do not reconstruct the frozen input features')
    config = dict(version=1, source=str(source), source_protocol=protocol, params=params, seeds=list(seeds),
                  frozen=array_digest(state['x'].numpy(), state['y'].numpy(), state['convex_weights'].numpy(),
                                      state['assignment'].numpy()), torch=str(torch.__version__), device=str(device))
    folder = Path(output_dir) / _fingerprint(config)
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(folder / 'protocol.json', config)
    _write_json(folder / 'teacher_recovery.json', recovery)
    _, _, validation, testing, _ = _prepare_dataset(protocol['dataset'], data_dir, device)
    cx = state['x'].to(device)
    labels = dict(cluster_mean=state['y'].to(device), mixture_mean=weighted)
    torch.save({name: value.cpu() for name, value in labels.items()}, folder / 'labels.pt')
    summary, repeats = [], []
    for method, cy in labels.items():
        rows = []
        for seed in tqdm(seeds, desc=f'{method}: paired students'):
            path = folder / f'{method}_{seed}.json'
            if path.exists():
                row = json.loads(path.read_text())
            else:
                val, test, epoch = _train_student(cx, cy, validation, params, seed, protocol['settings'], testing=testing)
                row = dict(method=method, seed=seed, validation=val * 100, test=test * 100, epoch=epoch)
                _write_json(path, row)
            rows.append(row)
        table = pd.DataFrame(rows)
        repeats.extend(rows)
        summary.append(dict(method=method, final_val=table.validation.mean(), final_val_std=table.validation.std(ddof=0),
                            test_mean=table.test.mean(), test_std=table.test.std(ddof=0)))
    summary, repeats = pd.DataFrame(summary), pd.DataFrame(repeats)
    pivot = repeats.pivot(index='seed', columns='method', values=['validation', 'test'])
    paired = pd.DataFrame({f'{metric}_delta': pivot[metric]['mixture_mean'] - pivot[metric]['cluster_mean']
                           for metric in ('validation', 'test')}).reset_index()
    original = labels['cluster_mean']
    diagnostics = dict(mean_label_l1=float((weighted - original).abs().sum(1).mean()),
                       changed_argmax=int((weighted.argmax(1) != original.argmax(1)).sum()), nodes=state['nodes'])
    _write_json(folder / 'diagnostics.json', diagnostics)
    for name, table in [('summary', summary), ('final_seeds', repeats), ('paired_deltas', paired)]:
        table.to_csv(folder / f'{name}.csv', index=False)
    return dict(summary=summary, paired=paired, diagnostics=diagnostics, folder=str(folder))


def plot_mixture_labels(report):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric, title in zip(axes, ('validation_delta', 'test_delta'), ('Validation', 'Test')):
        ax.scatter(report['paired'].seed, report['paired'][metric])
        ax.axhline(0, color='gray', linestyle='--')
        ax.set(xlabel='Student seed', ylabel='Mixture minus cluster mean (pp)', title=title)
    fig.savefig(Path(report['folder']) / 'paired_comparison.png', dpi=180)
    return fig
