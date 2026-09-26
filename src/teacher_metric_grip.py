import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch_geometric
from tqdm.auto import tqdm

from src.dataloader import get_dataset
from src.grid_search import GridStudy
from src.node_distances import array_digest
from src.ntk_readout_study import readout_features
from src.partition import geometric_medians, partition
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student
from src.trained_teacher_kernel import recover_teacher
from src.tree_distance import _neighbors, _write_json
from src.utils import BUDGET


def realize_partition(h, state):
    assignment = state['assignment'].to(h.device)
    return geometric_medians(h.double(), assignment, state['nodes']).float()


def prepare_metric_teacher(dataset, output_dir, data_dir='/content/data/', device='cuda', **settings):
    from src.empirical_ntk_study import _save
    from src.probe_teacher import fit_gcn_probe_teacher

    if dataset not in ('cora', 'citeseer', 'arxiv'):
        raise ValueError('Require a supported transductive dataset')
    graph = get_dataset(SimpleNamespace(dataset_name=dataset, raw_data_dir=str(data_dir).rstrip('/') + '/'))
    x, edges, y = graph.x.numpy(), graph.edge_index.numpy(), graph.y.numpy()
    train, val = graph.train_mask.numpy(), graph.val_mask.numpy()
    graph_hash = array_digest(x, edges)
    options = dict(hidden=256, dropout=.5, lr=.01, weight_decay=5e-4,
                   epochs=1000, eval_every=10, seed=0)
    options.update(settings)
    config = dict(version=1, dataset=dataset, graph_sha256=graph_hash,
                  supervision_sha256=array_digest(train, val, y[train], y[val]), settings=options,
                  torch=str(torch.__version__), pyg=str(torch_geometric.__version__), device=str(device))
    folder = Path(output_dir) / _fingerprint(config)
    folder.mkdir(parents=True, exist_ok=True)
    protocol = folder / 'protocol.json'
    if protocol.exists() and all((folder / name).exists() for name in ('teacher_state.pt', 'teacher_predictions.npz')):
        return dict(folder=str(folder), teacher=json.loads(protocol.read_text())['teacher'])
    teacher = fit_gcn_probe_teacher(x, edges, np.flatnonzero(train), y[train],
                                    np.flatnonzero(val), y[val], device=device, **options)
    temporary = folder / 'teacher_state.tmp'
    torch.save(teacher['state_dict'], temporary)
    temporary.replace(folder / 'teacher_state.pt')
    _save(folder / 'teacher_predictions.npz', probabilities=teacher['probabilities'], logits=teacher['logits'])
    teacher['sweep'].to_csv(folder / 'teacher_validation.csv', index=False)
    _write_json(protocol, dict(**config, teacher=teacher['config']))
    return dict(folder=str(folder), teacher=teacher['config'])


def run_teacher_metric_grip(teacher_run, output_dir, space, ratio=.026,
                            modes=('s2x', 'hidden', 'logits'), data_dir='/content/data/',
                            search_seeds=(0, 1, 2), final_seeds=tuple(range(100, 110)),
                            grip_seed=1234, grip_steps=300, grip_init='kmeans',
                            epochs=1000, eval_every=10, hidden=256, device='cuda',
                            gradient_projections=512, gradient_seeds=(6000, 7000), dataset='cora'):
    if dataset not in ('cora', 'citeseer', 'arxiv') or (dataset, ratio) not in BUDGET:
        raise ValueError('Require a supported transductive dataset and condensation ratio')
    if set(space) != {'T', 'kl_weight', 'dropout', 'lr', 'weight_decay'} or any(not v for v in space.values()):
        raise ValueError('Provide nonempty grids for T, kl_weight, dropout, lr, weight_decay')
    if not modes or not set(modes) <= {'s2x', 'hidden', 'logits', 'full_gradient'} or len(set(modes)) != len(modes):
        raise ValueError('Require distinct s2x/hidden/logits/full_gradient modes')
    if not search_seeds or not final_seeds or set(search_seeds) & set(final_seeds):
        raise ValueError('Require disjoint nonempty search and final student seeds')
    if min(space['T']) <= 0 or min(space['kl_weight']) < 0 or grip_steps < 1:
        raise ValueError('Require positive temperatures/steps and nonnegative KL weights')
    run = Path(teacher_run)
    original = json.loads((run / 'protocol.json').read_text())
    teacher = original['teacher']
    if teacher.get('model') != 'gcn' or teacher.get('layers') != 2 or teacher.get('T') != 1.:
        raise ValueError('Require the saved two-layer GCN teacher with T=1')
    if original.get('dataset', dataset) != dataset:
        raise ValueError('Saved teacher dataset differs from requested dataset')
    graph = get_dataset(SimpleNamespace(dataset_name=dataset, raw_data_dir=str(data_dir).rstrip('/') + '/'))
    x, edges = graph.x.numpy(), graph.edge_index.numpy()
    if 'graph_sha256' in original:
        digest, expected = array_digest(x, edges), original['graph_sha256']
        supervision = array_digest(graph.train_mask.numpy(), graph.val_mask.numpy(),
                                   graph.y[graph.train_mask].numpy(), graph.y[graph.val_mask].numpy())
        if supervision != original['supervision_sha256']:
            raise ValueError('Teacher training/validation supervision changed')
    else:
        _, canonical = _neighbors(edges, len(x), original['distance_protocol']['self_loops'])
        digest = hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes() + canonical.tobytes()).hexdigest()
        expected = original['distance_protocol']['input_sha256']
    if digest != expected:
        raise ValueError('Features/graph differ from the saved teacher experiment')
    with np.load(run / 'teacher_predictions.npz') as saved:
        targets = saved['probabilities'].copy()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tx, te = graph.x.to(device), graph.edge_index.to(device)
    model, recovery = recover_teacher(run, teacher, tx, te, graph.y.numpy(),
                                      graph.train_mask.numpy(), graph.val_mask.numpy(), targets, device=device)
    model.eval()
    features, logits, _, _ = readout_features(model, tx, te, 'gcn')
    _, _, validation, testing, h = _prepare_dataset(dataset, data_dir, device)
    embeddings = dict(s2x=h, hidden=features.detach(), logits=logits.detach())
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden, loss_weighting='uniform')
    config = dict(version=1, dataset=dataset, ratio=ratio, requested_nodes=BUDGET[dataset, ratio],
                  teacher=teacher, teacher_state=array_digest(*[p.cpu().numpy() for p in model.state_dict().values()]),
                  graph=array_digest(x, edges, graph.y.numpy(), graph.train_mask.numpy(),
                                     graph.val_mask.numpy(), graph.test_mask.numpy()),
                  grid=list(space.items()), modes=list(modes), settings=settings,
                  search_seeds=list(search_seeds), final_seeds=list(final_seeds),
                  grip_seed=grip_seed, grip_steps=grip_steps, grip_init=grip_init,
                  realization='geometric_median_in_original_S2X', teacher_label_override=False,
                  torch=str(torch.__version__), pyg=str(torch_geometric.__version__), device=str(device))
    if 'full_gradient' in modes:
        from src.teacher_gradient_features import full_gradient_features

        identity = {key: config[key] for key in ('teacher_state', 'graph', 'torch', 'pyg', 'device')}
        embeddings['full_gradient'], config['gradient'] = full_gradient_features(
            model, tx, te, Path(output_dir) / 'gradient_features', identity,
            projections=gradient_projections, seeds=gradient_seeds)
    folder = Path(output_dir) / _fingerprint(config)
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(folder / 'protocol.json', config)
    _write_json(folder / 'teacher_recovery.json', recovery)

    def condensed(mode, params):
        key = _fingerprint(dict(mode=mode, T=params['T'], kl_weight=params['kl_weight']))
        path = folder / f'partition_{key}.pt'
        if path.exists():
            return torch.load(path, map_location='cpu', weights_only=True)
        started = time.perf_counter()
        labels = (logits.detach() / params['T']).softmax(1)
        state = partition(embeddings[mode], labels, config['requested_nodes'],
                          kl_weight=params['kl_weight'], seed=grip_seed, iters=grip_steps,
                          init=grip_init, return_diagnostics=True)
        state['metric_centers'] = state['x']
        state['x'] = state['x'] if mode == 's2x' else realize_partition(h, state).cpu()
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        state['partition_seconds'] = time.perf_counter() - started
        temporary = path.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(path)
        return state

    selected = []
    for mode in modes:
        case = folder / mode
        case.mkdir(exist_ok=True)
        study = GridStudy(space, case)

        def objective(trial):
            state = condensed(mode, trial.params)
            cx, cy = state['x'].to(device), state['y'].to(device)
            values = [_train_student(cx, cy, validation, trial.params, seed, settings)[0]
                      for seed in search_seeds]
            trial.set_user_attr('validation_per_seed', values)
            trial.set_user_attr('nodes', state['nodes'])
            return np.mean(values)

        study.optimize(objective)
        study.trials_dataframe().to_csv(case / 'trials.csv', index=False)
        best = study.best_trial
        _write_json(case / 'best.json', dict(params=best.params, search_val=best.value))
        selected.append((mode, best))

    rows, repeats = [], []
    for mode, best in selected:
        state = condensed(mode, best.params)
        cx, cy = state['x'].to(device), state['y'].to(device)
        mode_rows = []
        for seed in tqdm(final_seeds, desc=f'{mode}: final students'):
            path = folder / mode / f'final_{seed}.json'
            if path.exists():
                record = json.loads(path.read_text())
            else:
                val, test, epoch = _train_student(cx, cy, validation, best.params, seed, settings, testing=testing)
                record = dict(mode=mode, seed=seed, validation=100 * val, test=100 * test, epoch=epoch)
                _write_json(path, record)
            mode_rows.append(record)
        frame = pd.DataFrame(mode_rows)
        repeats.extend(mode_rows)
        rows.append(dict(dataset=dataset, ratio=ratio, mode=mode, nodes=state['nodes'],
                         requested_nodes=config['requested_nodes'], **best.params,
                         search_val=100 * best.value, final_val=frame.validation.mean(),
                         final_val_std=frame.validation.std(ddof=0), test_mean=frame.test.mean(),
                         test_std=frame.test.std(ddof=0), J_initial=state['initial_J'],
                         J_final=state['final_J'], converged=state['converged'],
                         partition_seconds=state['partition_seconds']))
    summary, detail = pd.DataFrame(rows), pd.DataFrame(repeats)
    summary.to_csv(folder / 'summary.csv', index=False)
    detail.to_csv(folder / 'final_seeds.csv', index=False)
    return dict(summary=summary, repeats=detail, folder=str(folder))


def plot_teacher_metric_grip(report):
    import matplotlib.pyplot as plt

    table = report['summary']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, mean, std, title in zip(axes, ('final_val', 'test_mean'),
                                   ('final_val_std', 'test_std'), ('Validation', 'Test')):
        ax.bar(table['mode'], table[mean], yerr=table[std], capsize=4)
        for i, value in enumerate(table[mean]):
            ax.annotate(f'{value:.2f}', (i, value), xytext=(0, 8), textcoords='offset points', ha='center')
        ax.set(title=title, ylabel='Accuracy (%)', ylim=(0, 100))
    fig.savefig(Path(report['folder']) / 'accuracy.png', dpi=180)
    return fig
