import gc
import hashlib
import json
import subprocess
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.dataloader import get_dataset
from src.hyperparams import BEST_HYPERPARAMS_DICT
from src.models import GCN
from src.partition import partition as grip_partition
from src.risk_partition import risk_partition
from src.teacher import fit_logistic, get_kernel_features
from src.utils import BUDGET, normalize_adj_sparse


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def _prepare_dataset(name, data_dir, device):
    args = SimpleNamespace(dataset_name=name, raw_data_dir=str(data_dir).rstrip('/') + '/')
    datasets = get_dataset(args)

    def pack(graph):
        edges, weights = gcn_norm(graph.edge_index, graph.edge_attr, graph.num_nodes,
                                  dtype=graph.x.dtype)
        adjacency = torch.sparse_coo_tensor(edges.flip(0), weights,
                                            (graph.num_nodes, graph.num_nodes))
        adjacency = adjacency.coalesce().to_sparse_csr().to(device)
        return dict(x=graph.x.to(device), y=graph.y.to(device), adj=adjacency)

    if isinstance(datasets, list):
        train, val, test = [pack(g) for g in datasets]
        train_mask = datasets[0].train_mask.to(device)
        validation, testing = (val, None), (test, None)
    else:
        train = pack(datasets)
        train_mask = datasets.train_mask.to(device)
        validation = (train, datasets.val_mask.to(device))
        testing = (train, datasets.test_mask.to(device))
    with torch.no_grad():
        source = datasets[0] if isinstance(datasets, list) else datasets
        propagation = normalize_adj_sparse(source).coalesce().to_sparse_csr().to(device)
        H = torch.sparse.mm(propagation, torch.sparse.mm(propagation, train['x']))
    return train, train_mask, validation, testing, H


def _forward(model, x, adjacency=None):
    for i, layer in enumerate(model.layers):
        x = layer.lin(x)
        if adjacency is not None:
            x = torch.sparse.mm(adjacency, x)
        if layer.bias is not None:
            x = x + layer.bias
        if i + 1 < len(model.layers):
            x = F.dropout(F.relu(x), p=model.dropout, training=model.training)
    return F.log_softmax(x, dim=1)


@torch.no_grad()
def _accuracy(model, evaluation):
    graph, mask = evaluation
    model.eval()
    prediction = _forward(model, graph['x'], graph['adj']).argmax(1)
    correct = prediction.eq(graph['y'])
    return float(correct.float().mean() if mask is None else correct[mask].float().mean())


def _train_student(cx, cy, validation, params, seed, settings, testing=None):
    seed_everything(seed)
    model = GCN(cx.shape[1], settings['hidden'], cy.shape[1], 2, params['dropout']).to(cx.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'],
                                 weight_decay=params['weight_decay'])
    best_val, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, settings['epochs'] + 1):
        if epoch == settings['epochs'] // 2:
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'] * 0.1,
                                         weight_decay=params['weight_decay'])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = -(cy * _forward(model, cx)).sum(1).mean()
        loss.backward()
        optimizer.step()
        if epoch % settings['eval_every'] == 0 or epoch == settings['epochs']:
            value = _accuracy(model, validation)
            if value > best_val:
                best_val, best_epoch = value, epoch
                if testing is not None:
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
    test_value = None
    if testing is not None:
        model.load_state_dict(best_state)
        test_value = _accuracy(model, testing)
    return best_val, test_value, best_epoch


def run_experiments(datasets, output_dir, n_trials=20, space=None,
                    data_dir='/content/data/', search_seeds=(0, 1, 2),
                    final_seeds=tuple(range(100, 110)), partition=None, teacher=None,
                    seed=0, epochs=1000, eval_every=10, hidden=256,
                    lr=0.01, weight_decay=5e-4, device='cuda', method='risk',
                    grip_steps=300, evaluate_test=True, initial_configs=()):
    if method not in ('risk', 'grip') or grip_steps < 1:
        raise ValueError('Require risk or grip and positive grip_steps')
    if n_trials < 1 or not search_seeds or not final_seeds or min(epochs, eval_every) < 1:
        raise ValueError('Require positive trial/epoch counts and nonempty evaluation seeds')
    if space is None:
        space = dict(B=dict(low=0.01, high=100.0, log=True),
                     teacher_kernel=['erf', 'relu'], gamma=[0.01, 0.1, 1.0],
                     T=[0.2, 0.5, 1.0, 2.0], basis=[3000], dropout=[0.1, 0.5, 0.9])
        if method == 'grip':
            space.pop('B')
            space['kl_weight'] = dict(low=0.01, high=10.0, log=True)
    teacher_keys = {'teacher_kernel', 'gamma', 'T', 'basis'}
    coefficient = 'B' if method == 'risk' else 'kl_weight'
    if set(space) - teacher_keys - {coefficient, 'dropout', 'lr', 'weight_decay'}:
        raise ValueError(f'Unsupported search parameter for {method}')
    aliases = {'kernel': 'teacher_kernel', 'temperature': 'T'}
    teacher = {aliases.get(k, k): v for k, v in (teacher or {}).items()}
    if set(teacher) - teacher_keys:
        raise ValueError('Unknown teacher setting')
    solver = dict(max_sweeps=30, block_size=1024, atol=1e-12, rtol=1e-10)
    solver.update(partition or {})
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden)
    output_dir, device = Path(output_dir), torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    summaries = []

    for name, ratios in datasets.items():
        train, train_mask, validation, testing, H = _prepare_dataset(name, data_dir, device)
        teacher_cache = OrderedDict()
        feature_cache = {}

        def get_labels(params):
            kernel, gamma, temperature, basis = (params[k] for k in
                                                  ('teacher_kernel', 'gamma', 'T', 'basis'))
            if gamma < 0 or temperature <= 0 or basis < 1 or int(basis) != basis:
                raise ValueError('Require gamma >= 0, T > 0 and positive integer basis')
            basis = int(basis)
            key = (kernel, gamma, basis)
            if key not in teacher_cache:
                feature_key = (kernel, basis)
                if feature_cache.get('key') != feature_key:
                    feature_cache.clear()
                    gc.collect()
                    seed_everything(seed)
                    feature_cache['features'] = get_kernel_features(H, kernel, basis)
                    feature_cache['key'] = feature_key
                features = feature_cache['features']
                labels = F.one_hot(train['y'][train_mask], int(train['y'].max()) + 1)
                W = fit_logistic(features[train_mask], labels.to(features.dtype), gamma)
                teacher_cache[key] = (features @ W).detach().cpu()
                if len(teacher_cache) > 4:
                    teacher_cache.popitem(last=False)
            teacher_cache.move_to_end(key)
            return F.softmax(teacher_cache[key].to(device) / temperature, dim=1)

        for ratio in ratios:
            defaults = BEST_HYPERPARAMS_DICT[(name, ratio)]
            m = BUDGET[(name, ratio)]
            teacher_config = dict(teacher_kernel=defaults[0], gamma=defaults[1],
                                  T=defaults[2], basis=3000)
            teacher_config.update(teacher)
            protocol = dict(version=3, revision=revision, torch=str(torch.__version__),
                            method=method, grip_steps=grip_steps, evaluate_test=evaluate_test,
                            initial_configs=initial_configs,
                            dataset=name, ratio=ratio, budget=m, data_dir=str(data_dir),
                            teacher=teacher_config, seed=seed, partition=solver,
                            space=space, search_seeds=list(search_seeds),
                            final_seeds=list(final_seeds), student=settings,
                            lr=lr, weight_decay=weight_decay, layers=2, loss='uniform_soft_ce')
            case = output_dir / f'{name}_{ratio:g}_{_fingerprint(protocol)}'
            case.mkdir(parents=True, exist_ok=True)
            (case / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
            study = optuna.create_study(
                study_name='validation', storage=f'sqlite:///{case / "study.db"}',
                direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed),
                pruner=optuna.pruners.NopPruner(), load_if_exists=True)
            if not study.trials:
                for candidate in initial_configs:
                    if candidate['dataset'] == name and candidate['ratio'] == ratio:
                        for key, value in candidate['params'].items():
                            if key not in space:
                                raise ValueError(f'Initial parameter {key} is outside the search space')
                            specification = space[key]
                            valid = (value in specification if isinstance(specification, (list, tuple))
                                     else specification['low'] <= value <= specification['high'])
                            if not valid:
                                raise ValueError(f'Initial {key}={value} is outside the search space')
                        study.enqueue_trial(candidate['params'])
            def get_condensed(params):
                condensation_config = {k: params[k] for k in sorted(teacher_keys | {coefficient})}
                path = case / f'condensed_{_fingerprint(condensation_config)}.pt'
                if path.exists():
                    return torch.load(path, map_location='cpu', weights_only=True), path.name
                Q = get_labels(params)
                if method == 'risk':
                    condensed = risk_partition(H, Q, m, params['B'], seed=seed, **solver)
                else:
                    seed_everything(seed)
                    if H.is_cuda:
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    x, y, assignment, converged = grip_partition(
                        H, Q, m, kl_weight=params['kl_weight'], iters=grip_steps,
                        return_state=True, seed=seed)
                    condensed = dict(x=x.float().cpu(), y=y.float().cpu(),
                                     counts=torch.bincount(assignment).cpu(),
                                     converged=converged, J=None, history=[None], sweeps=None)
                    condensed['seconds'] = time.perf_counter() - started
                condensed['config'] = condensation_config
                temporary = path.with_suffix('.tmp')
                torch.save(condensed, temporary)
                temporary.replace(path)
                gc.collect()
                return condensed, path.name

            def objective(trial):
                params = dict(dropout=float(defaults[4].split(',')[0]),
                              lr=lr, weight_decay=weight_decay, **teacher_config)
                params[coefficient] = 1.0 if method == 'risk' else 0.5
                for key, specification in space.items():
                    if isinstance(specification, (list, tuple)):
                        params[key] = trial.suggest_categorical(key, list(specification))
                    elif key == 'basis':
                        params[key] = trial.suggest_int(key, **specification)
                    else:
                        params[key] = trial.suggest_float(key, **specification)
                condensed, artifact = get_condensed(params)
                cx, cy = condensed['x'].to(device), condensed['y'].to(device)
                values = [_train_student(cx, cy, validation, params, s, settings)[0]
                          for s in search_seeds]
                for key, value in dict(config=params, artifact=artifact, validation_runs=values,
                                       nodes=len(condensed['x']),
                                       J_initial=condensed['history'][0], J_final=condensed['J'],
                                       converged=condensed['converged'], sweeps=condensed['sweeps'],
                                       partition_seconds=condensed['seconds']).items():
                    trial.set_user_attr(key, value)
                return float(np.mean(values))

            completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
            remaining = max(0, n_trials - completed)
            if remaining:
                study.optimize(objective, n_trials=remaining, n_jobs=1,
                               gc_after_trial=True, show_progress_bar=True)
            study.trials_dataframe().to_csv(case / 'trials.csv', index=False)
            best = study.best_trial
            params = best.user_attrs['config']
            (case / 'best.json').write_text(json.dumps(dict(
                trial=best.number, validation=best.value, params=params,
                artifact=best.user_attrs['artifact']), indent=2), encoding='utf-8')
            final_path = case / f'final_trial_{best.number}.csv'
            if final_path.exists():
                final = pd.read_csv(final_path)
            else:
                condensed = torch.load(case / best.user_attrs['artifact'],
                                       map_location='cpu', weights_only=True)
                cx, cy = condensed['x'].to(device), condensed['y'].to(device)
                records = []
                for s in final_seeds:
                    val, test, epoch = _train_student(cx, cy, validation, params, s,
                                                     settings, testing if evaluate_test else None)
                    records.append(dict(seed=s, validation=val,
                                        test=test if test is not None else np.nan, best_epoch=epoch))
                final = pd.DataFrame(records)
                temporary = final_path.with_suffix('.tmp')
                final.to_csv(temporary, index=False)
                temporary.replace(final_path)
                del cx, cy
            summaries.append(dict(
                dataset=name, ratio=ratio, method=method, nodes=best.user_attrs['nodes'],
                requested_nodes=m, **params,
                search_val=100 * best.value, final_val=100 * final['validation'].mean(),
                final_val_std=100 * final['validation'].std(ddof=1) if len(final) > 1 else 0.0,
                test_mean=100 * final['test'].mean(),
                test_std=100 * final['test'].std(ddof=1) if len(final) > 1 else 0.0,
                J_initial=best.user_attrs['J_initial'], J_final=best.user_attrs['J_final'],
                converged=best.user_attrs['converged'], sweeps=best.user_attrs['sweeps'],
                partition_seconds=best.user_attrs['partition_seconds'], folder=str(case)))
            pd.DataFrame(summaries).to_csv(output_dir / 'summary.csv', index=False)
        teacher_cache.clear()
        feature_cache.clear()
        del train, train_mask, validation, testing, H
        gc.collect()
        torch.cuda.empty_cache()
    return pd.DataFrame(summaries)
