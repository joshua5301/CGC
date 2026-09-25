import json
import subprocess
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from torch_geometric import seed_everything
from tqdm.auto import tqdm

from src.hyperparams import BEST_HYPERPARAMS_DICT
from src.partition import partition
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student
from src.teacher import get_teacher_labels
from src.utils import BUDGET


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def save_tensor(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


@torch.no_grad()
def support_distances(features, train_mask, ks, block_size=512):
    ids = train_mask.nonzero().flatten()
    if not ks or min(ks) < 1 or max(ks) >= len(ids):
        raise ValueError('Require 1 <= k < number of labeled training nodes')
    x = features.double()
    centers = x[ids]
    result = {k: [] for k in ks}
    for start in range(0, len(x), block_size):
        rows = torch.arange(start, min(start + block_size, len(x)), device=x.device)
        distance = torch.cdist(x[rows], centers).square()
        distance.masked_fill_(rows[:, None].eq(ids[None, :]), torch.inf)
        nearest = distance.topk(max(ks), largest=False).values.cumsum(1)
        for k in ks:
            result[k].append((nearest[:, k - 1] / k).cpu())
    return {k: torch.cat(v).numpy() for k, v in result.items()}


def fit_reliability(distance, calibration_ids, errors, clip):
    if not np.isfinite(clip).all() or not 0 < clip[0] <= clip[1]:
        raise ValueError('Require finite ordered positive clipping limits')
    estimator = IsotonicRegression(increasing=True, out_of_bounds='clip')
    estimator.fit(distance[calibration_ids], errors)
    predicted = estimator.predict(distance)
    raw = np.clip(1 / np.maximum(predicted, 1e-12), *clip)
    weights = raw / raw.mean()
    curve = dict(distance=estimator.X_thresholds_.tolist(), error=estimator.y_thresholds_.tolist())
    return weights, predicted, curve


def blend_reliability(base, alpha):
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Require finite alpha in [0, 1]')
    return 1 + alpha * (base - 1)


def reliability_choices(ks, alphas):
    choices = []
    for alpha in sorted(set(alphas)):
        choices.extend([(None, 0.)] if alpha == 0 else [(k, alpha) for k in dict.fromkeys(ks)])
    return choices


def run_reliability_study(datasets, output_dir, grid, configs=None,
                          partition_seeds=(0, 1234), search_seeds=(0, 1, 2),
                          final_seeds=tuple(range(100, 110)), calibration_fraction=.5,
                          split_seed=2026, teacher_seed=0, clip=(.25, 4.),
                          basis=3000, grip_steps=1000, epochs=1000, eval_every=10,
                          hidden=256, lr=.01, weight_decay=.0005,
                          data_dir='/content/data/', device='cuda'):
    if set(grid) != {'k', 'alpha', 'kl_weight', 'dropout'} or any(not v for v in grid.values()):
        raise ValueError('Grid requires nonempty k, alpha, kl_weight and dropout lists')
    if any(not np.isfinite(a) or not 0 <= a <= 1 for a in grid['alpha']):
        raise ValueError('Require finite alpha in [0, 1]')
    if set(search_seeds) & set(final_seeds) or not search_seeds or not final_seeds:
        raise ValueError('Require disjoint nonempty search and final student seeds')
    if not 0 < calibration_fraction < 1 or not partition_seeds:
        raise ValueError('Require a validation split and partition seeds')
    output = Path(output_dir)
    device = torch.device(device)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    protocol = dict(version=2, weighting='inverse_brier_clipped_normalized_then_alpha',
        error_floor=1e-12, revision=revision, datasets=datasets, grid=grid,
        configs={f'{d}:{r}': v for (d, r), v in (configs or {}).items()},
        partition_seeds=list(partition_seeds), search_seeds=list(search_seeds),
        final_seeds=list(final_seeds), calibration_fraction=calibration_fraction,
        split_seed=split_seed, teacher_seed=teacher_seed, clip=list(clip), basis=basis,
        grip_steps=grip_steps, epochs=epochs, eval_every=eval_every, hidden=hidden,
        lr=lr, weight_decay=weight_decay, data_dir=str(data_dir), torch=str(torch.__version__),
        layers=2, loss='uniform_soft_ce', init='kmeans')
    root = output / _fingerprint(protocol)
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / 'protocol.json', protocol)
    save_json(output / 'latest.json', dict(folder=str(root)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden, loss_weighting='uniform')
    summaries, final_rows, diagnostic_rows = [], [], []
    for dataset, ratios in datasets.items():
        train, train_mask, validation, testing, features = _prepare_dataset(dataset, data_dir, device)
        graph, val_mask = validation
        if graph is not train or val_mask is None:
            raise ValueError('This study currently supports transductive datasets only')
        ids = val_mask.nonzero().flatten().cpu().numpy()
        cal, select = train_test_split(ids, train_size=calibration_fraction,
            random_state=split_seed, stratify=train['y'][val_mask].cpu().numpy())
        selection_mask = torch.zeros_like(val_mask)
        selection_mask[torch.as_tensor(select, device=device)] = True
        selection = (train, selection_mask)
        save_tensor(root / f'{dataset}_split.pt', dict(calibration=torch.tensor(cal), selection=torch.tensor(select)))
        distances = support_distances(features, train_mask, grid['k'])
        teacher_cache = {}
        for ratio in ratios:
            kernel, gamma, temperature, _, _ = BEST_HYPERPARAMS_DICT[(dataset, ratio)]
            config = dict(teacher_kernel=kernel, gamma=gamma, T=temperature, basis=basis)
            config.update((configs or {}).get((dataset, ratio), {}))
            if set(config) != {'teacher_kernel', 'gamma', 'T', 'basis'}:
                raise ValueError('Configs may only override teacher_kernel, gamma, T, basis')
            key = _fingerprint(config)
            teacher_path = root / f'{dataset}_teacher_{key}.pt'
            if key not in teacher_cache:
                if teacher_path.exists():
                    teacher_cache[key] = torch.load(teacher_path, weights_only=True)
                else:
                    seed_everything(teacher_seed)
                    teacher_cache[key] = get_teacher_labels(features, train_mask, train['y'],
                        config['teacher_kernel'], config['gamma'], config['T'], config['basis']).cpu()
                    save_tensor(teacher_path, teacher_cache[key])
            q = teacher_cache[key]
            y = train['y'].cpu()
            cal_error = (q[cal] - torch.nn.functional.one_hot(y[cal], q.shape[1])).square().sum(1).numpy()
            weights = {}
            for k in dict.fromkeys(grid['k']):
                base, predicted, curve = fit_reliability(distances[k], cal, cal_error, clip)
                tag = f'{dataset}_{ratio:g}_k{k}'
                save_json(root / f'{tag}_curve.json', curve)
                save_tensor(root / f'{tag}_weights.pt', dict(distance=torch.tensor(distances[k]),
                    predicted_error=torch.tensor(predicted), base_weights=torch.tensor(base)))
                for alpha in dict.fromkeys(grid['alpha']):
                    w = blend_reliability(base, alpha)
                    weights[(k, alpha)] = torch.as_tensor(w, device=device)
                    for split, subset in [('calibration', cal), ('selection', select)]:
                        error = (q[subset] - torch.nn.functional.one_hot(y[subset], q.shape[1])).square().sum(1).numpy()
                        frame = pd.DataFrame(dict(distance=distances[k][subset], error=error))
                        diagnostic_rows.append(dict(dataset=dataset, ratio=ratio, k=k, alpha=alpha, split=split,
                            spearman=frame.corr(method='spearman').iloc[0, 1],
                            error_prediction_mse=float(np.mean((predicted[subset] - error) ** 2)),
                            constant_prediction_mse=float(np.mean((cal_error.mean() - error) ** 2)),
                            weight_min=float(w.min()), weight_max=float(w.max())))
            q = q.to(device)
            for ps in partition_seeds:
                for method in ('baseline', 'reliability'):
                    folder = root / f'{dataset}_{ratio:g}_{ps}_{method}'
                    folder.mkdir(exist_ok=True)
                    choices = [(None, 0.)] if method == 'baseline' else reliability_choices(grid['k'], grid['alpha'])
                    candidates = list(product(choices, grid['kl_weight'], grid['dropout']))
                    trials = []
                    for (k, alpha), mu, dropout in tqdm(candidates, desc=f'{dataset} {ratio:g} {ps} {method}'):
                        params = dict(k=k, alpha=alpha, kl_weight=mu, dropout=dropout)
                        cid = _fingerprint(dict(k=k, alpha=alpha, kl_weight=mu))
                        artifact_path = folder / f'partition_{cid}.pt'
                        if artifact_path.exists():
                            artifact = torch.load(artifact_path, weights_only=True)
                        else:
                            artifact = partition(features, q, BUDGET[(dataset, ratio)], mu,
                                iters=grip_steps, seed=ps, return_diagnostics=True,
                                label_weights=None if k is None else weights[(k, alpha)])
                            save_tensor(artifact_path, artifact)
                        eligible = artifact['converged'] and artifact['nodes'] == BUDGET[(dataset, ratio)]
                        values = []
                        if eligible:
                            cx, cy = artifact['x'].to(device), artifact['y'].to(device)
                            student = dict(dropout=dropout, lr=lr, weight_decay=weight_decay)
                            for ss in search_seeds:
                                path = folder / f'search_{_fingerprint(params)}_{ss}.json'
                                if path.exists():
                                    value = json.loads(path.read_text())['valid']
                                else:
                                    value, _, _ = _train_student(cx, cy, selection, student, ss, settings)
                                    save_json(path, dict(valid=value))
                                values.append(value)
                        trials.append(dict(**params, eligible=eligible,
                            search_val=float(np.mean(values)) if values else None, artifact=artifact_path.name))
                        pd.DataFrame(trials).to_csv(folder / 'trials.csv', index=False)
                    eligible_trials = [t for t in trials if t['eligible']]
                    if not eligible_trials:
                        raise RuntimeError(f'No converged full-budget partition in {folder}')
                    best = max(eligible_trials, key=lambda t: t['search_val'])
                    save_json(folder / 'selected.json', best)
                    artifact = torch.load(folder / best['artifact'], weights_only=True)
                    cx, cy = artifact['x'].to(device), artifact['y'].to(device)
                    records = []
                    for ss in final_seeds:
                        path = folder / f'final_{_fingerprint(best)}_{ss}.json'
                        if path.exists():
                            record = json.loads(path.read_text())
                        else:
                            val, test, epoch = _train_student(cx, cy, selection,
                                dict(dropout=best['dropout'], lr=lr, weight_decay=weight_decay),
                                ss, settings, testing=testing)
                            record = dict(valid=100 * val, test=100 * test, epoch=epoch)
                            save_json(path, record)
                        identity = dict(dataset=dataset, ratio=ratio, partition_seed=ps, method=method)
                        records.append(record)
                        final_rows.append(dict(**identity, student_seed=ss, **record))
                    summaries.append(dict(**identity, **config, nodes=artifact['nodes'],
                        **{k: best[k] for k in ('k', 'alpha', 'kl_weight', 'dropout')},
                        search_val=100 * best['search_val'],
                        final_val=np.mean([r['valid'] for r in records]),
                        final_val_std=np.std([r['valid'] for r in records], ddof=1),
                        test_mean=np.mean([r['test'] for r in records]),
                        test_std=np.std([r['test'] for r in records], ddof=1),
                        J_initial=artifact['initial_J'], J_final=artifact['final_J']))
                    pd.DataFrame(summaries).to_csv(root / 'summary.csv', index=False)
    runs = pd.DataFrame(final_rows)
    keys = ['dataset', 'ratio', 'partition_seed', 'student_seed']
    paired = (runs[runs.method.eq('reliability')].set_index(keys)[['valid', 'test']]
              - runs[runs.method.eq('baseline')].set_index(keys)[['valid', 'test']]).reset_index()
    paired_summary = paired.groupby(keys[:-1])[['valid', 'test']].agg(['mean', 'std'])
    paired_summary.columns = ['_'.join(c) for c in paired_summary.columns]
    reports = dict(summary=pd.DataFrame(summaries), runs=runs, paired=paired,
                   paired_summary=paired_summary.reset_index(), diagnostics=pd.DataFrame(diagnostic_rows))
    for name, frame in reports.items():
        frame.to_csv(root / f'{name}.csv', index=False)
    return dict(**reports, folder=str(root))
