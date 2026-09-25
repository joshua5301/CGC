import json
import subprocess
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric import seed_everything
from tqdm.auto import tqdm

from src.grip_reliability import save_json, save_tensor, support_distances
from src.grip_mixture import fit_contamination, mixture_partition
from src.hyperparams import BEST_HYPERPARAMS_DICT
from src.partition import partition
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student
from src.teacher import get_teacher_labels
from src.utils import BUDGET


def run_mixture_study(datasets, output_dir, grid, configs=None,
                          partition_seeds=(0, 1234), search_seeds=(0, 1, 2),
                          final_seeds=tuple(range(100, 110)), calibration_fraction=.5,
                          split_seed=2026, teacher_seed=0, k=10, prior_clip=(.05, .95), smoothing=5.,
                          basis=3000, grip_steps=1000, epochs=1000, eval_every=10,
                          hidden=256, lr=.01, weight_decay=.0005,
                          data_dir='/content/data/', device='cuda'):
    if set(grid) != {'observations', 'kl_weight', 'dropout'} or any(not v for v in grid.values()):
        raise ValueError('Grid requires nonempty observations, kl_weight and dropout lists')
    if any(int(n) != n or n < 1 for n in grid['observations']):
        raise ValueError('Require positive integer observation counts')
    if set(search_seeds) & set(final_seeds) or not search_seeds or not final_seeds:
        raise ValueError('Require disjoint nonempty search and final student seeds')
    if not 0 < calibration_fraction < 1 or not partition_seeds:
        raise ValueError('Require a validation split and partition seeds')
    output = Path(output_dir)
    device = torch.device(device)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    protocol = dict(version=1, model='multinomial_contamination', revision=revision, datasets=datasets, grid=grid,
        configs={f'{d}:{r}': v for (d, r), v in (configs or {}).items()},
        partition_seeds=list(partition_seeds), search_seeds=list(search_seeds),
        final_seeds=list(final_seeds), calibration_fraction=calibration_fraction,
        split_seed=split_seed, teacher_seed=teacher_seed, k=k, prior_clip=list(prior_clip), smoothing=smoothing, basis=basis,
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
        distance = support_distances(features, train_mask, [k])[k]
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
            noise = {}
            for observations in grid['observations']:
                obs, rho, background, diagnostic = fit_contamination(q, cal, y, distance,
                    observations, prior_clip=prior_clip, smoothing=smoothing)
                correct = q[select].argmax(1).eq(y[select]).double().numpy()
                diagnostic_rows.append(dict(dataset=dataset, ratio=ratio, observations=observations,
                    calibration_errors=diagnostic['calibration_errors'],
                    prior_mean=float(rho.mean()), prior_min=float(rho.min()), prior_max=float(rho.max()),
                    selection_prior_brier=float(np.mean((rho[select].numpy() - correct) ** 2)),
                    quantization_l1=float((obs - q).abs().sum(1).mean())))
                save_json(root / f'{dataset}_{ratio:g}_noise_{observations}.json', diagnostic)
                save_tensor(root / f'{dataset}_{ratio:g}_noise_{observations}.pt',
                    dict(observed=obs, prior=rho, background=background, distance=torch.tensor(distance)))
                noise[observations] = (obs.to(device), rho.to(device), background.to(device))
            q = q.to(device)
            for ps in partition_seeds:
                for method in ('baseline', 'quantized', 'mixture'):
                    folder = root / f'{dataset}_{ratio:g}_{ps}_{method}'
                    folder.mkdir(exist_ok=True)
                    choices = [None] if method == 'baseline' else list(dict.fromkeys(grid['observations']))
                    candidates = list(product(choices, grid['kl_weight'], grid['dropout']))
                    trials = []
                    for observations, mu, dropout in tqdm(candidates, desc=f'{dataset} {ratio:g} {ps} {method}'):
                        params = dict(observations=observations, kl_weight=mu, dropout=dropout)
                        cid = _fingerprint(dict(observations=observations, kl_weight=mu))
                        artifact_path = folder / f'partition_{cid}.pt'
                        if artifact_path.exists():
                            artifact = torch.load(artifact_path, weights_only=True)
                        else:
                            if method == 'mixture':
                                obs, rho, background = noise[observations]
                                artifact = mixture_partition(features, obs, BUDGET[(dataset, ratio)],
                                    rho, background, observations, kl_weight=mu, seed=ps, steps=grip_steps)
                            else:
                                labels = q if method == 'baseline' else noise[observations][0]
                                artifact = partition(features, labels, BUDGET[(dataset, ratio)], mu,
                                    iters=grip_steps, seed=ps, return_diagnostics=True)
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
                            status=artifact.get('status', 'converged' if artifact['converged'] else 'iteration_limit'),
                            nodes=artifact['nodes'], posterior_mean=artifact.get('posterior_mean'),
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
                        **{k: best[k] for k in ('observations', 'kl_weight', 'dropout')},
                        search_val=100 * best['search_val'],
                        final_val=np.mean([r['valid'] for r in records]),
                        final_val_std=np.std([r['valid'] for r in records], ddof=1),
                        test_mean=np.mean([r['test'] for r in records]),
                        test_std=np.std([r['test'] for r in records], ddof=1),
                        J_initial=artifact['initial_J'], J_final=artifact['final_J'],
                        posterior_mean=artifact.get('posterior_mean'),
                        posterior_low_share=artifact.get('posterior_low_share'),
                        converged=artifact['converged']))
                    pd.DataFrame(summaries).to_csv(root / 'summary.csv', index=False)
    runs = pd.DataFrame(final_rows)
    keys = ['dataset', 'ratio', 'partition_seed', 'student_seed']
    pairs = []
    for reference in ('baseline', 'quantized'):
        frame = (runs[runs.method.eq('mixture')].set_index(keys)[['valid', 'test']]
                 - runs[runs.method.eq(reference)].set_index(keys)[['valid', 'test']]).reset_index()
        pairs.append(frame.assign(comparison=f'mixture_minus_{reference}'))
    paired = pd.concat(pairs, ignore_index=True)
    paired_summary = paired.groupby(keys[:-1] + ['comparison'])[['valid', 'test']].agg(['mean', 'std'])
    paired_summary.columns = ['_'.join(c) for c in paired_summary.columns]
    reports = dict(summary=pd.DataFrame(summaries), runs=runs, paired=paired,
                   paired_summary=paired_summary.reset_index(), diagnostics=pd.DataFrame(diagnostic_rows))
    for name, frame in reports.items():
        frame.to_csv(root / f'{name}.csv', index=False)
    return dict(**reports, folder=str(root))
