import gc
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import t
from torch_geometric import seed_everything

from src.risk_analysis import _save_csv
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student
from src.risk_partition import risk_partition
from src.teacher import fit_logistic, get_kernel_features
from src.utils import BUDGET


def compare_objectives(configs, output_dir, max_sweeps=30, partition_seeds=(0,),
                       student_seeds=tuple(range(200, 210)), teacher_seed=0,
                       data_dir='/content/data/', epochs=1000, eval_every=10,
                       hidden=256, block_size=1024, device='cuda'):
    configs = pd.read_csv(configs) if isinstance(configs, (str, Path)) else pd.DataFrame(configs)
    if configs.empty or configs.duplicated(['dataset', 'ratio']).any():
        raise ValueError('Require one configuration per dataset and ratio')
    if min(max_sweeps, epochs, eval_every, block_size) < 1 or not partition_seeds or not student_seeds:
        raise ValueError('Require positive limits and nonempty seeds')
    if len(set(student_seeds)) != len(student_seeds) or len(set(partition_seeds)) != len(partition_seeds):
        raise ValueError('Seeds must be unique')
    output_dir, device = Path(output_dir), torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    keys = ['B', 'teacher_kernel', 'gamma', 'T', 'basis', 'dropout', 'lr', 'weight_decay']
    frames = []
    for name, group in configs.groupby('dataset', sort=False):
        train, mask, validation, testing, H = _prepare_dataset(name, data_dir, device)
        for config in group.to_dict('records'):
            ratio = float(config['ratio'])
            params = {k: config[k] for k in keys}
            params['basis'] = int(params['basis'])
            m = BUDGET[(name, ratio)]
            protocol = dict(revision=revision, torch=str(torch.__version__), dataset=name,
                            ratio=ratio, params=params, budget=m, max_sweeps=max_sweeps,
                            partition_seeds=list(partition_seeds), student_seeds=list(student_seeds),
                            teacher_seed=teacher_seed, student=settings, block_size=block_size,
                            data_dir=str(data_dir), device=str(device), layers=2,
                            loss='uniform_soft_ce', evaluate_test=False)
            case = output_dir / f'{name}_{ratio:g}_{_fingerprint(protocol)}'
            case.mkdir(parents=True, exist_ok=True)
            (case / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
            teacher_path = case / 'teacher.pt'
            if teacher_path.exists():
                Q = torch.load(teacher_path, weights_only=True).to(device)
            else:
                seed_everything(teacher_seed)
                features = get_kernel_features(H, params['teacher_kernel'], params['basis'])
                labels = F.one_hot(train['y'][mask], int(train['y'].max()) + 1).to(features.dtype)
                W = fit_logistic(features[mask], labels, params['gamma'])
                Q = F.softmax(features @ W / params['T'], dim=1).detach()
                temporary = teacher_path.with_suffix('.tmp')
                torch.save(Q.cpu(), temporary)
                temporary.replace(teacher_path)
                del features, W
            runs_path = case / 'runs.csv'
            records = pd.read_csv(runs_path).to_dict('records') if runs_path.exists() else []
            for ps in partition_seeds:
                initial_path = case / f'initial_{ps}.pt'
                if initial_path.exists():
                    initial = torch.load(initial_path, weights_only=True)
                else:
                    initial = risk_partition(H, Q, m, params['B'], seed=ps, max_sweeps=0,
                                             block_size=block_size, return_initial_state=True)
                    temporary = initial_path.with_suffix('.tmp')
                    torch.save(initial, temporary)
                    temporary.replace(initial_path)
                for mode in ('initial', 'variance', 'combined'):
                    path = case / f'{mode}_{ps}.pt'
                    if mode == 'initial':
                        artifact = initial
                    elif path.exists():
                        artifact = torch.load(path, weights_only=True)
                    else:
                        artifact = risk_partition(H, Q, m, params['B'], seed=ps,
                            max_sweeps=max_sweeps, block_size=block_size, objective_mode=mode,
                            initial_state=initial['initial_state'])
                        temporary = path.with_suffix('.tmp')
                        torch.save(artifact, temporary)
                        temporary.replace(path)
                    mass = artifact['counts'].double() / len(H)
                    cx, cy = artifact['x'].to(device), artifact['y'].to(device)
                    for ss in student_seeds:
                        if any(r['method'] == mode and r['partition_seed'] == ps and
                               r['student_seed'] == ss for r in records):
                            continue
                        val, _, epoch = _train_student(cx, cy, validation, params, ss, settings)
                        records.append(dict(dataset=name, ratio=ratio, partition_seed=ps,
                            student_seed=ss, method=mode, validation=100 * val, best_epoch=epoch,
                            objective=artifact['J'], bound_J=artifact['bound_J'], V=artifact['V'],
                            moment_error=artifact['moment_error'], sweeps=artifact['sweeps'],
                            converged=artifact['converged'], mass_tv=float((mass - 1 / m).abs().sum() / 2)))
                        _save_csv(pd.DataFrame(records), runs_path)
                    del cx, cy
            frames.append(pd.DataFrame(records))
            del Q, initial, artifact
        del train, mask, validation, testing, H
        gc.collect()
        torch.cuda.empty_cache()
    runs = pd.concat(frames, ignore_index=True)
    summary = runs.groupby(['dataset', 'ratio', 'partition_seed', 'method'], sort=False).agg(
        validation_mean=('validation', 'mean'), validation_std=('validation', 'std'),
        n=('student_seed', 'size'), objective=('objective', 'first'),
        bound_J=('bound_J', 'first'), V=('V', 'first'), moment_error=('moment_error', 'first'),
        mass_tv=('mass_tv', 'first'), sweeps=('sweeps', 'first'),
        converged=('converged', 'first')).reset_index()
    paired = []
    for key, group in runs.groupby(['dataset', 'ratio', 'partition_seed']):
        wide = group.pivot(index='student_seed', columns='method', values='validation')
        for baseline, method in [('initial', 'variance'), ('initial', 'combined'), ('variance', 'combined')]:
            delta = (wide[method] - wide[baseline]).dropna()
            half = float(t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else np.nan
            paired.append(dict(zip(['dataset', 'ratio', 'partition_seed'], key), baseline=baseline,
                method=method, n=len(delta), delta_val_pp=delta.mean(),
                ci95_low=delta.mean() - half, ci95_high=delta.mean() + half))
    paired = pd.DataFrame(paired)
    for name, frame in [('runs', runs), ('summary', summary), ('paired', paired)]:
        _save_csv(frame, output_dir / f'{name}.csv')
    return dict(summary=summary, paired=paired, runs=runs)
