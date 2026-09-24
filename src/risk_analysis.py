import gc
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import t
from torch_geometric import seed_everything

from src.partition import partition as grip_partition
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student
from src.risk_partition import risk_partition
from src.teacher import fit_logistic, get_kernel_features
from src.utils import BUDGET


def _save_csv(frame, path):
    temporary = path.with_suffix('.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run_ablation(configs, output_dir, checkpoints=(0, 10, 30), max_sweeps=300,
                 partition_seeds=(0,), student_seeds=tuple(range(200, 210)),
                 teacher_seed=0, grip=True, kl_weight=0.5, grip_steps=300,
                 data_dir='/content/data/', epochs=1000, eval_every=10,
                 hidden=256, block_size=1024, evaluate_test=False, device='cuda'):
    configs = pd.read_csv(configs) if isinstance(configs, (str, Path)) else pd.DataFrame(configs)
    checkpoints = sorted(set([0, *checkpoints]))
    if not partition_seeds or not student_seeds or checkpoints[-1] > max_sweeps or checkpoints[0] < 0:
        raise ValueError('Require nonempty seeds and checkpoints within the sweep limit')
    if min(epochs, eval_every, max_sweeps, grip_steps) < 1:
        raise ValueError('Require positive epoch and sweep limits')
    if configs.duplicated(['dataset', 'ratio']).any():
        raise ValueError('Provide one fixed configuration per dataset and ratio')
    output_dir, device = Path(output_dir), torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frames, traces = [], []
    keys = ['B', 'dropout', 'lr', 'weight_decay', 'teacher_kernel', 'gamma', 'T', 'basis']
    for name, group in configs.groupby('dataset', sort=False):
        train, mask, validation, testing, H = _prepare_dataset(name, data_dir, device)
        for config in group.to_dict('records'):
            ratio = float(config['ratio'])
            params = {k: config[k] for k in keys}
            params['basis'] = int(params['basis'])
            m = BUDGET[(name, ratio)]
            protocol = dict(revision=revision, torch=str(torch.__version__), dataset=name,
                            ratio=ratio, params=params, budget=m, checkpoints=checkpoints,
                            max_sweeps=max_sweeps, partition_seeds=list(partition_seeds),
                            student_seeds=list(student_seeds), teacher_seed=teacher_seed,
                            grip=grip, kl_weight=kl_weight, grip_steps=grip_steps,
                            student=settings, block_size=block_size, evaluate_test=evaluate_test,
                            data_dir=str(data_dir), device=str(device),
                            layers=2, loss='uniform_soft_ce')
            case = output_dir / f'{name}_{ratio:g}_{_fingerprint(protocol)}'
            case.mkdir(parents=True, exist_ok=True)
            (case / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
            labels_path = case / 'teacher.pt'
            if labels_path.exists():
                Q = torch.load(labels_path, weights_only=True).to(device)
            else:
                seed_everything(teacher_seed)
                features = get_kernel_features(H, params['teacher_kernel'], params['basis'])
                labels = F.one_hot(train['y'][mask], int(train['y'].max()) + 1).to(features.dtype)
                W = fit_logistic(features[mask], labels, params['gamma'])
                Q = F.softmax(features @ W / params['T'], dim=1).detach()
                torch.save(Q.cpu(), labels_path)
                del features, W
            runs_path = case / 'runs.csv'
            records = pd.read_csv(runs_path).to_dict('records') if runs_path.exists() else []
            for ps in partition_seeds:
                path = case / f'partition_{ps}.pt'
                if path.exists():
                    artifacts = torch.load(path, weights_only=True)
                else:
                    result = risk_partition(H, Q, m, params['B'], seed=ps,
                                            max_sweeps=max_sweeps, block_size=block_size,
                                            checkpoints=checkpoints)
                    snapshots = result.pop('snapshots')
                    artifacts = {}
                    for sweep in checkpoints:
                        point = snapshots.get(sweep, result)
                        artifacts[f'risk_{sweep}'] = dict(point,
                            converged=result['converged'] and point['sweeps'] == result['sweeps'])
                    artifacts['risk_terminal'] = result
                    if grip:
                        seed_everything(ps)
                        if H.is_cuda:
                            torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        x, y, assignment, converged = grip_partition(
                            H, Q, m, kl_weight=kl_weight, iters=grip_steps, return_state=True, seed=ps)
                        artifacts['grip'] = dict(x=x.float().cpu(), y=y.float().cpu(),
                            counts=torch.bincount(assignment).cpu(), converged=converged,
                            J=float('nan'), V=float('nan'), moment_error=float('nan'),
                            sweeps=float('nan'), seconds=time.perf_counter() - start)
                    temporary = path.with_suffix('.tmp')
                    torch.save(artifacts, temporary)
                    temporary.replace(path)
                terminal = artifacts['risk_terminal']
                traces.extend(dict(dataset=name, ratio=ratio, partition_seed=ps, sweep=i, J=j)
                              for i, j in enumerate(terminal['history']))
                for method, artifact in artifacts.items():
                    counts = artifact['counts'].double()
                    mass = counts / counts.sum()
                    metadata = dict(dataset=name, ratio=ratio, method=method, partition_seed=ps,
                        nodes=len(counts), requested_nodes=m, J=artifact['J'], V=artifact['V'],
                        moment_error=artifact['moment_error'], sweeps=artifact['sweeps'],
                        converged=artifact['converged'], partition_seconds=artifact['seconds'],
                        mass_tv=float((mass - 1 / len(mass)).abs().sum() / 2),
                        count_cv=float(counts.std(unbiased=False) / counts.mean()),
                        min_count=int(counts.min()), max_count=int(counts.max()))
                    cx, cy = artifact['x'].to(device), artifact['y'].to(device)
                    for ss in student_seeds:
                        if any(r['method'] == method and r['partition_seed'] == ps and
                               r['student_seed'] == ss for r in records):
                            continue
                        val, test, epoch = _train_student(cx, cy, validation, params, ss,
                                                         settings, testing if evaluate_test else None)
                        records.append(dict(metadata, student_seed=ss, validation=100 * val,
                                            test=100 * test if test is not None else np.nan,
                                            best_epoch=epoch))
                        _save_csv(pd.DataFrame(records), runs_path)
                    del cx, cy
            frames.append(pd.DataFrame(records))
            del Q, artifacts
        del train, mask, validation, testing, H
        gc.collect()
        torch.cuda.empty_cache()
    runs, history = pd.concat(frames, ignore_index=True), pd.DataFrame(traces)
    _save_csv(runs, output_dir / 'runs.csv')
    _save_csv(history, output_dir / 'history.csv')
    return analyze_ablation(runs, output_dir)


def analyze_ablation(runs, output_dir):
    runs = pd.read_csv(runs) if isinstance(runs, (str, Path)) else runs.copy()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = ['dataset', 'ratio', 'partition_seed', 'method']
    summary = runs.groupby(groups, sort=False).agg(
        validation_mean=('validation', 'mean'), validation_std=('validation', 'std'),
        test_mean=('test', 'mean'), test_std=('test', 'std'),
        n=('student_seed', 'size'), J=('J', 'first'), V=('V', 'first'),
        moment_error=('moment_error', 'first'), converged=('converged', 'first'),
        sweeps=('sweeps', 'first'), nodes=('nodes', 'first'),
        mass_tv=('mass_tv', 'first'), count_cv=('count_cv', 'first'),
        partition_seconds=('partition_seconds', 'first')).reset_index()
    paired = []
    for key, group in runs.groupby(['dataset', 'ratio', 'partition_seed']):
        wide = group.pivot(index='student_seed', columns='method', values='validation')
        comparisons = [('risk_0', x) for x in wide if x.startswith('risk_') and x != 'risk_0']
        comparisons += [('risk_30', 'risk_terminal'), ('grip', 'risk_terminal')]
        for base, method in comparisons:
            if base not in wide or method not in wide:
                continue
            delta = (wide[method] - wide[base]).dropna()
            half = float(t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else np.nan
            paired.append(dict(zip(['dataset', 'ratio', 'partition_seed'], key),
                baseline=base, method=method, n=len(delta), delta_val_pp=delta.mean(),
                ci95_low=delta.mean() - half, ci95_high=delta.mean() + half))
    paired = pd.DataFrame(paired)
    _save_csv(summary, output_dir / 'summary.csv')
    _save_csv(paired, output_dir / 'paired.csv')
    import matplotlib.pyplot as plt
    for (name, ratio, ps), group in summary.groupby(['dataset', 'ratio', 'partition_seed']):
        trajectory = group[group.method.str.startswith('risk_')].sort_values('sweeps')
        trajectory = trajectory.drop_duplicates('sweeps')
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        initial = float(trajectory.iloc[0].J)
        axes[0].plot(trajectory.sweeps, trajectory.J / initial if initial else trajectory.J, 'o-')
        axes[0].set(xlabel='Sweeps', ylabel='J / initial J' if initial else 'J')
        axes[1].errorbar(trajectory.sweeps, trajectory.validation_mean,
                         yerr=trajectory.validation_std.fillna(0), fmt='o-')
        baseline = group[group.method == 'grip']
        if not baseline.empty:
            axes[1].axhline(baseline.iloc[0].validation_mean, color='gray', linestyle='--', label='GRIP')
            axes[1].legend()
        axes[1].set(xlabel='Sweeps', ylabel='Validation accuracy (%) / seed SD')
        axes[2].plot(trajectory.sweeps, trajectory.mass_tv, 'o-')
        axes[2].set(xlabel='Sweeps', ylabel='TV(cell mass, uniform)')
        fig.suptitle(f'{name} / {ratio:g} / partition seed {ps}')
        fig.tight_layout()
        fig.savefig(output_dir / f'{name}_{ratio:g}_{ps}.png', dpi=180)
        plt.close(fig)
    return dict(summary=summary, paired=paired, runs=runs)
