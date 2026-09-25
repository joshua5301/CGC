import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric import seed_everything
from tqdm.auto import tqdm

from src.hyperparams import BEST_HYPERPARAMS_DICT
from src.partition import partition
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student
from src.teacher import get_teacher_labels
from src.utils import BUDGET


def _save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def selection_plan(costs, sizes, repeats, seed):
    rows = []
    for init, group in costs.groupby('init', sort=True):
        rng = np.random.default_rng(seed)
        eligible = group[(~group['reference']) & group['converged'] &
                         (group['nodes'] == group['requested_nodes'])].sort_values('partition_seed')
        if len(eligible) < max(sizes):
            raise ValueError(f'{init}: only {len(eligible)} converged full-budget candidates; inspect costs.csv')
        for repeat in range(repeats):
            order = rng.permutation(len(eligible))
            for size in sizes:
                pool = eligible.iloc[order[:size]]
                choices = {'random': pool.iloc[0],
                           'min_sse': pool.loc[pool.initial_sse.idxmin()],
                           'min_grip': pool.loc[pool.final_J.idxmin()]}
                for rule, chosen in choices.items():
                    rows.append(dict(init=init, repeat=repeat, candidates=size, rule=rule,
                                     candidate=chosen.candidate))
    return pd.DataFrame(rows)


def run_initialization_study(datasets, output_dir, partition_seeds=tuple(range(10, 30)),
                             inits=('kmeans', 'kmeans++'), candidate_sizes=(1, 5, 10, 20),
                             selection_repeats=20, selection_seed=2026, reference_seed=1234,
                             discovery_seeds=tuple(range(5)), confirmation_seeds=tuple(range(200, 210)),
                             teacher_seed=0, configs=None, data_dir='/content/data/',
                             grip_steps=300, epochs=1000, eval_every=10, hidden=256,
                             basis=3000, lr=.01, weight_decay=.0005, device='cuda'):
    if (not discovery_seeds or not confirmation_seeds or
        set(discovery_seeds) & set(confirmation_seeds) or
        len(set(partition_seeds)) != len(partition_seeds) or reference_seed in partition_seeds):
        raise ValueError('Require disjoint student seeds and unique candidate seeds excluding the reference')
    if not candidate_sizes or min(candidate_sizes) < 1 or selection_repeats < 1 or grip_steps < 1:
        raise ValueError('Require positive candidate sizes, repetitions, and GRIP steps')
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden, loss_weighting='uniform')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    output_dir, device = Path(output_dir), torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    candidates_all, selections_all, correlations_all = [], [], []
    for dataset, ratios in datasets.items():
        train, train_mask, validation, testing, H = _prepare_dataset(dataset, data_dir, device)
        for ratio in ratios:
            kernel, gamma, temperature, kl_weight, dropout = BEST_HYPERPARAMS_DICT[(dataset, ratio)]
            params = dict(teacher_kernel=kernel, gamma=gamma, T=temperature, kl_weight=kl_weight,
                          dropout=float(dropout), basis=basis, lr=lr, weight_decay=weight_decay)
            params.update((configs or {}).get((dataset, ratio), {}))
            protocol = dict(revision=revision, torch=str(torch.__version__), dataset=dataset, ratio=ratio,
                            data_dir=str(data_dir), params=params, partition_seeds=list(partition_seeds),
                            inits=list(inits), candidate_sizes=list(candidate_sizes),
                            selection_repeats=selection_repeats, selection_seed=selection_seed,
                            reference_seed=reference_seed, discovery_seeds=list(discovery_seeds),
                            confirmation_seeds=list(confirmation_seeds), teacher_seed=teacher_seed,
                            grip_steps=grip_steps, student=settings)
            case = output_dir / f'{dataset}_{ratio:g}_{_fingerprint(protocol)}'
            case.mkdir(exist_ok=True)
            _save_json(case / 'protocol.json', protocol)
            teacher_path = case / 'teacher.pt'
            if teacher_path.exists():
                Q = torch.load(teacher_path, map_location=device, weights_only=True)
            else:
                seed_everything(teacher_seed)
                Q = get_teacher_labels(H, train_mask, train['y'], params['teacher_kernel'],
                                       params['gamma'], params['T'], params['basis'])
                temporary = teacher_path.with_suffix('.tmp')
                torch.save(Q.cpu(), temporary)
                temporary.replace(teacher_path)
            m, rows = BUDGET[(dataset, ratio)], []
            jobs = [(init, s, False) for init in inits for s in partition_seeds]
            jobs.append(('kmeans', reference_seed, True))
            for init, seed, reference in tqdm(jobs, desc=f'{dataset} {ratio:g} partitions'):
                candidate = f'{init}_{seed}'
                path = case / f'{candidate}.pt'
                if path.exists():
                    result = torch.load(path, weights_only=True, map_location='cpu')
                else:
                    result = partition(H, Q, m, params['kl_weight'], iters=grip_steps,
                                       seed=seed, init=init, return_diagnostics=True)
                    temporary = path.with_suffix('.tmp')
                    torch.save(result, temporary)
                    temporary.replace(path)
                row = {k: v for k, v in result.items() if not isinstance(v, torch.Tensor)}
                rows.append(dict(dataset=dataset, ratio=ratio, candidate=candidate, init=init,
                                 partition_seed=seed, reference=reference, requested_nodes=m, **row))
            costs = pd.DataFrame(rows)
            costs.to_csv(case / 'costs.csv', index=False)
            plan = selection_plan(costs, candidate_sizes, selection_repeats, selection_seed)
            plan.to_csv(case / 'selection_plan.csv', index=False)

            def evaluate(candidate, seeds):
                path = case / f'{candidate}_validation.json'
                runs = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
                result = torch.load(case / f'{candidate}.pt', map_location='cpu', weights_only=True)
                cx, cy = result['x'].to(device), result['y'].to(device)
                for seed in seeds:
                    if str(seed) not in runs:
                        val, _, epoch = _train_student(cx, cy, validation, params, seed, settings)
                        runs[str(seed)] = dict(validation=100 * val, best_epoch=epoch)
                        _save_json(path, runs)
                values = [runs[str(seed)]['validation'] for seed in seeds]
                return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.

            for row in tqdm(rows, desc=f'{dataset} {ratio:g} discovery'):
                row['discovery_val'], row['discovery_std'] = evaluate(row['candidate'], discovery_seeds)
            costs = pd.DataFrame(rows)
            costs.to_csv(case / 'candidates.csv', index=False)
            selected = set(plan.candidate) | set(costs.loc[costs.reference, 'candidate'])
            confirmed = {}
            for candidate in tqdm(sorted(selected), desc=f'{dataset} {ratio:g} confirmation'):
                confirmed[candidate] = evaluate(candidate, confirmation_seeds)
            costs['confirmation_val'] = costs.candidate.map(lambda c: confirmed.get(c, (np.nan, np.nan))[0])
            costs['confirmation_std'] = costs.candidate.map(lambda c: confirmed.get(c, (np.nan, np.nan))[1])
            costs.to_csv(case / 'candidates.csv', index=False)
            plan = plan.merge(costs[['candidate', 'discovery_val', 'confirmation_val']], on='candidate', validate='many_to_one')
            plan['dataset'], plan['ratio'] = dataset, ratio
            plan.to_csv(case / 'selections.csv', index=False)
            for init, group in costs[~costs.reference].groupby('init'):
                for metric in ('initial_sse', 'initial_J', 'final_J', 'final_feature', 'final_weighted_kl'):
                    correlations_all.append(dict(dataset=dataset, ratio=ratio, init=init, metric=metric,
                        spearman=group[metric].corr(group.discovery_val, method='spearman'), n=len(group)))
            candidates_all.append(costs)
            selections_all.append(plan)
            del Q
        del train, train_mask, validation, testing, H
        torch.cuda.empty_cache()
    candidates = pd.concat(candidates_all, ignore_index=True)
    selections = pd.concat(selections_all, ignore_index=True)
    summary = selections.groupby(['dataset', 'ratio', 'init', 'candidates', 'rule'], as_index=False).agg(
        discovery_val=('discovery_val', 'mean'), confirmation_val=('confirmation_val', 'mean'),
        subset_std=('confirmation_val', 'std'), unique_selected=('candidate', 'nunique'))
    baseline = summary[summary.rule == 'random'][['dataset', 'ratio', 'init', 'candidates', 'confirmation_val']]
    summary = summary.merge(baseline.rename(columns={'confirmation_val': 'random_val'}),
                            on=['dataset', 'ratio', 'init', 'candidates'], validate='many_to_one')
    summary['gain_vs_random'] = summary.confirmation_val - summary.random_val
    correlations = pd.DataFrame(correlations_all)
    for name, frame in dict(candidates=candidates, selections=selections, summary=summary,
                            correlations=correlations).items():
        frame.to_csv(output_dir / f'{name}.csv', index=False)
    return dict(candidates=candidates, selections=selections, summary=summary, correlations=correlations)
