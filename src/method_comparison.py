import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from src.risk_experiment import run_experiments


def compare_methods(datasets, output_dir, common_space, risk_space, grip_space,
                    n_trials=100, search_seeds=(0, 1, 2),
                    final_seeds=tuple(range(200, 210)), initial_configs=None,
                    max_sweeps=300, grip_steps=300, block_size=1024,
                    data_dir='/content/data/', epochs=1000, eval_every=10,
                    hidden=256, seed=0, device='cuda'):
    if set(search_seeds) & set(final_seeds):
        raise ValueError('Search and final GCN seeds must be disjoint')
    shared = {'teacher_kernel', 'gamma', 'T', 'basis', 'dropout', 'lr', 'weight_decay'}
    if set(common_space) != shared or set(risk_space) != {'B'} or set(grip_space) != {'kl_weight'}:
        raise ValueError('Specify all seven common parameters, risk B and GRIP kl_weight')
    output_dir = Path(output_dir)
    initial_configs = initial_configs or {}
    options = dict(data_dir=data_dir, search_seeds=search_seeds, final_seeds=final_seeds,
                   partition=dict(max_sweeps=max_sweeps, block_size=block_size),
                   grip_steps=grip_steps, epochs=epochs, eval_every=eval_every,
                   hidden=hidden, seed=seed, device=device, evaluate_test=False)
    selected = {}
    for method, specific in [('risk', risk_space), ('grip', grip_space)]:
        selected[method] = run_experiments(
            datasets, output_dir / 'search' / method, n_trials=n_trials,
            space={**common_space, **specific}, method=method,
            initial_configs=initial_configs.get(method, []), **options)
    tuned = pd.concat(selected.values(), ignore_index=True)
    tuned.to_csv(output_dir / 'tuned.csv', index=False)
    summaries, records = [], []
    for name, ratios in datasets.items():
        for ratio in ratios:
            best = {method: frame[(frame.dataset == name) & (frame.ratio == ratio)].iloc[0]
                    for method, frame in selected.items()}
            for source in ('risk', 'grip'):
                for method, coefficient in [('risk', 'B'), ('grip', 'kl_weight')]:
                    if source == method:
                        row = best[method]
                    else:
                        fixed = {k: [best[source][k]] for k in sorted(shared)}
                        fixed[coefficient] = [best[method][coefficient]]
                        row = run_experiments(
                            {name: [ratio]}, output_dir / 'cross' / source / method,
                            n_trials=1, space=fixed, method=method, **options).iloc[0]
                    summaries.append(dict(row, setting_source=source))
                    folder = Path(row['folder'])
                    trial = json.loads((folder / 'best.json').read_text())['trial']
                    runs = pd.read_csv(folder / f'final_trial_{trial}.csv')
                    records.extend(dict(dataset=name, ratio=ratio, setting_source=source,
                                        method=method, seed=int(r.seed), validation=100 * r.validation)
                                   for r in runs.itertuples())
    cross, runs = pd.DataFrame(summaries), pd.DataFrame(records)
    paired = []
    for (name, ratio), group in runs.groupby(['dataset', 'ratio']):
        wide = group.pivot(index='seed', columns=['setting_source', 'method'], values='validation')
        comparisons = [
            ('risk_settings', ('risk', 'grip'), ('risk', 'risk')),
            ('grip_settings', ('grip', 'grip'), ('grip', 'risk')),
            ('independently_tuned', ('grip', 'grip'), ('risk', 'risk')),
        ]
        for label, baseline, target in comparisons:
            delta = (wide[target] - wide[baseline]).dropna()
            half = float(t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else np.nan
            paired.append(dict(dataset=name, ratio=ratio, comparison=label, n=len(delta),
                               risk_minus_grip_pp=delta.mean(),
                               ci95_low=delta.mean() - half, ci95_high=delta.mean() + half))
    paired = pd.DataFrame(paired)
    cross.to_csv(output_dir / 'cross.csv', index=False)
    runs.to_csv(output_dir / 'cross_runs.csv', index=False)
    paired.to_csv(output_dir / 'paired.csv', index=False)
    return dict(tuned=tuned, cross=cross, paired=paired, runs=runs)
