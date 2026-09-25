import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from src.initialization_study import _save_json
from src.risk_analysis import _save_csv
from src.risk_experiment import _fingerprint
from src.risk_initialization_study import compare_risk_initializations


def select_best_b(summary, grid):
    joined = summary.merge(grid[['dataset', 'ratio', 'B', 'B_base', 'B_factor']],
                           on=['dataset', 'ratio', 'B'], validate='many_to_one')
    selected = joined.sort_values(['validation_mean', 'B_factor'], ascending=[False, True]).drop_duplicates(
        ['dataset', 'ratio', 'init']).copy()
    selected['grid_boundary'] = selected.B_factor.isin([grid.B_factor.min(), grid.B_factor.max()])
    return selected.sort_values(['dataset', 'ratio', 'init']).reset_index(drop=True)


def run_b_study(configs, output_dir, factors=(.25, .5, 1., 2., 4., 8.),
                search_partition_seeds=(0, 1, 2, 3, 4), search_student_seeds=(0, 1, 2),
                confirmation_partition_seeds=tuple(range(100, 110)),
                confirmation_student_seeds=tuple(range(300, 305)), **settings):
    configs = pd.DataFrame(configs)
    if configs.empty or configs.duplicated(['dataset', 'ratio']).any():
        raise ValueError('Require one baseline configuration per dataset and ratio')
    if not factors or len(set(factors)) != len(factors) or any(not np.isfinite(f) or f <= 0 for f in factors):
        raise ValueError('Require unique finite positive B factors')
    if (set(search_partition_seeds) & set(confirmation_partition_seeds) or
        set(search_student_seeds) & set(confirmation_student_seeds)):
        raise ValueError('Search and confirmation seeds must be disjoint in both stages')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    protocol = dict(revision=revision, configs=configs.to_dict('records'), factors=list(factors),
        search_partition_seeds=list(search_partition_seeds), search_student_seeds=list(search_student_seeds),
        confirmation_partition_seeds=list(confirmation_partition_seeds),
        confirmation_student_seeds=list(confirmation_student_seeds), settings=settings)
    output_dir = Path(output_dir)
    root = output_dir / _fingerprint(protocol)
    root.mkdir(parents=True, exist_ok=True)
    _save_json(root / 'protocol.json', protocol)
    grid = pd.DataFrame([dict(config, B=config['B'] * factor, B_base=config['B'], B_factor=factor)
                         for config in configs.to_dict('records') for factor in factors])
    search = compare_risk_initializations(grid, root / 'search', partition_seeds=search_partition_seeds,
                                         student_seeds=search_student_seeds, teacher_cache_dir=root / 'teachers', **settings)
    selected = select_best_b(search['summary'], grid)
    _save_csv(selected, root / 'selected.csv')
    confirmations, summaries = [], []
    for init in ('surrogate', 'split'):
        choices = selected[selected.init == init]
        chosen_configs = choices[['dataset', 'ratio', 'B']].merge(grid, on=['dataset', 'ratio', 'B'], validate='one_to_one')
        result = compare_risk_initializations(chosen_configs, root / f'confirmation_{init}',
            partition_seeds=confirmation_partition_seeds, student_seeds=confirmation_student_seeds,
            inits=(init,), teacher_cache_dir=root / 'teachers', **settings)
        confirmations.append(result['per_seed'])
        summaries.append(result['summary'])
    confirmation = pd.concat(confirmations, ignore_index=True)
    final = pd.concat(summaries, ignore_index=True).rename(columns={
        'validation_mean': 'confirmation_val', 'partition_std': 'confirmation_partition_std'})
    final = final.merge(selected[['dataset', 'ratio', 'init', 'B', 'B_factor', 'grid_boundary', 'validation_mean']].rename(
        columns={'validation_mean': 'search_val'}), on=['dataset', 'ratio', 'init', 'B'], validate='one_to_one')
    wide = confirmation.pivot(index=['dataset', 'ratio', 'partition_seed'], columns='init', values='validation_mean')
    paired = (wide['split'] - wide['surrogate']).rename('delta_val_pp').reset_index()
    paired_summary = paired.groupby(['dataset', 'ratio'], as_index=False).agg(
        delta_val_pp=('delta_val_pp', 'mean'), partition_delta_std=('delta_val_pp', 'std'),
        n=('delta_val_pp', 'size'))
    half = t.ppf(.975, paired_summary.n - 1) * paired_summary.partition_delta_std / np.sqrt(paired_summary.n)
    paired_summary['ci95_low'] = paired_summary.delta_val_pp - half
    paired_summary['ci95_high'] = paired_summary.delta_val_pp + half
    report = dict(search=search['summary'].merge(grid[['dataset', 'ratio', 'B', 'B_factor']],
                  on=['dataset', 'ratio', 'B'], validate='many_to_one'), selected=selected, final=final,
                  confirmation=confirmation, paired=paired, paired_summary=paired_summary)
    for name, frame in report.items():
        _save_csv(frame, root / f'{name}.csv')
    _save_json(output_dir / 'latest.json', dict(folder=str(root)))
    return dict(**report, folder=str(root))
