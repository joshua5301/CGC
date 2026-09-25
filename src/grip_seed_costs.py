import subprocess
from pathlib import Path

import pandas as pd
import torch
from torch_geometric import seed_everything
from tqdm.auto import tqdm

from src.hyperparams import BEST_HYPERPARAMS_DICT
from src.initialization_study import _save_json
from src.partition import partition
from src.risk_analysis import _save_csv
from src.risk_experiment import _fingerprint, _prepare_dataset
from src.teacher import get_teacher_labels
from src.utils import BUDGET


def compare_costs(costs, reference_seed=1234, atol=1e-8, rtol=1e-6):
    ref = costs.loc[costs.partition_seed == reference_seed].iloc[0]
    costs = costs.copy()
    costs['eligible'] = costs.converged & costs.nodes.eq(costs.requested_nodes)
    reference_ok = bool(ref.converged and ref.nodes == ref.requested_nodes)
    tolerance = atol + rtol * abs(ref.final_J)
    costs['delta_J'] = costs.final_J - ref.final_J
    costs['delta_feature'] = costs.final_feature - ref.final_feature
    costs['delta_weighted_kl'] = costs.final_weighted_kl - ref.final_weighted_kl
    costs['comparison'] = 'excluded'
    eligible = costs.eligible & reference_ok
    costs.loc[eligible & (costs.delta_J < -tolerance), 'comparison'] = 'lower'
    costs.loc[eligible & (costs.delta_J.abs() <= tolerance), 'comparison'] = 'near_equal'
    costs.loc[eligible & (costs.delta_J > tolerance), 'comparison'] = 'higher'
    costs.loc[costs.partition_seed == reference_seed, 'comparison'] = 'reference'
    others = costs[costs.partition_seed != reference_seed]
    valid = others[others.eligible & reference_ok]
    summary = dict(dataset=ref.dataset, ratio=float(ref.ratio), reference_J=float(ref.final_J),
        reference_valid=reference_ok, tolerance=tolerance, other_seeds=len(others), comparable_seeds=len(valid),
        excluded_seeds=len(others) - len(valid), lower=int(valid.comparison.eq('lower').sum()),
        near_equal=int(valid.comparison.eq('near_equal').sum()), higher=int(valid.comparison.eq('higher').sum()),
        other_J_min=valid.final_J.min(), other_J_mean=valid.final_J.mean(),
        other_J_std=valid.final_J.std(), other_J_max=valid.final_J.max(),
        mean_delta_J=valid.delta_J.mean(), reference_rank=1 + int(valid.comparison.eq('lower').sum()) if reference_ok else None)
    return costs, summary


def measure_grip_seed_costs(datasets, output_dir, partition_seeds=tuple(range(100)),
                            reference_seed=1234, teacher_seed=0, basis=3000,
                            grip_steps=1000, configs=None, data_dir='/content/data/',
                            device='cuda', atol=1e-8, rtol=1e-6):
    if grip_steps < 1 or not partition_seeds or min(atol, rtol) < 0:
        raise ValueError('Require positive iteration limit, seeds and nonnegative tolerances')
    seeds = list(dict.fromkeys([reference_seed, *partition_seeds]))
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    root, device = Path(output_dir), torch.device(device)
    root.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frames, summaries = [], []
    for dataset, ratios in datasets.items():
        train, mask, validation, testing, H = _prepare_dataset(dataset, data_dir, device)
        for ratio in ratios:
            kernel, gamma, temperature, weight, _ = BEST_HYPERPARAMS_DICT[(dataset, ratio)]
            params = dict(teacher_kernel=kernel, gamma=gamma, T=temperature, kl_weight=weight, basis=basis)
            params.update((configs or {}).get((dataset, ratio), {}))
            teacher_protocol = dict(revision=revision, torch=str(torch.__version__), dataset=dataset,
                data_dir=str(data_dir), teacher_seed=teacher_seed,
                **{k: params[k] for k in ('teacher_kernel', 'gamma', 'T', 'basis')})
            teacher_path = root / f'teacher_{_fingerprint(teacher_protocol)}.pt'
            if teacher_path.exists():
                Q = torch.load(teacher_path, map_location=device, weights_only=True)
            else:
                seed_everything(teacher_seed)
                Q = get_teacher_labels(H, mask, train['y'], params['teacher_kernel'],
                    params['gamma'], params['T'], params['basis'])
                temporary = teacher_path.with_suffix('.tmp')
                torch.save(Q.cpu(), temporary)
                temporary.replace(teacher_path)
            m = BUDGET[(dataset, ratio)]
            protocol = dict(**teacher_protocol, ratio=ratio, params=params, budget=m,
                            grip_steps=grip_steps, init='kmeans')
            case = root / f'{dataset}_{ratio:g}_{_fingerprint(protocol)}'
            case.mkdir(exist_ok=True)
            _save_json(case / 'protocol.json', dict(**protocol, seeds=seeds,
                reference_seed=reference_seed, atol=atol, rtol=rtol))
            rows = []
            for seed in tqdm(seeds, desc=f'{dataset} {ratio:g} GRIP objective'):
                path = case / f'{seed}.pt'
                if path.exists():
                    result = torch.load(path, map_location='cpu', weights_only=True)
                else:
                    result = partition(H, Q, m, kl_weight=params['kl_weight'], iters=grip_steps,
                                       seed=seed, init='kmeans', return_diagnostics=True)
                    temporary = path.with_suffix('.tmp')
                    torch.save(result, temporary)
                    temporary.replace(path)
                rows.append(dict(dataset=dataset, ratio=ratio, partition_seed=seed,
                    requested_nodes=m, **params,
                    **{k: v for k, v in result.items() if not isinstance(v, torch.Tensor)}))
            costs, summary = compare_costs(pd.DataFrame(rows), reference_seed, atol, rtol)
            _save_csv(costs, case / 'costs.csv')
            frames.append(costs)
            summaries.append(summary)
            _save_csv(pd.concat(frames, ignore_index=True), root / 'costs.csv')
            _save_csv(pd.DataFrame(summaries), root / 'summary.csv')
        del train, mask, validation, testing, H, Q
        torch.cuda.empty_cache()
    return dict(costs=pd.concat(frames, ignore_index=True), summary=pd.DataFrame(summaries))
