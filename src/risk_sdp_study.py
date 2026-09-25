import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric import seed_everything
from tqdm.auto import tqdm

from src.initialization_study import _save_json
from src.risk_analysis import _save_csv
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student
from src.risk_partition import risk_partition
from src.risk_sdp import solve_risk_sdp, round_risk_sdp, normalize_features, partition_value
from src.teacher import get_teacher_labels
from src.utils import BUDGET


def run_sdp_study(configs, output_dir, sample_nodes=256, sample_clusters=8, sample_seed=0,
                  partition_seeds=(0, 1, 2), teacher_seed=0, max_sweeps=100,
                  sdp=None, student_seeds=(), epochs=1000, eval_every=10, hidden=256,
                  data_dir='/content/data/', device='cuda', teacher_cache_dir=None):
    if not partition_seeds or len(set(partition_seeds)) != len(partition_seeds):
        raise ValueError('Require unique nonempty partition seeds')
    if sample_nodes is not None and (sample_nodes < 2 or not 1 <= sample_clusters <= sample_nodes):
        raise ValueError('Require 1 <= sample_clusters <= sample_nodes')
    if sample_nodes is not None and student_seeds:
        raise ValueError('Subset diagnostics do not report full-graph student accuracy; use sample_nodes=None')
    configs = pd.DataFrame(configs)
    if configs.empty or configs.duplicated(['dataset', 'ratio', 'B']).any():
        raise ValueError('Require unique nonempty dataset, ratio, B configurations')
    output_dir, device = Path(output_dir), torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    solver = dict(eps=1e-5, max_iters=10000, max_nodes=512, time_limit_secs=1800., safety=1e-8)
    solver.update(sdp or {})
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden, loss_weighting='uniform')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    all_runs, summaries = [], []
    for config in tqdm(configs.to_dict('records'), desc='Risk convex relaxation'):
        dataset, ratio = config['dataset'], float(config['ratio'])
        params = {k: config[k] for k in ('B', 'teacher_kernel', 'gamma', 'T', 'basis', 'dropout', 'lr', 'weight_decay')}
        params['basis'] = int(params['basis'])
        protocol = dict(revision=revision, torch=str(torch.__version__), config=config,
            sample_nodes=sample_nodes, sample_clusters=sample_clusters, sample_seed=sample_seed,
            partition_seeds=list(partition_seeds), teacher_seed=teacher_seed, max_sweeps=max_sweeps,
            solver=solver, student_seeds=list(student_seeds), student=settings, data_dir=str(data_dir))
        case = output_dir / f'{dataset}_{ratio:g}_{_fingerprint(protocol)}'
        case.mkdir(exist_ok=True)
        _save_json(case / 'protocol.json', protocol)
        train, mask, validation, testing, H = _prepare_dataset(dataset, data_dir, device)
        teacher_key = dict(revision=revision, dataset=dataset, data_dir=str(data_dir),
            torch=str(torch.__version__), seed=teacher_seed,
            **{k: params[k] for k in ('teacher_kernel', 'gamma', 'T', 'basis')})
        teacher_folder = Path(teacher_cache_dir) if teacher_cache_dir is not None else case
        teacher_folder.mkdir(parents=True, exist_ok=True)
        teacher_path = teacher_folder / f'teacher_{_fingerprint(teacher_key)}.pt'
        if teacher_path.exists():
            Q = torch.load(teacher_path, map_location=device, weights_only=True)
        else:
            seed_everything(teacher_seed)
            Q = get_teacher_labels(H, mask, train['y'], params['teacher_kernel'], params['gamma'], params['T'], params['basis'])
            temporary = teacher_path.with_suffix('.tmp')
            torch.save(Q.cpu(), temporary)
            temporary.replace(teacher_path)
        full_nodes = len(H)
        if sample_nodes is None:
            ids, m, scope = np.arange(full_nodes), BUDGET[(dataset, ratio)], 'full'
        else:
            if sample_nodes > full_nodes:
                raise ValueError('sample_nodes exceeds dataset size')
            ids = np.sort(np.random.default_rng(sample_seed).choice(full_nodes, sample_nodes, replace=False))
            m, scope = int(sample_clusters), 'subset'
        np.save(case / 'sample_indices.npy', ids)
        h, q = H[torch.as_tensor(ids, device=device)], Q[torch.as_tensor(ids, device=device)]
        h_np, q_np = h.double().cpu().numpy(), q.double().cpu().numpy()
        X = normalize_features(h_np)
        path = case / 'relaxation.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                relaxed = {k: saved[k].item() if saved[k].ndim == 0 else saved[k] for k in saved.files}
        else:
            relaxed = solve_risk_sdp(h_np, q_np, m, params['B'], **solver)
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **relaxed)
            temporary.replace(path)
        records = []
        for method in ('surrogate', 'sdp'):
            for seed in partition_seeds:
                artifact_path = case / f'{method}_{seed}.pt'
                if artifact_path.exists():
                    artifact = torch.load(artifact_path, map_location='cpu', weights_only=True)
                else:
                    state = None
                    if method == 'sdp':
                        assignment = round_risk_sdp(relaxed['Z'], m, seed)
                        state = dict(assignment=torch.as_tensor(assignment),
                            generator_state=torch.Generator(device=device).manual_seed(seed).get_state())
                    artifact = risk_partition(h, q, m, params['B'], seed=seed, initial_state=state,
                        move_seed=100000 + seed, max_sweeps=max_sweeps, return_assignment=True)
                    temporary = artifact_path.with_suffix('.tmp')
                    torch.save(artifact, temporary)
                    temporary.replace(artifact_path)
                exact = partition_value(X, q_np, artifact['assignment'].numpy(), params['B'])
                tolerance = 1e-7 * max(1., abs(exact))
                if abs(exact - artifact['J']) > tolerance or relaxed['lower_bound'] > exact + tolerance:
                    raise FloatingPointError('Partition objective or lower-bound consistency failed')
                records.append(dict(dataset=dataset, ratio=ratio, B=params['B'], scope=scope,
                    nodes=len(h), clusters=m, method=method, seed=seed, J_initial=artifact['history'][0],
                    J_final=exact, lower_bound=relaxed['lower_bound'], gap_upper=exact - relaxed['lower_bound'],
                    converged=artifact['converged'], partition_seconds=artifact['seconds']))
        frame = pd.DataFrame(records)
        _save_csv(frame, case / 'runs.csv')
        for method, group in frame.groupby('method', sort=False):
            best = group.loc[group.J_final.idxmin()].to_dict()
            evaluation_path = case / f'{method}_evaluation.csv'
            evaluation = pd.read_csv(evaluation_path).to_dict('records') if evaluation_path.exists() else []
            artifact = torch.load(case / f'{method}_{int(best["seed"])}.pt', weights_only=True)
            for seed in student_seeds:
                if any(row['seed'] == seed for row in evaluation):
                    continue
                val, _, epoch = _train_student(artifact['x'].to(device), artifact['y'].to(device),
                    validation, params, seed, settings)
                evaluation.append(dict(seed=seed, validation=100 * val, best_epoch=epoch))
                _save_csv(pd.DataFrame(evaluation), evaluation_path)
            values = [r['validation'] for r in evaluation]
            best.update(J_mean=float(group.J_final.mean()), rounding_trials=len(group),
                solver_status=relaxed['status'], relaxed_value=relaxed['relaxed_value'],
                sdp_seconds=relaxed['seconds'], row_residual=relaxed['row_residual'],
                trace_residual=relaxed['trace_residual'], psd_violation=relaxed['psd_violation'],
                nonnegative_violation=relaxed['nonnegative_violation'],
                dual_trace_shift=relaxed['dual_trace_shift'], bound_kind=relaxed['bound_kind'],
                validation_mean=float(np.mean(values)) if values else None,
                validation_std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                folder=str(case))
            summaries.append(best)
        all_runs.append(frame)
        _save_csv(pd.DataFrame(summaries), output_dir / 'summary.csv')
        del train, mask, validation, testing, H, Q, h, q, artifact, relaxed
        torch.cuda.empty_cache()
    runs, summary = pd.concat(all_runs, ignore_index=True), pd.DataFrame(summaries)
    _save_csv(runs, output_dir / 'runs.csv')
    return dict(runs=runs, summary=summary)
