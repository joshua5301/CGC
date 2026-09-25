import subprocess
from pathlib import Path

import pandas as pd
import torch
from torch_geometric import seed_everything
from tqdm.auto import tqdm

from src.initialization_study import _save_json
from src.risk_analysis import _save_csv
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student
from src.risk_partition import risk_partition
from src.teacher import get_teacher_labels
from src.utils import BUDGET


def compare_risk_initializations(configs, output_dir, partition_seeds=(0, 1, 2, 3, 4),
                                 student_seeds=tuple(range(200, 210)), teacher_seed=0,
                                 max_sweeps=100, split_random_directions=2, block_size=1024,
                                 data_dir='/content/data/', epochs=1000, eval_every=10,
                                 hidden=256, device='cuda', inits=('surrogate', 'split'), teacher_cache_dir=None):
    configs = pd.DataFrame(configs)
    if configs.empty or configs.duplicated(['dataset', 'ratio', 'B']).any():
        raise ValueError('Require unique dataset, ratio, B configurations')
    if not inits or len(set(inits)) != len(inits) or set(inits) - {'surrogate', 'split'}:
        raise ValueError('Invalid initialization methods')
    if not partition_seeds or not student_seeds or len(set(partition_seeds)) != len(partition_seeds) or len(set(student_seeds)) != len(student_seeds):
        raise ValueError('Require nonempty unique seeds')
    output_dir, device = Path(output_dir), torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_cache_dir = Path(teacher_cache_dir) if teacher_cache_dir is not None else output_dir
    teacher_cache_dir.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden, loss_weighting='uniform')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frames = []
    for dataset, group in configs.groupby('dataset', sort=False):
        train, mask, validation, testing, H = _prepare_dataset(dataset, data_dir, device)
        for config in group.to_dict('records'):
            ratio = float(config['ratio'])
            params = {k: config[k] for k in ('B', 'teacher_kernel', 'gamma', 'T', 'basis', 'dropout', 'lr', 'weight_decay')}
            params['basis'] = int(params['basis'])
            m = BUDGET[(dataset, ratio)]
            protocol = dict(revision=revision, torch=str(torch.__version__), dataset=dataset, ratio=ratio,
                            params=params, budget=m, partition_seeds=list(partition_seeds),
                            student_seeds=list(student_seeds), teacher_seed=teacher_seed,
                            max_sweeps=max_sweeps, split_random_directions=split_random_directions,
                            block_size=block_size, data_dir=str(data_dir), student=settings,
                            move_seed_offset=100000, evaluate_test=False, inits=list(inits))
            case = output_dir / f'{dataset}_{ratio:g}_{_fingerprint(protocol)}'
            case.mkdir(exist_ok=True)
            _save_json(case / 'protocol.json', protocol)
            teacher_protocol = {k: protocol[k] for k in ('revision', 'torch', 'dataset', 'data_dir', 'teacher_seed')}
            teacher_protocol.update({k: params[k] for k in ('teacher_kernel', 'gamma', 'T', 'basis')})
            teacher_path = teacher_cache_dir / f'teacher_{_fingerprint(teacher_protocol)}.pt'
            if teacher_path.exists():
                Q = torch.load(teacher_path, map_location=device, weights_only=True)
            else:
                seed_everything(teacher_seed)
                Q = get_teacher_labels(H, mask, train['y'], params['teacher_kernel'],
                                       params['gamma'], params['T'], params['basis'])
                temporary = teacher_path.with_suffix('.tmp')
                torch.save(Q.cpu(), temporary)
                temporary.replace(teacher_path)
            path = case / 'runs.csv'
            records = pd.read_csv(path).to_dict('records') if path.exists() else []
            jobs = [(ps, init) for ps in partition_seeds for init in inits]
            for ps, init in tqdm(jobs, desc=f'{dataset} {ratio:g} B={params["B"]:.4g}'):
                artifact_path = case / f'{init}_{ps}.pt'
                if artifact_path.exists():
                    artifact = torch.load(artifact_path, map_location='cpu', weights_only=True)
                else:
                    artifact = risk_partition(H, Q, m, params['B'], seed=ps, init=init,
                        move_seed=100000 + ps, max_sweeps=max_sweeps, block_size=block_size,
                        split_random_directions=split_random_directions, return_assignment=True)
                    temporary = artifact_path.with_suffix('.tmp')
                    torch.save(artifact, temporary)
                    temporary.replace(artifact_path)
                cx, cy = artifact['x'].to(device), artifact['y'].to(device)
                for ss in student_seeds:
                    if any(r['init'] == init and r['partition_seed'] == ps and r['student_seed'] == ss for r in records):
                        continue
                    val, _, epoch = _train_student(cx, cy, validation, params, ss, settings)
                    records.append(dict(dataset=dataset, ratio=ratio, B=params['B'], init=init, partition_seed=ps,
                        student_seed=ss, validation=100 * val, best_epoch=epoch, nodes=len(cx),
                        J_initial=artifact['history'][0], J_final=artifact['J'],
                        V_initial=artifact['initial_V'], E_initial=artifact['initial_moment_error'],
                        V_final=artifact['V'], E_final=artifact['moment_error'],
                        forced_splits=artifact.get('forced_splits', 0),
                        initialization_seconds=artifact['initialization_seconds'],
                        partition_seconds=artifact['seconds'], sweeps=artifact['sweeps'],
                        converged=artifact['converged']))
                    _save_csv(pd.DataFrame(records), path)
                del cx, cy
            frames.append(pd.DataFrame(records))
            del Q, artifact
        del train, mask, validation, testing, H
        torch.cuda.empty_cache()
    runs = pd.concat(frames, ignore_index=True)
    metrics = ['nodes', 'J_initial', 'J_final', 'V_initial', 'E_initial', 'V_final', 'E_final',
               'forced_splits', 'initialization_seconds', 'partition_seconds', 'sweeps', 'converged']
    per_seed = runs.groupby(['dataset', 'ratio', 'B', 'init', 'partition_seed'], as_index=False).agg(
        validation_mean=('validation', 'mean'), student_std=('validation', 'std'),
        **{k: (k, 'first') for k in metrics})
    summary = per_seed.groupby(['dataset', 'ratio', 'B', 'init'], as_index=False).agg(
        validation_mean=('validation_mean', 'mean'), partition_std=('validation_mean', 'std'),
        J_initial=('J_initial', 'mean'), J_final=('J_final', 'mean'),
        V_final=('V_final', 'mean'), E_final=('E_final', 'mean'),
        converged_fraction=('converged', 'mean'), initialization_seconds=('initialization_seconds', 'mean'))
    wide = per_seed.pivot(index=['dataset', 'ratio', 'B', 'partition_seed'], columns='init',
                          values=['validation_mean', 'J_initial', 'J_final'])
    paired = pd.DataFrame({f'delta_{metric}': wide[metric]['split'] - wide[metric]['surrogate']
                           for metric in ('validation_mean', 'J_initial', 'J_final')}).reset_index() if len(inits) == 2 else pd.DataFrame()
    for name, frame in dict(runs=runs, per_seed=per_seed, summary=summary, paired=paired).items():
        _save_csv(frame, output_dir / f'{name}.csv')
    return dict(runs=runs, per_seed=per_seed, summary=summary, paired=paired)
