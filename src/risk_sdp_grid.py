import json
import subprocess
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.initialization_study import _save_json
from src.risk_analysis import _save_csv
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student
from src.risk_sdp_study import run_sdp_study


def run_sdp_grid(dataset, ratio, grid, output_dir, partition_seeds=(0, 1, 2),
                 search_seeds=(0, 1, 2), final_seeds=tuple(range(100, 110)),
                 teacher_seed=0, max_sweeps=100, sdp=None, epochs=1000,
                 eval_every=10, hidden=256, data_dir='/content/data/', device='cuda',
                 methods=('sdp', 'surrogate')):
    partition_keys = ('teacher_kernel', 'gamma', 'T', 'basis', 'B')
    student_keys = ('dropout', 'lr', 'weight_decay')
    if set(grid) != set(partition_keys + student_keys) or any(
            not isinstance(v, (list, tuple)) or not v for v in grid.values()):
        raise ValueError('Provide nonempty lists for teacher_kernel, gamma, T, basis, B, dropout, lr, weight_decay')
    if not methods or len(set(methods)) != len(methods) or set(methods) - {'sdp', 'surrogate'}:
        raise ValueError('Invalid comparison methods')
    if not search_seeds or not final_seeds or set(search_seeds) & set(final_seeds):
        raise ValueError('Require disjoint nonempty search and final student seeds')
    if any(len(set(seeds)) != len(seeds) for seeds in (partition_seeds, search_seeds, final_seeds)):
        raise ValueError('Seeds must be unique within each stage')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    protocol = dict(dataset=dataset, ratio=ratio, grid=grid, revision=revision,
        torch=str(torch.__version__), partition_seeds=list(partition_seeds),
        search_seeds=list(search_seeds), final_seeds=list(final_seeds), teacher_seed=teacher_seed,
        max_sweeps=max_sweeps, sdp=sdp, epochs=epochs, eval_every=eval_every,
        hidden=hidden, data_dir=str(data_dir), methods=list(methods), loss_weighting='uniform')
    output_dir = Path(output_dir)
    root = output_dir / _fingerprint(protocol)
    root.mkdir(parents=True, exist_ok=True)
    _save_json(root / 'protocol.json', protocol)
    _save_json(output_dir / 'latest.json', dict(folder=str(root)))
    device = torch.device(device)
    train, mask, validation, testing, H = _prepare_dataset(dataset, data_dir, device)
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden, loss_weighting='uniform')
    partitions = [dict(zip(partition_keys, values)) for values in product(*(grid[k] for k in partition_keys))]
    students = [dict(zip(student_keys, values)) for values in product(*(grid[k] for k in student_keys))]
    rows = []
    progress = tqdm(partitions, desc=f'{dataset} {ratio:g} full SDP grid')
    for params in progress:
        config = dict(dataset=dataset, ratio=ratio, **params, **students[0])
        study = run_sdp_study([config], root / 'partitions' / _fingerprint(params),
            sample_nodes=None, partition_seeds=partition_seeds, teacher_seed=teacher_seed,
            max_sweeps=max_sweeps, sdp=sdp, student_seeds=(), data_dir=data_dir, device=device,
            teacher_cache_dir=root / 'teachers')
        for entry in study['summary'].to_dict('records'):
            method = entry['method']
            if method not in methods:
                continue
            artifact_path = Path(entry['folder']) / f'{method}_{int(entry["seed"])}.pt'
            artifact = torch.load(artifact_path, map_location=device, weights_only=True)
            for student in students:
                candidate = dict(**params, **student)
                key = _fingerprint(dict(method=method, **candidate))
                folder = root / 'validation' / key
                folder.mkdir(parents=True, exist_ok=True)
                values = []
                for seed in search_seeds:
                    path = folder / f'{seed}.json'
                    if path.exists():
                        record = json.loads(path.read_text(encoding='utf-8'))
                    else:
                        val, _, epoch = _train_student(artifact['x'], artifact['y'], validation,
                                                       candidate, seed, settings)
                        record = dict(validation=100 * val, best_epoch=epoch)
                        _save_json(path, record)
                    values.append(record['validation'])
                rows.append(dict(dataset=dataset, ratio=ratio, method=method, **candidate,
                    search_val=float(np.mean(values)), partition_seed=int(entry['seed']),
                    J_final=entry['J_final'], lower_bound=entry['lower_bound'], gap_upper=entry['gap_upper'],
                    solver_status=entry['solver_status'], sdp_seconds=entry['sdp_seconds'],
                    nodes=entry['clusters'], artifact=str(artifact_path)))
                _save_csv(pd.DataFrame(rows), root / 'search.csv')
        progress.set_postfix(best_val=max(r['search_val'] for r in rows))
    search = pd.DataFrame(rows)
    selected = search.loc[search.groupby('method', sort=False).search_val.idxmax()].reset_index(drop=True)
    _save_csv(selected, root / 'selected.csv')
    final_rows = []
    for entry in selected.to_dict('records'):
        artifact = torch.load(entry['artifact'], map_location=device, weights_only=True)
        folder = root / 'final' / entry['method']
        folder.mkdir(parents=True, exist_ok=True)
        values = []
        for seed in tqdm(final_seeds, desc=f'{entry["method"]} final evaluation'):
            path = folder / f'{seed}.json'
            if path.exists():
                record = json.loads(path.read_text(encoding='utf-8'))
            else:
                val, test, epoch = _train_student(artifact['x'], artifact['y'], validation,
                    entry, seed, settings, testing=testing)
                record = dict(seed=seed, validation=100 * val, test=100 * test, best_epoch=epoch)
                _save_json(path, record)
            values.append(record)
        final = pd.DataFrame(values)
        _save_csv(final, folder / 'runs.csv')
        entry.update(final_val=float(final.validation.mean()),
            final_val_std=float(final.validation.std(ddof=0)), test_mean=float(final.test.mean()),
            test_std=float(final.test.std(ddof=0)))
        final_rows.append(entry)
        _save_csv(pd.DataFrame(final_rows), root / 'summary.csv')
    return dict(search=search, selected=selected, summary=pd.DataFrame(final_rows), folder=str(root))
