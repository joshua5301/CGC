"""Cora GRIP: full teacher, held-out teacher, and directional residual correction."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import faiss
from torch_geometric import seed_everything
from torch_geometric.data import Data
from sweep_distance import atomic_json, atomic_torch, code_digest, data_digest, digest
from extend_robust import paired_stats
from src.dataloader import get_dataset, set_dataset
from src.models import GCN
from src.partition import partition
from src.teacher import get_kernel_features, fit_logistic
from src.utils import budget, conv_graph_multi, model_training
from src.residual_teacher import split_training_indices, residual_correction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='/content/drive/MyDrive/GRIP_cora_residual_teacher')
    parser.add_argument('--raw-data-dir', default='/content/data/')
    parser.add_argument('--slopes', default='0,0.5,2,8')
    parser.add_argument('--calibration-fraction', type=float, default=.5)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--delta', type=float, default=.05)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--seed-start', type=int, default=3)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    slopes = list(dict.fromkeys(float(s) for s in args.slopes.split(',')))
    if (not slopes or any(not np.isfinite(s) or s < 0 for s in slopes)
            or args.repeat < 2 or args.seed_start < 0 or not 1 <= args.eval_every <= args.epoch
            or not 0 < args.calibration_fraction < 1 or not 0 < args.delta < 1):
        raise ValueError('Invalid slope, split or training configuration')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.output).resolve()
    for folder in ['teachers', 'condensed', 'runs']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    log_path = out / f'residual_{datetime.now():%Y%m%d_%H%M%S}.log'
    print('GPU:', torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU')
    print(f'Results: {out}\nLog: {log_path}')
    try:
        from IPython import get_ipython
        from IPython.display import display
        handle = display('Preparing residual correction...', display_id=True) if get_ipython() else None
    except ImportError:
        handle = None

    def progress(message):
        if handle:
            handle.update(message)
        else:
            print('\r'+message.ljust(100), end='', flush=True)

    train_args = SimpleNamespace(dataset_name='cora', ratio=.052, device=args.device,
        raw_data_dir=str(Path(args.raw_data_dir).resolve())+'/', n_dim=256,
        lr=.01, weight_decay=5e-4, epoch=args.epoch, eval_every=args.eval_every)
    with log_path.open('w', encoding='utf-8') as logfile:
        with redirect_stdout(logfile), redirect_stderr(logfile):
            seed_everything(0)
            train_args, data, data_val, data_test = set_dataset(train_args, get_dataset(train_args))
            _, _, h2 = conv_graph_multi(train_args, data)
        fit_indices, cal_indices = split_training_indices(data.train_mask, args.calibration_fraction, args.split_seed)
        fit_indices, cal_indices = fit_indices.to(args.device), cal_indices.to(args.device)
        base = dict(source=code_digest(), runner=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    data=data_digest(data), dataset='cora', ratio=.052, gamma=.01, T=2., kernel='relu', basis=3000,
                    calibration_fraction=args.calibration_fraction, split_seed=args.split_seed,
                    torch=str(torch.__version__), device=args.device, threads=4)
        teacher_path = out/'teachers'/f'{digest(base)}.pt'
        if teacher_path.exists():
            teachers = torch.load(teacher_path, map_location=args.device, weights_only=True)
        else:
            progress('Fitting full and held-out teachers...')
            with redirect_stdout(logfile), redirect_stderr(logfile):
                seed_everything(0)
                kernel_features = get_kernel_features(h2, 'relu', 3000)
                teachers = {}
                for name, indices in [('full_teacher', data.train_mask.nonzero().flatten()), ('split_teacher', fit_indices)]:
                    labels = torch.nn.functional.one_hot(data.y[indices], train_args.num_class).double()
                    w = fit_logistic(kernel_features[indices], labels, .01)
                    teachers[name] = ((kernel_features@w)/2.).softmax(1).cpu()
                teachers.update(fit_indices=fit_indices.cpu(), calibration_indices=cal_indices.cpu())
                atomic_torch(teacher_path, teachers)
                del kernel_features, w
        variants = [('full_teacher', None), ('split_teacher', None)] + [(f'residual_{s:g}', s) for s in slopes]
        rows, diagnostics, manifest = [], {}, []
        for name, slope in variants:
            config = dict(**base, variant=name, slope=slope, delta=args.delta, coefficient=.5,
                          condensation_seed=0, m=budget(train_args))
            artifact_path = out/'condensed'/f'{digest(config)}.pt'
            if artifact_path.exists():
                artifact = torch.load(artifact_path, map_location='cpu', weights_only=True)
            else:
                progress(f'Condensing {name}...')
                with redirect_stdout(logfile), redirect_stderr(logfile):
                    if slope is None:
                        target = teachers[name].to(args.device)
                        correction = None
                        diag = {}
                    else:
                        correction = residual_correction(h2, teachers['split_teacher'].to(args.device),
                            cal_indices, data.y[cal_indices], slope=slope, delta=args.delta)
                        target = correction['probabilities']
                        diag = correction['diagnostics']
                    # Validation metrics are recorded only AFTER constructing the target.
                    val_prob = target[data.val_mask]
                    val_labels = data.y[data.val_mask]
                    diag = dict(**diag, teacher_val_acc=float((val_prob.argmax(1) == val_labels).double().mean()),
                        teacher_val_nll=float(-val_prob.clamp_min(1e-12).log().gather(1, val_labels[:, None]).mean()))
                    seed_everything(0)
                    x, y, assignment = partition(h2, target, budget(train_args), .5)
                    artifact = dict(config=config, x=x.float().cpu(), y=y.float().cpu(), assign=assignment.cpu(),
                        target=target.cpu(), diagnostics=diag, fit_indices=fit_indices.cpu(), calibration_indices=cal_indices.cpu())
                    if correction is not None:
                        artifact.update(chosen_k=correction['chosen_k'].cpu(), conditional_bound=correction['conditional_bound'].cpu())
                    atomic_torch(artifact_path, artifact)
                    del target, correction, x, y, assignment
            diagnostics[name] = artifact['diagnostics']
            count = len(artifact['x'])
            ids = torch.arange(count, device=args.device)
            graph = Data(x=artifact['x'].to(args.device), y=artifact['y'].to(args.device),
                edge_index=torch.stack([ids, ids]), edge_attr=torch.ones(count, device=args.device),
                train_mask=torch.ones(count, dtype=torch.bool, device=args.device))
            student_config = dict(**config, epoch=args.epoch, eval_every=args.eval_every,
                dropout=.9, lr=.01, weight_decay=5e-4, student='GCN-2-256', loss='uniform-soft-CE',
                artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest())
            key = digest(student_config)
            manifest.append(dict(id=key, config=student_config, artifact=str(artifact_path)))
            atomic_json(out/'manifest.json', dict(entries=manifest, seed_start=args.seed_start, repeat=args.repeat))
            for seed in range(args.seed_start, args.seed_start+args.repeat):
                progress(f'Student {name}, seed {seed} ({len(rows)+1}/{len(variants)*args.repeat})')
                path = out/'runs'/f'{key}_{seed}.json'
                if path.exists():
                    row = json.loads(path.read_text(encoding='utf-8'))
                else:
                    seed_everything(seed)
                    model = GCN(data.num_features, 256, train_args.num_class, 2, .9).to(args.device)
                    with redirect_stdout(logfile), redirect_stderr(logfile):
                        print(f'{name} seed={seed}', flush=True)
                        val, test = model_training(model, train_args, data, graph, data_val, data_test)
                    row = dict(variant=name, seed=seed, val=100*val, test=100*test)
                    atomic_json(path, row)
                    del model
                rows.append(row)
            del graph, artifact
    if handle:
        handle.update('Complete. Fixed-configuration comparisons below.')
    else:
        print()
    import csv
    with (out/'student_runs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['variant', 'seed', 'val', 'test'])
        writer.writeheader()
        writer.writerows(rows)
    split_test = [r['test'] for r in rows if r['variant'] == 'split_teacher']
    full_test = [r['test'] for r in rows if r['variant'] == 'full_teacher']
    summary = []
    print('variant          val +/- sd       test +/- sd      delta vs split [95% CI]')
    for name, slope in variants:
        subset = [r for r in rows if r['variant'] == name]
        values = np.array([[r['val'], r['test']] for r in subset])
        mean, std = values.mean(0), values.std(0, ddof=1)
        paired = paired_stats(values[:, 1], split_test)
        summary.append(dict(variant=name, val=float(mean[0]), test=float(mean[1]), test_std=float(std[1]),
                            versus_split=paired, versus_full=paired_stats(values[:, 1], full_test)))
        print(f'{name:16s} {mean[0]:.2f} +/- {std[0]:.2f}   {mean[1]:.2f} +/- {std[1]:.2f}   '
              f'{paired["mean"]:+.2f} [{paired["low"]:+.2f}, {paired["high"]:+.2f}]')
    print('Correction diagnostics: assumed slope; bound coverage is NOT certified.')
    for name, slope in variants:
        if slope is not None:
            d = diagnostics[name]
            print(f'{name}: k={d["k_counts"]}; nonvacuous bound={100*d["nonvacuous_fraction"]:.1f}%')
    atomic_json(out/'summary.json', dict(results=summary, diagnostics=diagnostics,
        caveat='Pointwise seed intervals, fixed split/condensation; assumed residual smoothness, not verified coverage.'))
    print('Full results: student_runs.csv | summary.json')


if __name__ == '__main__':
    main()
