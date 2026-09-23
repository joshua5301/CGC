"""Quiet, resumable Cora experiment: H2 feature distance + PS-S structure + KL.

Select lambda using seeds 0-2 validation; confirm on fresh student seeds 3-12.
GCN-2-256, identity condensed edges, uniform soft CE throughout.
"""
import argparse
import csv
import hashlib
import json
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.data import Data

from sweep_distance import atomic_json, atomic_torch, code_digest, data_digest, digest
from extend_robust import paired_stats
from src.dataloader import get_dataset, set_dataset
from src.models import GCN
from src.partition import partition, geometric_medians
from src.partition_hybrid import HybridIdentity, cut_cost
from src.partition_ot import build_transition
from src.teacher import get_kernel_features, fit_logistic
from src.utils import budget, conv_graph_multi, model_training


def choose_variant(rows):
    """Only selection-seed validation; tie: original GRIP, then smaller lambda."""
    names = sorted({r['variant'] for r in rows}, key=lambda x: (-1 if x == 'grip' else float(x.split('_')[1])))
    return max(names, key=lambda name: np.mean([r['val'] for r in rows if r['variant'] == name]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='/content/drive/MyDrive/GRIP_cora_hybrid_structure')
    p.add_argument('--raw-data-dir', default='/content/data/')
    p.add_argument('--lambdas', default='0,0.001,0.003,0.01,0.03,0.1,0.3,1')
    p.add_argument('--gamma', type=float, default=.01)
    p.add_argument('--temperature', type=float, default=2.)
    p.add_argument('--coefficient', type=float, default=.5)
    p.add_argument('--dropout', type=float, default=.9)
    p.add_argument('--selection-seeds', type=int, default=3)
    p.add_argument('--confirmation-seeds', type=int, default=10)
    p.add_argument('--outer-iters', type=int, default=20)
    p.add_argument('--epoch', type=int, default=1000)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    lambdas = sorted(set([0.] + [float(x) for x in args.lambdas.split(',')]))
    if (not np.isfinite(lambdas).all() or min(lambdas) < 0
            or not np.isfinite([args.gamma, args.temperature, args.coefficient, args.dropout]).all()
            or args.gamma <= 0 or args.temperature <= 0 or args.coefficient < 0
            or not 0 <= args.dropout < 1 or args.selection_seeds < 2
            or args.confirmation_seeds < 2 or args.outer_iters < 1
            or not 1 <= args.eval_every <= args.epoch):
        raise ValueError('Invalid grid or training configuration')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.output).resolve()
    for folder in ('teachers', 'condensed', 'runs'):
        (out/folder).mkdir(parents=True, exist_ok=True)
    log_path = out/f'hybrid_{datetime.now():%Y%m%d_%H%M%S}.log'
    print('GPU:', torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU')
    print(f'Results: {out}\nLog: {log_path}')
    try:
        from IPython import get_ipython
        from IPython.display import display
        handle = display('Preparing hybrid experiment...', display_id=True) if get_ipython() else None
    except ImportError:
        handle = None

    def progress(message):
        if handle:
            handle.update(message)
        else:
            print('\r'+message.ljust(110), end='', flush=True)

    train = SimpleNamespace(dataset_name='cora', ratio=.052, device=args.device,
        raw_data_dir=str(Path(args.raw_data_dir).resolve())+'/', n_dim=256,
        lr=.01, weight_decay=5e-4, epoch=args.epoch, eval_every=args.eval_every)
    rows, artifacts, manifest = [], {}, []
    with log_path.open('w', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            seed_everything(0)
            train, data, data_val, data_test = set_dataset(train, get_dataset(train))
            raw, _, h2 = conv_graph_multi(train, data)
        P, operator_meta = build_transition(data.edge_index.cpu().numpy(), len(raw))
        base = dict(source=code_digest(), runner=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            data=data_digest(data), dataset='cora', ratio=.052, gamma=args.gamma, T=args.temperature,
            kernel='relu', basis=3000, condensation_seed=0, torch=str(torch.__version__),
            device=args.device, threads=4, operator=operator_meta)
        teacher_path = out/'teachers'/f'{digest(base)}.pt'
        if teacher_path.exists():
            teacher = torch.load(teacher_path, map_location=args.device, weights_only=True)
        else:
            progress('Fitting teacher...')
            with redirect_stdout(log), redirect_stderr(log):
                seed_everything(0)
                features = get_kernel_features(h2, 'relu', 3000)
                labels = torch.nn.functional.one_hot(data.y[data.train_mask], train.num_class).double()
                w = fit_logistic(features[data.train_mask], labels, args.gamma)
                teacher = ((features@w)/args.temperature).softmax(1)
                # Match GRIP's target floor, with explicit probability normalization.
                teacher = teacher.double().clamp_min(1e-12)
                teacher /= teacher.sum(1, keepdim=True)
                atomic_torch(teacher_path, teacher.cpu())
                del features, w
        config = dict(**base, coefficient=args.coefficient, m=budget(train))
        grip_path = out/'condensed'/f'{digest(dict(**config, variant="grip"))}.pt'
        if grip_path.exists():
            grip = torch.load(grip_path, map_location='cpu', weights_only=True)
        else:
            progress('Condensing original GRIP baseline...')
            with redirect_stdout(log), redirect_stderr(log):
                seed_everything(0)
                x, y, a = partition(h2, teacher, budget(train), args.coefficient)
                center = geometric_medians(h2.double(), torch.zeros(len(h2), dtype=torch.long, device=args.device), 1)
                fs = float((h2.double()-center).norm(dim=1).mean().clamp_min(1e-12))
                ks = float((teacher*(teacher.log()-teacher.mean(0).log())).sum(1).mean().clamp_min(1e-12))
                grip = dict(x=x.cpu(), y=y.cpu(), assign=a.cpu(), feature_scale=fs, kl_scale=ks)
                atomic_torch(grip_path, grip)
        H, F = h2.double().cpu().numpy(), teacher.cpu().numpy()
        smoothing = (raw.double()-h2.double()).norm(dim=1)
        diagnostics = dict(smoothing_mean=float(smoothing.mean()), smoothing_max=float(smoothing.max()),
            raw_norm_mean=float(raw.double().norm(dim=1).mean()),
            feature_operator='existing GRIP symmetric-normalized adjacency, squared',
            structure_operator=operator_meta, risk_bound_certified=False)
        variants = [('grip', None)] + [(f'hybrid_{lam:g}', lam) for lam in lambdas]
        for name, lam in variants:
            conf = dict(**config, variant=name, structure_weight=lam,
                        outer_iters=args.outer_iters if lam is not None else None)
            path = grip_path if lam is None else out/'condensed'/f'{digest(conf)}.pt'
            if path.exists():
                artifact = torch.load(path, map_location='cpu', weights_only=True)
            else:
                progress(f'Condensing {name}...')
                with redirect_stdout(log), redirect_stderr(log):
                    alg = HybridIdentity(H, F, P, grip['assign'].numpy(), grip['x'].numpy(),
                        grip['y'].numpy(), grip['feature_scale'], grip['kl_scale'], lam, args.coefficient)
                    result = alg.run(args.outer_iters)
                    artifact = {k: torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v for k, v in result.items()}
                    atomic_torch(path, artifact)
            # Recompute diagnostics from the saved tensors, including baseline.
            alg = HybridIdentity(H, F, P, artifact['assign'].numpy(), artifact['x'].numpy(),
                artifact['y'].numpy(), grip['feature_scale'], grip['kl_scale'], lam or 0., args.coefficient)
            diagnostics[name] = dict(**alg.objective(), cells=len(artifact['x']),
                min_cell=int(alg.counts.min()), max_cell=int(alg.counts.max()),
                moved_from_grip=int((artifact['assign'] != grip['assign']).sum()))
            student = dict(**conf, epoch=args.epoch, eval_every=args.eval_every, dropout=args.dropout,
                lr=.01, weight_decay=5e-4, student='GCN-2-256', loss='uniform-soft-CE',
                artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            artifacts[name] = (artifact, student)
            manifest.append(dict(variant=name, artifact=str(path), config=student))
        atomic_json(out/'manifest.json', dict(entries=manifest, arguments=vars(args), diagnostics=diagnostics))

        def fit(name, seed, stage):
            progress(f'{stage}: {name}, seed {seed}')
            artifact, conf = artifacts[name]
            path = out/'runs'/f'{digest(conf)}_{seed}.json'
            if path.exists():
                result = json.loads(path.read_text(encoding='utf-8'))
                if result['config'] != conf or result['seed'] != seed:
                    raise ValueError(f'Cached run mismatch: {path}')
            else:
                with redirect_stdout(log), redirect_stderr(log):
                    seed_everything(seed)
                    n = len(artifact['x'])
                    ids = torch.arange(n, device=args.device)
                    graph = Data(x=artifact['x'].float().to(args.device), y=artifact['y'].float().to(args.device),
                        edge_index=torch.stack([ids, ids]), edge_attr=torch.ones(n, device=args.device),
                        train_mask=torch.ones(n, dtype=torch.bool, device=args.device))
                    model = GCN(data.num_features, 256, train.num_class, 2, args.dropout).to(args.device)
                    print(f'{name} seed={seed}', flush=True)
                    val, test = model_training(model, train, data, graph, data_val, data_test)
                    result = dict(config=conf, seed=seed, val=100*val, test=100*test)
                    atomic_json(path, result)
            rows.append(dict(stage=stage, variant=name, seed=seed, val=result['val'], test=result['test']))

        for name, _ in variants:
            for seed in range(args.selection_seeds):
                fit(name, seed, 'selection')
        selected = choose_variant(rows)
        atomic_json(out/'selection.json', dict(selected=selected, seeds=list(range(args.selection_seeds)),
            criterion='mean validation only; ties prefer GRIP then smaller lambda',
            rows=[{k: r[k] for k in ('variant', 'seed', 'val')} for r in rows]))
        for name in dict.fromkeys(['grip', 'hybrid_0', selected]):
            for seed in range(args.selection_seeds, args.selection_seeds+args.confirmation_seeds):
                fit(name, seed, 'confirmation')
    with (out/'student_runs.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['stage', 'variant', 'seed', 'val', 'test'])
        writer.writeheader()
        writer.writerows(rows)
    if handle:
        handle.update('Complete. Validation selection and fresh-seed confirmation below.')
    else:
        print()
    print('Selection (validation only):')
    for name, _ in variants:
        values = [r['val'] for r in rows if r['stage'] == 'selection' and r['variant'] == name]
        print(f'{name:16s} {np.mean(values):.2f}' + ('  selected' if name == selected else ''))
    fresh = [r for r in rows if r['stage'] == 'confirmation']
    baseline = [r['test'] for r in fresh if r['variant'] == 'grip']
    zero = [r['test'] for r in fresh if r['variant'] == 'hybrid_0']
    summary = []
    print('Fresh seeds:       val +/- sd       test +/- sd       delta vs GRIP [95% CI]')
    for name in dict.fromkeys(r['variant'] for r in fresh):
        values = np.array([[r['val'], r['test']] for r in fresh if r['variant'] == name])
        mean, sd = values.mean(0), values.std(0, ddof=1)
        stats = paired_stats(values[:, 1], baseline)
        summary.append(dict(variant=name, val=float(mean[0]), val_std=float(sd[0]), test=float(mean[1]),
            test_std=float(sd[1]), versus_grip=stats, versus_hybrid_zero=paired_stats(values[:, 1], zero)))
        print(f'{name:16s} {mean[0]:.2f} +/- {sd[0]:.2f}   {mean[1]:.2f} +/- {sd[1]:.2f}   '
              f'{stats["mean"]:+.2f} [{stats["low"]:+.2f}, {stats["high"]:+.2f}]')
    if selected not in ('grip', 'hybrid_0'):
        s = next(r for r in summary if r['variant'] == selected)['versus_hybrid_zero']
        print(f'Selected vs lambda=0: {s["mean"]:+.2f} [{s["low"]:+.2f}, {s["high"]:+.2f}] pp')
    atomic_json(out/'summary.json', dict(selected=selected, results=summary, diagnostics=diagnostics,
        caveat='Pointwise paired t intervals across fresh student seeds; fixed condensation/data. No certified GCN risk bound.'))
    print('Full results: student_runs.csv | summary.json (seed CIs conditional on fixed condensation)')


if __name__ == '__main__':
    main()
