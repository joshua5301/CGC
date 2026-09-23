"""Resumable Colab sweep; two-layer GCN with uniform or cell-size-weighted student CE.

python -u sweep_distance.py --preset pilot --output /content/drive/MyDrive/grip_distance
See docs/mpnn_evaluation.md for the default MPNN surrogate evaluation protocol.
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.data import Data

from src.dataloader import get_dataset, set_dataset
from src.hyperparams import BEST_HYPERPARAMS_DICT
from src.models import GCN
from src.partition import partition
from src.partition_distance import DistanceIdentity
from src.partition_mpnn import MPNNIdentity
from src.partition_robust import distance_radii, robust_partition
from src.teacher import get_kernel_features, fit_logistic
from src.utils import budget, conv_graph_multi, normalize_adj_sparse, model_training


def numbers(text, cast=float):
    return [cast(v) for v in text.split(',')]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def code_digest():
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in [Path(__file__), *sorted((root / 'src').glob('*.py'))]:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def data_digest(data):
    h = hashlib.sha256()
    for key in ['x', 'edge_index', 'y', 'train_mask', 'val_mask', 'test_mask']:
        value = getattr(data, key, None)
        if value is not None:
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def atomic_torch(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def normalized_copy(data):
    if data is None:
        return None
    result = data.clone()
    a = normalize_adj_sparse(data).coalesce()
    result.edge_index, result.edge_attr = a.indices(), a.values()
    return result


def summarize(output, manifest, repeats, dropouts):
    """Rank complete configurations only, using mean validation across fixed seeds."""
    rows = []
    for entry in manifest:
        for dropout in dropouts:
            runs = []
            for repeat in range(repeats):
                path = output / 'runs' / f"{entry['id']}_d{dropout:g}_r{repeat}.json"
                if path.exists():
                    runs.append(json.loads(path.read_text(encoding='utf-8')))
            if len(runs) != repeats:
                continue
            config = entry['config']
            rows.append(dict(id=entry['id'], dataset=config['dataset'], ratio=config['ratio'],
                             method=config['method'], gamma=config['gamma'], T=config['T'],
                             coefficient=config['coefficient'], rho=config['rho'], loss=config['loss'], radius_cap=config.get('radius_cap'), dropout=dropout,
                             val=100 * np.mean([r['val'] for r in runs]),
                             test=100 * np.mean([r['test'] for r in runs]),
                             test_std=100 * np.std([r['test'] for r in runs], ddof=1) if repeats > 1 else 0.,
                             imbalance_multiplier=runs[0]['imbalance_multiplier'],
                             repeats=repeats))
    rows.sort(key=lambda r: (r['dataset'], r['ratio'], r['method'], -r['val']))
    if rows:
        temp = output / 'summary.csv.tmp'
        with temp.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, output / 'summary.csv')
        best = {}
        for row in rows:
            best.setdefault((row['dataset'], row['ratio'], row['method']), row)
        atomic_json(output / 'best.json', list(best.values()))
        best_by_rho = {}
        for row in rows:
            best_by_rho.setdefault((row['dataset'], row['ratio'], row['method'], row['rho']), row)
        atomic_json(output / 'best_by_rho.json', list(best_by_rho.values()))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--cases', default='cora:0.052,citeseer:0.036')
    parser.add_argument('--methods', default='mpnn,raw,grip', help='mpnn: K=2; raw: K=0; grip: original baseline; robust: worst-label GRIP; distance: legacy GCN-specific objective')
    parser.add_argument('--gammas', default=None)
    parser.add_argument('--temperatures', default=None)
    parser.add_argument('--mus', default='0.1,0.3,1,3,10')
    parser.add_argument('--rhos', default='0.5', help='mpnn comparison weights; raw always uses rho=0')
    parser.add_argument('--baseline-kl', default='0.1,0.2,0.5,1,2')
    parser.add_argument('--dropouts', default='0.1,0.5,0.9')
    parser.add_argument('--robust-caps', default='0,0.05,0.1,0.2')
    parser.add_argument('--robust-floor', type=float, default=0.)
    parser.add_argument('--robust-radius-mode', choices=['distance', 'constant'], default='distance')
    parser.add_argument('--robust-label-steps', type=int, default=150)
    parser.add_argument('--student-loss', choices=['uniform', 'cell-size'], default='uniform')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0, help='fixed condensation/teacher seed; student seeds seed+repeat')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--outer-iters', type=int, default=20)
    parser.add_argument('--median-iters', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--raw-data-dir', default='/content/data/')
    parser.add_argument('--output', default='results/mpnn_identity')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; select an A100 runtime or pass --device cpu')
    if args.repeat < 1 or args.epoch < 1 or not 1 <= args.eval_every <= args.epoch or args.threads < 1:
        raise ValueError('invalid repeat/epoch/eval-every/threads')
    torch.set_num_threads(args.threads)
    import faiss
    faiss.omp_set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    methods = args.methods.split(',')
    if not methods or not set(methods) <= {'mpnn', 'raw', 'distance', 'grip', 'robust'}:
        raise ValueError('methods must be mpnn, raw, distance, grip and/or robust')
    caps = list(dict.fromkeys(numbers(args.robust_caps)))
    if not caps or any(not np.isfinite(c) or not 0 <= args.robust_floor <= c <= 2 for c in caps) or args.robust_label_steps < 1:
        raise ValueError('Invalid robust radius/label steps')
    dropouts = numbers(args.dropouts)
    rhos = list(dict.fromkeys(numbers(args.rhos)))
    if not rhos or any(not np.isfinite(r) or not 0 <= r <= 1 for r in rhos):
        raise ValueError('rhos must be finite and in [0,1]')
    if not dropouts or any(not 0 <= d < 1 for d in dropouts):
        raise ValueError('dropouts must lie in [0,1)')
    source = code_digest()
    output = Path(args.output).resolve()
    for name in ['teachers', 'condensed', 'runs']:
        (output / name).mkdir(parents=True, exist_ok=True)
    manifest = []
    print(f'output: {output}\nsource fingerprint: {source}', flush=True)
    for case in args.cases.split(','):
        dataset, ratio_text = case.split(':')
        ratio = float(ratio_text)
        kernel, default_gamma, default_t, _, _ = BEST_HYPERPARAMS_DICT[(dataset, ratio)]
        gammas = numbers(args.gammas) if args.gammas else ([.01, .1, 1.] if args.preset == 'full' else [default_gamma])
        temperatures = numbers(args.temperatures) if args.temperatures else ([.2, .5, 1., 2.] if args.preset == 'full' else [default_t])
        if any(g <= 0 for g in gammas) or any(t <= 0 for t in temperatures):
            raise ValueError('gamma and temperature must be positive')
        train_args = SimpleNamespace(dataset_name=dataset, ratio=ratio, device=args.device,
                                     raw_data_dir=str(Path(args.raw_data_dir).resolve()) + os.sep,
                                     n_dim=256, lr=.01, weight_decay=5e-4,
                                     epoch=args.epoch, eval_every=args.eval_every)
        seed_everything(args.seed)
        train_args, data, data_val, data_test = set_dataset(train_args, get_dataset(train_args))
        fingerprint = digest([data_digest(d) for d in [data, data_val, data_test] if d is not None])
        h0, _, h2 = conv_graph_multi(train_args, data)
        operator = normalize_adj_sparse(data).coalesce() if 'distance' in methods else None
        served = [normalized_copy(d) for d in [data, data_val, data_test]] if 'distance' in methods else None
        m = budget(train_args)
        teacher_features = None
        case_entries = []
        for gamma, temp, method in itertools.product(gammas, temperatures, methods):
            coefficients = numbers(args.baseline_kl if method in ('grip', 'robust') else args.mus)
            method_rhos = rhos if method == 'mpnn' else ([0.] if method == 'raw' else [None])
            for coefficient, rho, cap in itertools.product(coefficients, method_rhos, caps if method == 'robust' else [None]):
                if coefficient < 0:
                    raise ValueError('coefficients must be nonnegative')
                config = dict(source=source, data=fingerprint, dataset=dataset, ratio=ratio,
                              method=method, gamma=gamma, T=temp, coefficient=coefficient,
                              kernel=kernel, basis=3000, seed=args.seed, m=m,
                              outer_iters=args.outer_iters, median_iters=args.median_iters,
                              batch_size=args.batch_size, epoch=args.epoch, eval_every=args.eval_every,
                              student='GCN-2-256', lr=.01, weight_decay=5e-4, loss='cell-size-soft-CE' if args.student_loss == 'cell-size' else 'uniform-soft-CE',
                              objective=method, comparison_depth=0 if method == 'raw' else 2,
                              comparison_operator='(1-rho)I+rho P_closed' if method in ('mpnn', 'raw') else None,
                              rho=rho, radius_cap=cap, radius_floor=args.robust_floor if method == 'robust' else None,
                              radius_mode=args.robust_radius_mode if method == 'robust' else None,
                              robust_label_steps=args.robust_label_steps if method == 'robust' else None,
                              torch=str(torch.__version__), device=args.device, threads=args.threads)
                entry = dict(id=digest(config), config=config)
                case_entries.append(entry)
        manifest.extend(case_entries)
        atomic_json(output / 'manifest.json', manifest)
        print(f'\n{case}: {len(case_entries)} condensations, {len(case_entries)*len(dropouts)*args.repeat} student fits (completed fits skipped)', flush=True)
        for entry in case_entries:
            key, config = entry['id'], entry['config']
            gamma, temp, method, coefficient = (config[k] for k in ['gamma', 'T', 'method', 'coefficient'])
            rho_label = f" rho={config['rho']:g}" if config['rho'] is not None else ''
            if config['radius_cap'] is not None:
                rho_label += f" radius_cap={config['radius_cap']:g}"
            paths = [output / 'runs' / f'{key}_d{d:g}_r{r}.json' for d in dropouts for r in range(args.repeat)]
            if all(p.exists() for p in paths):
                print(f'SKIP completed {method} gamma={gamma:g} T={temp:g} coef={coefficient:g}{rho_label}', flush=True)
                continue
            condensed_path = output / 'condensed' / f'{key}.pt'
            if condensed_path.exists():
                artifact = torch.load(condensed_path, map_location='cpu', weights_only=True)
            else:
                teacher_key = digest(dict(source=source, data=fingerprint, kernel=kernel, basis=3000,
                                          gamma=gamma, seed=args.seed, torch=str(torch.__version__), device=args.device,
                                          threads=args.threads))
                teacher_path = output / 'teachers' / f'{teacher_key}.pt'
                if teacher_path.exists():
                    logits = torch.load(teacher_path, map_location=args.device, weights_only=True)
                else:
                    if teacher_features is None:
                        seed_everything(args.seed)
                        teacher_features = get_kernel_features(h2, kernel, 3000)
                    print(f'Fit teacher gamma={gamma:g}', flush=True)
                    y_train = torch.nn.functional.one_hot(data.y[data.train_mask], train_args.num_class).double()
                    w = fit_logistic(teacher_features[data.train_mask], y_train, gamma)
                    logits = teacher_features @ w
                    atomic_torch(teacher_path, logits.cpu())
                f = (logits / temp).softmax(1)
                seed_everything(args.seed)
                print(f'Condense {method} gamma={gamma:g} T={temp:g} coef={coefficient:g}{rho_label}', flush=True)
                start = time.perf_counter()
                diagnostics = {}
                if method == 'robust':
                    epsilon, radius_info = distance_radii(h2, data.train_mask, cap=config['radius_cap'],
                                                        floor=config['radius_floor'], mode=config['radius_mode'])
                    result = robust_partition(h2, f, m, epsilon, coefficient=coefficient,
                                              outer_iters=args.outer_iters, label_steps=args.robust_label_steps,
                                              batch_size=args.batch_size)
                    xc, yc, assignment = result['H_cond'], result['Y_cond'], result['assign']
                    history, diagnostics = result['history'], {**result['diagnostics'], **radius_info}
                elif method in ('mpnn', 'raw'):
                    result = MPNNIdentity(h0, data.edge_index, f, m, mu=coefficient, seed=args.seed,
                                          depth=2 if method == 'mpnn' else 0, rho=config['rho'],
                                          batch_size=args.batch_size, median_iters=args.median_iters).run(args.outer_iters)
                    xc, yc, assignment = result['H_cond'], result['Y_cond'], result['assign']
                    history, diagnostics = result['history'], result['diagnostics']
                elif method == 'distance':
                    result = DistanceIdentity(h0, operator, f, m, mu=coefficient, seed=args.seed,
                                              batch_size=args.batch_size, median_iters=args.median_iters).run(args.outer_iters)
                    xc, yc, assignment = result['H_cond'], result['Y_cond'], result['assign']
                    history = result['history']
                else:
                    xc, yc, assignment = partition(h2, f, m, coefficient)
                    history = []
                if args.device.startswith('cuda'):
                    torch.cuda.synchronize()
                artifact = dict(x=xc.float().cpu(), y=yc.float().cpu(), assign=assignment.cpu(),
                                config=config, history=history, diagnostics=diagnostics,
                                seconds=time.perf_counter()-start,
                                teacher_val=float((f.argmax(1)[data.val_mask] == data.y[data.val_mask]).float().mean())
                                if getattr(data, 'val_mask', None) is not None else None)
                if method == 'robust':
                    artifact['uncertainty_radius'] = epsilon.cpu()
                atomic_torch(condensed_path, artifact)
                del f, logits, xc, yc, assignment
                if method != 'grip':
                    del result
            count = len(artifact['x'])
            cell_counts = torch.bincount(artifact['assign'], minlength=count)
            sample_weight = cell_counts.to(args.device) if args.student_loss == 'cell-size' else None
            ids = torch.arange(count, device=args.device)
            graph = Data(x=artifact['x'].to(args.device), y=artifact['y'].to(args.device),
                         edge_index=torch.stack([ids, ids]), edge_attr=torch.ones(count, device=args.device),
                         train_mask=torch.ones(count, dtype=torch.bool, device=args.device))
            for dropout, repeat in itertools.product(dropouts, range(args.repeat)):
                path = output / 'runs' / f'{key}_d{dropout:g}_r{repeat}.json'
                if path.exists():
                    continue
                seed_everything(args.seed + repeat)
                # mpnn/raw/grip use IDENTICAL GCN construction and original evaluation graphs.
                # Only the legacy GCN-specific distance method uses precomputed normalization.
                model = GCN(data.num_features, 256, train_args.num_class, 2, dropout,
                            normalize=method != 'distance').to(args.device)
                evaluation_data = served if method == 'distance' else [data, data_val, data_test]
                print(f'Student {method} gamma={gamma:g} T={temp:g} coef={coefficient:g}{rho_label} dropout={dropout:g} seed={args.seed+repeat}', flush=True)
                val, test = model_training(model, train_args, evaluation_data[0], graph,
                                           evaluation_data[1], evaluation_data[2], sample_weight=sample_weight)
                atomic_json(path, dict(config=config, dropout=dropout, repeat=repeat,
                                       student_seed=args.seed+repeat, val=val, test=test,
                                       condensed_nodes=count, condensation_seconds=artifact['seconds'],
                                       imbalance_multiplier=float(count * cell_counts.max() / len(artifact['assign']))))
                del model
                summarize(output, manifest, args.repeat, dropouts)
            del graph, artifact
        del teacher_features, h0, h2, operator, served, data, data_val, data_test
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    summarize(output, manifest, args.repeat, dropouts)
    print(f'\nDone. Validation-ranked results: {output / "summary.csv"}', flush=True)
    if (output / 'best.json').exists():
        print((output / 'best.json').read_text(encoding='utf-8'), flush=True)


if __name__ == '__main__':
    main()
