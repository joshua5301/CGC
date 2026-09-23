"""Full T/mu/lambda sweep, fixed Cora gamma=.01 and GCN dropout=.9."""
import argparse
import csv
import hashlib
import json
import os
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
from src.partition_hybrid import HybridIdentity
from src.partition_ot import build_transition
from src.teacher import get_kernel_features, fit_logistic
from src.utils import budget, conv_graph_multi, model_training


TEMPERATURES = '0.1,0.2,0.5,1,2,5'
MUS = '0,0.1,0.2,0.5,1,2,5'
LAMBDAS = '0,0.0001,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10'


def grid(text, positive=False):
    values = sorted(set(float(v) for v in text.split(',')))
    if not values or not np.isfinite(values).all() or min(values) < 0 or (positive and min(values) == 0):
        raise ValueError('Grid values must be finite and nonnegative (temperatures strictly positive)')
    return values


def family(row):
    return 'grip' if row['method'] == 'grip' else ('zero' if row['lam'] == 0 else 'positive')


def choose(rows):
    """Group winners use validation only; deterministic ties prefer small lambda,T,mu."""
    result = {}
    for group in ('grip', 'zero', 'positive'):
        candidates = [r for r in rows if family(r) == group]
        if candidates:
            result[group] = min(candidates, key=lambda r: (-r['val'], r['lam'], r['T'], r['mu']))
    return result


def save_csv(path, rows):
    if not rows:
        return
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='/content/drive/MyDrive/GRIP_cora_hybrid_full')
    parser.add_argument('--raw-data-dir', default='/content/data/')
    parser.add_argument('--temperatures', default=TEMPERATURES)
    parser.add_argument('--mus', default=MUS)
    parser.add_argument('--lambdas', default=LAMBDAS)
    parser.add_argument('--selection-seeds', type=int, default=3)
    parser.add_argument('--confirmation-seeds', type=int, default=10, help='0 skips confirmation')
    parser.add_argument('--outer-iters', type=int, default=20)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    temps, mus = grid(args.temperatures, True), grid(args.mus)
    lambdas = sorted(set([0.]+grid(args.lambdas)))
    if (args.selection_seeds < 2 or (args.confirmation_seeds != 0 and args.confirmation_seeds < 2)
            or args.outer_iters < 1 or not 1 <= args.eval_every <= args.epoch):
        raise ValueError('Invalid training or seed settings')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.output).resolve()
    for folder in ('teachers', 'condensed', 'runs'):
        (out/folder).mkdir(parents=True, exist_ok=True)
    logfile = out/f'full_{datetime.now():%Y%m%d_%H%M%S}.log'
    total = len(temps)*len(mus)*(len(lambdas)+1)
    print('GPU:', torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU')
    print(f'Results: {out}\nLog: {logfile}')
    print(f'Fixed gamma=0.01, dropout=0.9; {total} settings, {total*args.selection_seeds} selection fits.')
    try:
        from IPython import get_ipython
        from IPython.display import display
        handle = display('Preparing full sweep...', display_id=True) if get_ipython() else None
    except ImportError:
        handle = None

    def progress(message):
        if handle:
            handle.update(message)
        else:
            print('\r'+message.ljust(115), end='', flush=True)

    train = SimpleNamespace(dataset_name='cora', ratio=.052, device=args.device,
        raw_data_dir=str(Path(args.raw_data_dir).resolve())+'/', n_dim=256, lr=.01,
        weight_decay=5e-4, epoch=args.epoch, eval_every=args.eval_every)
    manifest, summaries, runs = [], [], []
    with logfile.open('w', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            seed_everything(0)
            train, data, data_val, data_test = set_dataset(train, get_dataset(train))
            raw, _, h2 = conv_graph_multi(train, data)
        P, operator = build_transition(data.edge_index.cpu().numpy(), len(raw))
        base = dict(source=code_digest(), runner=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            data=data_digest(data), dataset='cora', ratio=.052, gamma=.01, seed=0,
            kernel='relu', basis=3000, torch=str(torch.__version__), device=args.device, threads=4)
        teacher_path = out/'teachers'/f'{digest(base)}.pt'
        if teacher_path.exists():
            logits = torch.load(teacher_path, map_location=args.device, weights_only=True)
        else:
            progress('Fitting shared teacher once...')
            with redirect_stdout(log), redirect_stderr(log):
                seed_everything(0)
                features = get_kernel_features(h2, 'relu', 3000)
                labels = torch.nn.functional.one_hot(data.y[data.train_mask], train.num_class).double()
                w = fit_logistic(features[data.train_mask], labels, .01)
                logits = features@w
                atomic_torch(teacher_path, logits.cpu())
                del features, w
        H = h2.double().cpu().numpy()
        center = geometric_medians(h2.double(), torch.zeros(len(h2), dtype=torch.long, device=args.device), 1)
        fs = float((h2.double()-center).norm(dim=1).mean().clamp_min(1e-12))
        smoothing = (raw.double()-h2.double()).norm(dim=1)
        diagnostics = dict(smoothing_mean=float(smoothing.mean()), smoothing_max=float(smoothing.max()),
            structure_operator=operator, risk_bound_certified=False)

        def fit(entry, artifact, seed, stage):
            conf, key = entry['student'], entry['id']
            position = f'setting {len(summaries)+1}/{total}, ' if stage == 'selection' else ''
            progress(f'{stage}: {position}T={conf["T"]:g} mu={conf["mu"]:g} '
                     f'{conf["method"]} lambda={conf["lam"]:g}, seed={seed}')
            path = out/'runs'/f'{key}_{seed}.json'
            if path.exists():
                result = json.loads(path.read_text(encoding='utf-8'))
                if result['config'] != conf or result['seed'] != seed:
                    raise ValueError(f'Cache mismatch: {path}')
            else:
                with redirect_stdout(log), redirect_stderr(log):
                    seed_everything(seed)
                    n = len(artifact['x'])
                    ids = torch.arange(n, device=args.device)
                    graph = Data(x=artifact['x'].float().to(args.device), y=artifact['y'].float().to(args.device),
                        edge_index=torch.stack([ids, ids]), edge_attr=torch.ones(n, device=args.device),
                        train_mask=torch.ones(n, dtype=torch.bool, device=args.device))
                    model = GCN(data.num_features, 256, train.num_class, 2, .9).to(args.device)
                    print(f'{key} {stage} seed={seed}', flush=True)
                    val, test = model_training(model, train, data, graph, data_val, data_test)
                    result = dict(config=conf, seed=seed, val=100*val, test=100*test)
                    atomic_json(path, result)
            row = dict(id=key, stage=stage, method=conf['method'], T=conf['T'], mu=conf['mu'],
                       lam=conf['lam'], seed=seed, val=result['val'], test=result['test'])
            runs.append(row)
            return row

        for T in temps:
            teacher = (logits/T).softmax(1).double().clamp_min(1e-12)
            teacher /= teacher.sum(1, keepdim=True)
            F = teacher.cpu().numpy()
            ks = float((teacher*(teacher.log()-teacher.mean(0).log())).sum(1).mean().clamp_min(1e-12))
            for mu in mus:
                grip_config = dict(**base, T=T, mu=mu, method='grip', lam=0., m=budget(train))
                grip_path = out/'condensed'/f'{digest(grip_config)}.pt'
                if grip_path.exists():
                    grip = torch.load(grip_path, map_location='cpu', weights_only=True)
                else:
                    progress(f'Condensing GRIP T={T:g}, mu={mu:g}...')
                    with redirect_stdout(log), redirect_stderr(log):
                        seed_everything(0)
                        x, y, a = partition(h2, teacher, budget(train), mu)
                        grip = dict(x=x.cpu(), y=y.cpu(), assign=a.cpu())
                        atomic_torch(grip_path, grip)
                for method, lam in [('grip', 0.)]+[('hybrid', x) for x in lambdas]:
                    conf = dict(**base, T=T, mu=mu, method=method, lam=lam, m=budget(train),
                        operator=operator, outer_iters=args.outer_iters if method == 'hybrid' else None)
                    path = grip_path if method == 'grip' else out/'condensed'/f'{digest(conf)}.pt'
                    if path.exists():
                        artifact = torch.load(path, map_location='cpu', weights_only=True)
                    else:
                        progress(f'Condensing T={T:g}, mu={mu:g}, lambda={lam:g}...')
                        with redirect_stdout(log), redirect_stderr(log):
                            alg = HybridIdentity(H, F, P, grip['assign'].numpy(), grip['x'].numpy(),
                                grip['y'].numpy(), fs, ks, lam, mu)
                            result = alg.run(args.outer_iters)
                            artifact = {k: torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v for k, v in result.items()}
                            atomic_torch(path, artifact)
                    alg = HybridIdentity(H, F, P, artifact['assign'].numpy(), artifact['x'].numpy(),
                        artifact['y'].numpy(), fs, ks, lam, mu)
                    student = dict(**conf, epoch=args.epoch, eval_every=args.eval_every, dropout=.9,
                        student='GCN-2-256', loss='uniform-soft-CE', lr=.01, weight_decay=5e-4,
                        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    entry = dict(id=digest(student), student=student, artifact=str(path),
                        diagnostics=dict(**alg.objective(), cells=len(artifact['x']), feature_scale=fs, kl_scale=ks))
                    manifest.append(entry)
                    atomic_json(out/'manifest.json', dict(entries=manifest, arguments=vars(args), diagnostics=diagnostics))
                    values = [fit(entry, artifact, seed, 'selection') for seed in range(args.selection_seeds)]
                    summaries.append(dict(id=entry['id'], method=method, T=T, mu=mu, lam=lam,
                        val=float(np.mean([v['val'] for v in values])),
                        test=float(np.mean([v['test'] for v in values])),
                        test_std=float(np.std([v['test'] for v in values], ddof=1)),
                        **entry['diagnostics']))
                    save_csv(out/'summary.csv', summaries)
                    save_csv(out/'student_runs.csv', runs)
        winners = choose(summaries)
        targets = dict(winners)
        if 'positive' in winners:
            best = winners['positive']
            targets['matched_zero'] = next(r for r in summaries if family(r) == 'zero'
                                           and r['T'] == best['T'] and r['mu'] == best['mu'])
        # Commit selection BEFORE fresh-seed fits. Never use test in selection.
        atomic_json(out/'selection.json', dict(winners={k: {f: r[f] for f in ('id', 'method', 'T', 'mu', 'lam', 'val')}
            for k, r in targets.items()}, seeds=list(range(args.selection_seeds)),
            criterion='mean validation; ties prefer smaller lambda,T,mu'))
        lookup = {entry['id']: entry for entry in manifest}
        for key in dict.fromkeys(row['id'] for row in targets.values()):
            entry = lookup[key]
            artifact = torch.load(entry['artifact'], map_location='cpu', weights_only=True)
            for seed in range(args.selection_seeds, args.selection_seeds+args.confirmation_seeds):
                fit(entry, artifact, seed, 'confirmation')
                save_csv(out/'student_runs.csv', runs)
    if handle:
        handle.update('Complete. Best validation settings and fresh-seed results below.')
    else:
        print()
    print('Validation selection: gamma=.01, dropout=.9 fixed')
    print('group          T      mu    lambda    val')
    for group, row in targets.items():
        print(f'{group:13s} {row["T"]:5g} {row["mu"]:7g} {row["lam"]:9g} {row["val"]:6.2f}')
    fresh = [r for r in runs if r['stage'] == 'confirmation']
    confirmation = []
    if fresh:
        by_id = {key: np.array([[r['val'], r['test']] for r in fresh if r['id'] == key])
                 for key in dict.fromkeys(r['id'] for r in fresh)}
        baseline = by_id[winners['grip']['id']][:, 1]
        print('Fresh seeds:    val +/- sd       test +/- sd      delta vs tuned GRIP [95% CI]')
        for group, row in targets.items():
            values = by_id[row['id']]
            mean, sd = values.mean(0), values.std(0, ddof=1)
            stat = paired_stats(values[:, 1], baseline)
            confirmation.append(dict(group=group, id=row['id'], val=float(mean[0]), val_std=float(sd[0]),
                test=float(mean[1]), test_std=float(sd[1]), versus_grip=stat))
            print(f'{group:13s} {mean[0]:.2f} +/- {sd[0]:.2f}   {mean[1]:.2f} +/- {sd[1]:.2f}   '
                  f'{stat["mean"]:+.2f} [{stat["low"]:+.2f}, {stat["high"]:+.2f}]')
        if 'positive' in winners:
            stat = paired_stats(by_id[winners['positive']['id']][:, 1], by_id[targets['matched_zero']['id']][:, 1])
            print(f'Positive vs matched lambda=0: {stat["mean"]:+.2f} [{stat["low"]:+.2f}, {stat["high"]:+.2f}] pp')
            confirmation.append(dict(group='positive_vs_matched_zero', paired=stat))
    atomic_json(out/'confirmation.json', dict(results=confirmation,
        caveat='Pointwise paired t intervals, fixed condensation/data; fresh student seeds only. No certified GCN risk bound.'))
    print('Full results: summary.csv | student_runs.csv | confirmation.json')


if __name__ == '__main__':
    main()
