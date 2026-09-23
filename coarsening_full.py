"""Cora .052 gamma/T/mu sweep with fixed GCN dropout .9 and uniform soft CE."""
import argparse
import hashlib
import itertools
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

from coarsening_grip import train_best, measure
from extend_robust import paired_stats
from hybrid_full import grid, save_csv
from sweep_distance import atomic_json, atomic_torch, code_digest, data_digest, digest
from src.coarsening_features import fixed_coarsening, optimize_features, feature_diagnostics, gcn_operator
from src.dataloader import get_dataset, set_dataset
from src.models import GCN
from src.partition import partition
from src.teacher import get_kernel_features, fit_logistic
from src.utils import budget, conv_graph_multi


GAMMAS = '0.001,0.01,0.1,1'
TEMPERATURES = '0.1,0.2,0.5,1,2,5,10'
MUS = '0,0.1,0.2,0.5,1,2,5,10'
VARIANTS = ('grip', 'coarse_mean', 'coarse_optimized')
COMPARISONS = (
    ('mean_vs_grip', 'coarse_mean', 'grip'),
    ('optimized_vs_grip', 'coarse_optimized', 'grip'),
    ('optimized_vs_matched_mean', 'coarse_optimized', 'mean_at_optimized'),
    ('matched_optimized_vs_mean', 'optimized_at_mean', 'coarse_mean'),
)


def choose(rows):
    """Validation-only selection; include both directions of the matched ablation."""
    winners = {name: min((r for r in rows if r['variant'] == name),
                        key=lambda r: (-r['val'], r['gamma'], r['T'], r['mu']))
               for name in VARIANTS}
    targets = dict(winners)
    for label, selected, variant in (
            ('mean_at_optimized', 'coarse_optimized', 'coarse_mean'),
            ('optimized_at_mean', 'coarse_mean', 'coarse_optimized')):
        best = winners[selected]
        targets[label] = next(r for r in rows if r['variant'] == variant
                             and all(r[k] == best[k] for k in ('gamma', 'T', 'mu')))
    return targets


def paired_values(runs, key, seeds):
    """Pair by seed explicitly, independently of the order cached runs were read."""
    selected = [r for r in runs if r['stage'] == 'confirmation' and r['id'] == key]
    indexed = {r['seed']: r for r in selected}
    if len(indexed) != len(selected) or set(indexed) != set(seeds):
        raise ValueError(f'Missing or duplicate confirmation seeds: {key}')
    return np.array([[indexed[s]['val'], indexed[s]['test']] for s in seeds])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='/content/drive/MyDrive/GRIP_cora_coarsening_full')
    p.add_argument('--raw-data-dir', default='/content/data/')
    p.add_argument('--gammas', default=GAMMAS)
    p.add_argument('--temperatures', default=TEMPERATURES)
    p.add_argument('--mus', default=MUS)
    p.add_argument('--selection-seeds', type=int, default=3)
    p.add_argument('--confirmation-seeds', type=int, default=10, help='0 skips confirmation')
    p.add_argument('--confirmation-start', type=int, default=13)
    p.add_argument('--epoch', type=int, default=1000)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--feature-steps', type=int, default=1000)
    p.add_argument('--feature-tolerance', type=float, default=1e-3)
    p.add_argument('--smooth-relative', type=float, default=1e-4)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    gammas, temps, mus = grid(args.gammas, True), grid(args.temperatures, True), grid(args.mus)
    if (args.selection_seeds < 2 or (args.confirmation_seeds != 0 and args.confirmation_seeds < 2)
            or args.confirmation_start < args.selection_seeds
            or not 1 <= args.eval_every <= args.epoch or args.feature_steps < 1
            or not np.isfinite([args.feature_tolerance, args.smooth_relative]).all()
            or min(args.feature_tolerance, args.smooth_relative) <= 0):
        p.error('Invalid solver/epoch settings or overlapping selection/confirmation seeds')
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
    total = len(gammas)*len(temps)*len(mus)*len(VARIANTS)
    print('GPU:', torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU')
    print(f'Results: {out}\nLog: {logfile}')
    print(f'Cora .052, dropout=.9; {total} settings, {total*args.selection_seeds} selection fits.')
    try:
        from IPython import get_ipython
        from IPython.display import display
        handle = display('Preparing coarsening sweep...', display_id=True) if get_ipython() else None
    except ImportError:
        handle = None

    def progress(message):
        if handle:
            handle.update(message)
        else:
            print('\r'+message.ljust(130), end='', flush=True)

    train = SimpleNamespace(dataset_name='cora', ratio=.052, device=args.device,
        raw_data_dir=str(Path(args.raw_data_dir).resolve())+'/', n_dim=256,
        lr=.01, weight_decay=5e-4, epoch=args.epoch, eval_every=args.eval_every)
    selection_seeds = list(range(args.selection_seeds))
    confirm_seeds = list(range(args.confirmation_start, args.confirmation_start+args.confirmation_seeds))
    entries, summaries, runs = [], [], []
    with logfile.open('w', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            seed_everything(0)
            train, data, _, _ = set_dataset(train, get_dataset(train))
            _, _, h2 = conv_graph_multi(train, data)
        # Include imported runner/statistics helpers as well as the core src digest.
        helper_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in ('coarsening_full.py', 'coarsening_grip.py',
                                      'hybrid_full.py', 'extend_robust.py')}
        base = dict(source=code_digest(), helpers=helper_hashes, data=data_digest(data),
            dataset='cora', ratio=.052, kernel='relu', basis=3000, condensation_seed=0,
            torch=str(torch.__version__), device=args.device, threads=4)
        P = gcn_operator(data.edge_index, data.edge_attr, len(data.x)).to(args.device)
        target = torch.sparse.mm(P, data.x.double())
        omega = torch.sparse.sum(P, dim=0).to_dense()
        radius = float(data.x.double().norm(dim=1).max())
        teacher_paths = {}
        features = None
        for gamma in gammas:
            path = out/'teachers'/f'{digest(dict(**base, gamma=gamma))}.pt'
            teacher_paths[gamma] = path
            if not path.exists():
                progress(f'Teacher gamma={gamma:g}...')
                with redirect_stdout(log), redirect_stderr(log):
                    seed_everything(0)
                    if features is None:
                        features = get_kernel_features(h2, 'relu', 3000)
                    labels = torch.nn.functional.one_hot(data.y[data.train_mask], train.num_class).double()
                    w = fit_logistic(features[data.train_mask], labels, gamma)
                    atomic_torch(path, (features@w).cpu())
                    del w
        del features

        def probabilities(gamma, temperature):
            logits = torch.load(teacher_paths[gamma], map_location=args.device, weights_only=True)
            teacher = (logits/temperature).softmax(1).clamp_min(1e-12)
            return teacher/teacher.sum(1, keepdim=True)

        def fit(entry, artifact, teacher, seed, stage):
            conf, key = entry['config'], entry['id']
            progress(f'{stage} {len(summaries)+1 if stage == "selection" else ""}'
                     f'{"/"+str(total) if stage == "selection" else ""}: '
                     f'g={conf["gamma"]:g} T={conf["T"]:g} mu={conf["mu"]:g} '
                     f'{conf["variant"]}, seed={seed}')
            path = out/'runs'/f'{key}_{seed}.json'
            if path.exists():
                result = json.loads(path.read_text(encoding='utf-8'))
                if result['config'] != conf or result['seed'] != seed:
                    raise ValueError(f'Cache mismatch: {path}')
            else:
                with redirect_stdout(log), redirect_stderr(log):
                    seed_everything(seed)
                    m = len(artifact['x'])
                    graph = Data(x=artifact['x'].float().to(args.device), y=artifact['y'].float().to(args.device),
                        edge_index=artifact['edge_index'].to(args.device),
                        edge_attr=artifact['edge_weight'].float().to(args.device),
                        train_mask=torch.ones(m, dtype=torch.bool, device=args.device))
                    model = GCN(data.num_features, 256, train.num_class, 2, .9).to(args.device)
                    print(f'{key} {stage} seed={seed}', flush=True)
                    val, test, epoch = train_best(model, train, data, graph)
                    diag = None
                    if stage == 'confirmation' and epoch is not None:
                        Q = gcn_operator(graph.edge_index, graph.edge_attr, m).to_dense().to(args.device)
                        diag = measure(model, data, graph, artifact['assign'], teacher, artifact['y'], P, Q)
                    result = dict(config=conf, seed=seed, val=100*val, test=100*test,
                                  best_epoch=epoch, diagnostics=diag)
                    atomic_json(path, result)
                    del model, graph
            row = dict(id=key, stage=stage, variant=conf['variant'], gamma=conf['gamma'],
                       T=conf['T'], mu=conf['mu'], seed=seed, val=result['val'], test=result['test'])
            runs.append(row)
            save_csv(out/'student_runs.csv', runs)
            return row

        for gamma, T in itertools.product(gammas, temps):
            teacher = probabilities(gamma, T)
            for mu in mus:
                config = dict(**base, gamma=gamma, T=T, mu=mu, m=budget(train),
                    feature_steps=args.feature_steps, tolerance=args.feature_tolerance,
                    smooth_relative=args.smooth_relative, radius=radius)
                path = out/'condensed'/f'{digest(config)}.pt'
                if path.exists():
                    bundle = torch.load(path, map_location='cpu', weights_only=True)
                else:
                    progress(f'Condensing g={gamma:g} T={T:g} mu={mu:g}...')
                    with redirect_stdout(log), redirect_stderr(log):
                        seed_everything(0)
                        x, y, a = partition(h2, teacher, budget(train), mu)
                        ops = fixed_coarsening(data.edge_index, data.edge_attr, len(data.x), a)
                        m = len(x)
                        mean = torch.zeros((m, data.num_features), dtype=torch.float64, device=args.device)
                        mean.index_add_(0, a, data.x.double())
                        mean /= ops['counts'].to(args.device)[:, None]
                        result = optimize_features(target, ops['Q'].to(args.device), a, omega, mean, radius,
                            args.feature_steps, args.feature_tolerance, args.smooth_relative)
                        fitted = result.pop('x').cpu()
                        ids = torch.arange(m)
                        shared = dict(y=y.cpu(), assign=a.cpu())
                        coarse_edges = dict(edge_index=ops['edge_index'], edge_weight=ops['edge_weight'])
                        artifacts = dict(
                            grip=dict(**shared, x=x.cpu(), edge_index=torch.stack([ids, ids]),
                                      edge_weight=torch.ones(m, dtype=torch.float64)),
                            coarse_mean=dict(**shared, **coarse_edges, x=mean.cpu()),
                            coarse_optimized=dict(**shared, **coarse_edges, x=fitted))
                        diagnostics = {}
                        for name, artifact in artifacts.items():
                            Q = gcn_operator(artifact['edge_index'], artifact['edge_weight'].float(), m).to_dense()
                            diagnostics[name] = feature_diagnostics(P, Q, data.x, artifact['x'].float().to(args.device), a)
                        bundle = dict(artifacts=artifacts, diagnostics=diagnostics, optimization=result)
                        atomic_torch(path, bundle)
                artifact_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                for name in VARIANTS:
                    conf = dict(**config, variant=name, epoch=args.epoch, eval_every=args.eval_every,
                        dropout=.9, lr=.01, weight_decay=5e-4, student='GCN-2-256', loss='uniform-soft-CE',
                        artifact_sha256=artifact_hash)
                    entry = dict(id=digest(conf), config=conf, artifact=str(path),
                        diagnostics=bundle['diagnostics'][name],
                        optimization=bundle['optimization'] if name == 'coarse_optimized' else None)
                    entries.append(entry)
                    atomic_json(out/'manifest.json', dict(arguments=vars(args), entries=entries))
                    values = [fit(entry, bundle['artifacts'][name], teacher, seed, 'selection')
                              for seed in selection_seeds]
                    summaries.append(dict(id=entry['id'], variant=name, gamma=gamma, T=T, mu=mu,
                        val=float(np.mean([r['val'] for r in values])),
                        val_std=float(np.std([r['val'] for r in values], ddof=1)),
                        test=float(np.mean([r['test'] for r in values])),
                        test_std=float(np.std([r['test'] for r in values], ddof=1)),
                        **entry['diagnostics']))
                    save_csv(out/'summary.csv', summaries)
        targets = choose(summaries)
        # Record validation-only decisions before running any confirmation seed.
        atomic_json(out/'selection.json', dict(
            targets={label: {k: row[k] for k in ('id', 'variant', 'gamma', 'T', 'mu', 'val')}
                     for label, row in targets.items()},
            selection_seeds=selection_seeds, confirmation_seeds=confirm_seeds,
            criterion='mean validation; exact ties prefer smaller gamma,T,mu'))
        lookup = {entry['id']: entry for entry in entries}
        for key in dict.fromkeys(row['id'] for row in targets.values()):
            entry = lookup[key]
            bundle = torch.load(entry['artifact'], map_location='cpu', weights_only=True)
            teacher = probabilities(entry['config']['gamma'], entry['config']['T'])
            for seed in confirm_seeds:
                fit(entry, bundle['artifacts'][entry['config']['variant']], teacher, seed, 'confirmation')
    if handle:
        handle.update('Complete. Validation selection and confirmation below.')
    else:
        print()
    print('Validation selection (Cora .052, dropout=.9 fixed)')
    print('group                    gamma      T     mu     val')
    for label, row in targets.items():
        print(f'{label:24s} {row["gamma"]:5g} {row["T"]:6g} {row["mu"]:6g} {row["val"]:7.2f}')
    confirmation, comparisons = [], []
    if confirm_seeds:
        values = {label: paired_values(runs, row['id'], confirm_seeds) for label, row in targets.items()}
        print(f'Confirmation seeds {confirm_seeds[0]}-{confirm_seeds[-1]}:')
        print('group                    val +/- sd       test +/- sd      delta vs tuned GRIP [95% CI]')
        for label, row in targets.items():
            avg, sd = values[label].mean(0), values[label].std(0, ddof=1)
            stat = paired_stats(values[label][:, 1], values['grip'][:, 1])
            confirmation.append(dict(group=label, id=row['id'], val=float(avg[0]), val_std=float(sd[0]),
                test=float(avg[1]), test_std=float(sd[1]), versus_grip=stat))
            print(f'{label:24s} {avg[0]:.2f} +/- {sd[0]:.2f}   {avg[1]:.2f} +/- {sd[1]:.2f}   '
                  f'{stat["mean"]:+.2f} [{stat["low"]:+.2f}, {stat["high"]:+.2f}]')
        for label, candidate, baseline in COMPARISONS:
            stat = paired_stats(values[candidate][:, 1], values[baseline][:, 1])
            comparisons.append(dict(comparison=label, paired_test=stat,
                paired_val=paired_stats(values[candidate][:, 0], values[baseline][:, 0]),
                bonferroni_test=paired_stats(values[candidate][:, 1], values[baseline][:, 1], len(COMPARISONS))))
            if 'matched' in label:
                print(f'{label}: {stat["mean"]:+.2f} [{stat["low"]:+.2f}, {stat["high"]:+.2f}] pp')
    atomic_json(out/'confirmation.json', dict(results=confirmation, comparisons=comparisons,
        seeds=confirm_seeds, caveat='Pointwise paired student-seed CIs, fixed data split and condensation seed. '
        'Bonferroni CIs for the four planned comparisons are also saved. No risk/generalization certificate.'))
    print('Full results: summary.csv | student_runs.csv | selection.json | confirmation.json')


if __name__ == '__main__':
    main()
