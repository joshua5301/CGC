"""Refine GRIP with a frozen Fisher ensemble; independently retrain paired students."""
import argparse
import copy
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
from extend_robust import paired_stats
from fisher_diagnostic import identity_graph, parameters, pair_costs, save_csv
from fisher_partition import build_factors, refine
from src.dataloader import get_dataset, set_dataset
from src.models import GCN
from src.utils import soft_label_ce
from sweep_distance import atomic_json, atomic_torch, code_digest, data_digest, digest


VARIANTS = ('grip', 'euclidean_refined', 'fisher_refined')
PROTOCOLS = ('graphless', 'gcn_transfer')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source(directory, data, device):
    """Consume the exact diagnostic centers and frozen students, never retrain them."""
    directory = Path(directory)
    manifest_path = directory/'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Run fisher_diagnostic.py first: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    conf = manifest['config']
    if (conf['dataset'] != 'cora' or conf['ratio'] != .052 or conf['data'] != data_digest(data)
            or conf['student'] != 'GCN-with-identity-edges=MLP'
            or conf['checkpoint'] != 'fixed-final-epoch' or conf['source'] != code_digest()
            or conf['torch'] != str(torch.__version__)
            or conf['runner'] != sha(Path(__file__).with_name('fisher_diagnostic.py'))):
        raise ValueError('Diagnostic source/data/runtime mismatch; rerun its cell with this code/runtime')
    matches = [path for path in (directory/'cache').glob('*.pt') if sha(path) == conf['artifact_sha256']]
    if len(matches) != 1:
        raise ValueError('Could not uniquely locate the diagnostic teacher/GRIP artifact')
    source = torch.load(matches[0], map_location='cpu', weights_only=True)
    if source['Z'].shape != data.x.shape or len(source['a']) != len(data.x):
        raise ValueError('Invalid diagnostic artifact dimensions')
    expected = sorted(manifest['seeds'])
    indexed = {entry['seed']: entry for entry in manifest['entries']}
    if len(indexed) != len(manifest['entries']) or sorted(indexed) != expected:
        raise ValueError('Diagnostic ensemble is incomplete or duplicated; finish its cell first')
    weights, hashes = [], {}
    for seed in expected:
        entry = indexed[seed]
        model_path = directory/'models'/Path(entry['model']).name
        report_path = directory/'diagnostics'/Path(entry['report']).name
        report = json.loads(report_path.read_text(encoding='utf-8'))
        model_hash = sha(model_path)
        expected_report = digest(dict(**conf, seed=seed, model_sha256=model_hash))
        if report['config'] != conf or report['seed'] != seed or report_path.stem != expected_report:
            raise ValueError('Diagnostic report/config mismatch')
        model = GCN(data.num_features, conf['hidden'], int(data.y.max())+1, 2, conf['dropout']).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        weights.append(parameters(model))
        hashes[str(seed)] = model_hash
    return source, weights, dict(config=conf, seeds=expected, models=hashes,
                                artifact=sha(matches[0]))


def train_dual(model, graph, datasets, epochs, eval_every):
    """One training trajectory, independently selected validation checkpoints per domain."""
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=5e-4)
    best = {name: dict(val=-1., test=None, epoch=None, state=None) for name in datasets}
    for epoch in range(1, epochs+1):
        if epoch == epochs//2:
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
        model.train()
        loss = soft_label_ce(model(graph), graph.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % eval_every and epoch != epochs:
            continue
        model.eval()
        with torch.no_grad():
            for name, data in datasets.items():
                predicted = model(data).argmax(1)
                val = int((predicted[data.val_mask] == data.y[data.val_mask]).sum())/int(data.val_mask.sum())
                test = int((predicted[data.test_mask] == data.y[data.test_mask]).sum())/int(data.test_mask.sum())
                if val > best[name]['val']:
                    best[name] = dict(val=val, test=test, epoch=epoch, state=copy.deepcopy(model.state_dict()))
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: loss={float(loss):.6f}', flush=True)
    return best


@torch.no_grad()
def prediction_diagnostics(z, c, teacher, assignment, models, batch_size=512):
    """Mean actual KL, fixed-Fisher cost and signed/absolute correction across models."""
    rows = []
    for weights in models:
        collected = {key: [] for key in ('kl', 'quadratic', 'correction', 'ce_delta')}
        for start in range(0, len(z), batch_size):
            x = z[start:start+batch_size]
            cost = pair_costs(x, c[assignment[start:start+len(x)]], teacher[start:start+len(x)], weights)
            for key in collected:
                collected[key].append(cost[key].cpu())
        values = {key: torch.cat(parts) for key, parts in collected.items()}
        rows.append(dict(kl=float(values['kl'].mean()), quadratic=float(values['quadratic'].mean()),
            correction=float(values['correction'].mean()), correction_abs=float(values['correction'].abs().mean()),
            replacement_ce_delta=float(values['ce_delta'].mean())))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic-dir', default='/content/drive/MyDrive/GRIP_cora_fisher_diagnostic')
    p.add_argument('--output', default='/content/drive/MyDrive/GRIP_cora_fisher_clustering')
    p.add_argument('--raw-data-dir', default='/content/data/')
    p.add_argument('--outer-steps', type=int, default=20)
    p.add_argument('--center-steps', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epoch', type=int, default=1000)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--seeds', default='26,27,28,29,30,31,32,33,34,35')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    seeds = sorted(set(int(value) for value in args.seeds.split(',')))
    if (not seeds or min(seeds) < 0 or len(seeds) < 2
            or min(args.outer_steps, args.center_steps, args.batch_size) < 1
            or not 1 <= args.eval_every <= args.epoch):
        p.error('Invalid settings')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.output).resolve()
    for folder in ('condensed', 'runs', 'models'):
        (out/folder).mkdir(parents=True, exist_ok=True)
    logfile = out/f'clustering_{datetime.now():%Y%m%d_%H%M%S}.log'
    print('GPU:', torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU')
    print(f'Results: {out}\nLog: {logfile}')
    try:
        from IPython import get_ipython
        from IPython.display import display
        handle = display('Preparing paired Fisher clustering...', display_id=True) if get_ipython() else None
    except ImportError:
        handle = None

    def progress(message):
        if handle:
            handle.update(message)
        else:
            print('\r'+message.ljust(120), end='', flush=True)

    train = SimpleNamespace(dataset_name='cora', ratio=.052, device=args.device,
        raw_data_dir=str(Path(args.raw_data_dir).resolve())+'/')
    entries, runs, diagnostics = [], [], {}
    with logfile.open('w', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            seed_everything(0)
            train, data, _, _ = set_dataset(train, get_dataset(train))
            source, weights, provenance = load_source(args.diagnostic_dir, data, args.device)
        if set(seeds) & set(provenance['seeds']):
            raise ValueError('Evaluation seeds must be disjoint from frozen metric-student seeds')
        source_config = provenance['config']
        print(f'Cora .052; frozen seeds {provenance["seeds"]}; evaluation seeds {seeds[0]}-{seeds[-1]}.')
        print(f'gamma={source_config["gamma"]:g}, T={source_config["T"]:g}, mu={source_config["mu"]:g}, '
              f'dropout={source_config["dropout"]:g}; {3*len(seeds)} fits, two evaluation domains.')
        z, teacher, c0 = (source[key].double().to(args.device) for key in ('Z', 'F', 'C'))
        a0 = source['a'].to(args.device)
        zdata = identity_graph(z, teacher)
        # Evaluation targets/masks belong to original nodes, not condensed labels.
        zdata.y = data.y
        zdata.train_mask, zdata.val_mask, zdata.test_mask = data.train_mask, data.val_mask, data.test_mask
        datasets = dict(graphless=zdata, gcn_transfer=data)
        hashes = {name: sha(Path(__file__).with_name(name)) for name in
                  ('fisher_grip.py', 'fisher_partition.py', 'fisher_diagnostic.py', 'extend_robust.py')}
        base = dict(source=code_digest(), helpers=hashes, provenance=provenance,
            outer_steps=args.outer_steps, center_steps=args.center_steps, batch_size=args.batch_size,
            coefficient=source_config['mu'], device=args.device,
            normalization='fixed-global-mean-cost-per-geometry;global-teacher-KL')
        factors = None
        for variant in VARIANTS:
            conf = dict(**base, variant=variant)
            path = out/'condensed'/f'{digest(conf)}.pt'
            if path.exists():
                artifact = torch.load(path, map_location='cpu', weights_only=True)
            else:
                progress(f'Constructing {variant}...')
                with redirect_stdout(log), redirect_stderr(log):
                    if variant == 'grip':
                        artifact = dict(x=source['C'], y=source['Y'], assign=source['a'], status='original_GRIP')
                    else:
                        if variant == 'fisher_refined' and factors is None:
                            factors = build_factors(z, weights, args.batch_size)
                        artifact = refine(z, teacher, c0, a0,
                            factors=factors if variant == 'fisher_refined' else None,
                            coefficient=source_config['mu'], outer_steps=args.outer_steps,
                            center_steps=args.center_steps, batch_size=args.batch_size)
                    # Diagnostics at actual float32 student inputs, evaluated in float64.
                    artifact['frozen_diagnostics'] = prediction_diagnostics(z, artifact['x'].float().double().to(args.device),
                        teacher, artifact['assign'].to(args.device), weights)
                    atomic_torch(path, artifact)
            diagnostics[variant] = artifact['frozen_diagnostics']
            student_config = dict(**conf, epoch=args.epoch, eval_every=args.eval_every,
                dropout=source_config['dropout'], hidden=256, loss='uniform-soft-CE',
                lr=.01, weight_decay=5e-4, artifact_sha256=sha(path),
                checkpoint='independent-first-best-validation-per-domain')
            key = digest(student_config)
            entries.append(dict(id=key, config=student_config, artifact=str(path)))
            atomic_json(out/'manifest.json', dict(entries=entries, arguments=vars(args), frozen_diagnostics=diagnostics))
        # Release the large low-rank factor tensor before student fitting.
        del factors
        for entry in entries:
            conf, key = entry['config'], entry['id']
            variant = conf['variant']
            artifact = torch.load(entry['artifact'], map_location='cpu', weights_only=True)
            graph = identity_graph(artifact['x'].to(args.device), artifact['y'].to(args.device))
            for seed in seeds:
                progress(f'Retraining {variant}, seed={seed}...')
                run_path = out/'runs'/f'{key}_{seed}.json'
                if run_path.exists():
                    result = json.loads(run_path.read_text(encoding='utf-8'))
                    if result['config'] != conf or result['seed'] != seed:
                        raise ValueError('Cached student configuration mismatch')
                else:
                    with redirect_stdout(log), redirect_stderr(log):
                        seed_everything(seed)
                        model = GCN(data.num_features, 256, train.num_class, 2, conf['dropout']).to(args.device)
                        best = train_dual(model, graph, datasets, args.epoch, args.eval_every)
                        scores = {}
                        for domain, fit in best.items():
                            model.load_state_dict(fit['state'])
                            model.eval()
                            with torch.no_grad():
                                lp = model(datasets[domain]).double()
                                syn_lp = model(graph).double()
                                measured = dict(original_teacher_ce=float(-(teacher*lp).sum(1).mean()),
                                    condensed_uniform_ce=float(-(graph.y.double()*syn_lp).sum(1).mean()))
                                if domain == 'graphless':
                                    measured.update(prediction_diagnostics(z, graph.x.double(), teacher,
                                        artifact['assign'].to(args.device), [parameters(model)]))
                            checkpoint = out/'models'/f'{key}_{seed}_{domain}.pt'
                            atomic_torch(checkpoint, {k: v.cpu() for k, v in fit['state'].items()})
                            scores[domain] = dict(val=100*fit['val'], test=100*fit['test'], best_epoch=fit['epoch'],
                                                 diagnostics=measured, checkpoint=str(checkpoint))
                        result = dict(config=conf, seed=seed, scores=scores)
                        atomic_json(run_path, result)
                        del model, best
                for domain, score in result['scores'].items():
                    runs.append(dict(variant=variant, seed=seed, domain=domain,
                                     val=score['val'], test=score['test'], **score['diagnostics']))
                # Domains have different diagnostic columns; save scores as a common table.
                save_csv(out/'student_runs.csv', [{k: r[k] for k in ('variant', 'seed', 'domain', 'val', 'test')} for r in runs])
    if handle:
        handle.update('Complete. Frozen metric and independent retraining results below.')
    else:
        print()
    print('Frozen ensemble: actual prediction KL and local quadratic (lower is better).')
    print('variant                  KL       quadratic   signed CE change')
    for variant, values in diagnostics.items():
        print(f'{variant:21s} {values["kl"]:10.4g} {values["quadratic"]:12.4g} '
              f'{values["replacement_ce_delta"]:+16.4g}')
    summaries, comparisons = [], []
    for domain in PROTOCOLS:
        print(f'{domain}: val +/- sd       test +/- sd      test delta vs GRIP [95% CI]')
        by_variant = {}
        for variant in VARIANTS:
            indexed = {r['seed']: r for r in runs if r['variant'] == variant and r['domain'] == domain}
            if set(indexed) != set(seeds):
                raise ValueError('Incomplete paired results')
            values = np.array([[indexed[seed]['val'], indexed[seed]['test']] for seed in seeds])
            by_variant[variant] = values
            avg, sd = values.mean(0), values.std(0, ddof=1)
            stat = paired_stats(values[:, 1], by_variant['grip'][:, 1])
            measured = {k: float(np.mean([indexed[seed][k] for seed in seeds])) for k in indexed[seeds[0]]
                        if k not in ('variant', 'seed', 'domain', 'val', 'test')}
            summaries.append(dict(domain=domain, variant=variant, val=float(avg[0]), val_std=float(sd[0]),
                test=float(avg[1]), test_std=float(sd[1]), versus_grip=stat, diagnostics=measured))
            print(f'{variant:21s} {avg[0]:.2f} +/- {sd[0]:.2f}   {avg[1]:.2f} +/- {sd[1]:.2f}   '
                  f'{stat["mean"]:+.2f} [{stat["low"]:+.2f}, {stat["high"]:+.2f}]')
        for baseline in ('grip', 'euclidean_refined'):
            stat = paired_stats(by_variant['fisher_refined'][:, 1], by_variant[baseline][:, 1])
            comparisons.append(dict(domain=domain, baseline=baseline, paired_test=stat,
                paired_val=paired_stats(by_variant['fisher_refined'][:, 0], by_variant[baseline][:, 0]),
                bonferroni_test=paired_stats(by_variant['fisher_refined'][:, 1], by_variant[baseline][:, 1], 4)))
            if baseline == 'euclidean_refined':
                print(f'Fisher - Euclidean: {stat["mean"]:+.2f} [{stat["low"]:+.2f}, {stat["high"]:+.2f}] pp')
    atomic_json(out/'summary.json', dict(results=summaries, comparisons=comparisons,
        frozen_diagnostics=diagnostics, seeds=seeds, provenance=provenance,
        caveat='Fixed data split, teacher, initialization and frozen metric ensemble. '
        'Pointwise student-seed intervals; four planned Fisher contrasts also have Bonferroni intervals. '
        'Graphless evaluation is distinct from raw-graph GCN transfer. No CE-risk guarantee from quadratic descent.'))
    print('Full results: summary.json | student_runs.csv | condensed/*.pt | runs/*.json')


if __name__ == '__main__':
    main()
