"""Compare GRIP with global diagonal centered-logit sensitivity scaling."""
import argparse
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
from fisher_diagnostic import identity_graph, parameters, save_csv
from diagonal_metric import sensitivity, condense
from fisher_grip import load_source, sha, train_dual, prediction_diagnostics
from src.dataloader import get_dataset, set_dataset
from src.models import GCN
from sweep_distance import atomic_json, atomic_torch, code_digest, digest


ALPHAS = (0., .1, .3, .5, .8)
VARIANTS = ('grip',) + tuple(f'alpha_{a:g}' for a in ALPHAS[1:])
PROTOCOLS = ('graphless', 'gcn_transfer')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic-dir', default='/content/drive/MyDrive/GRIP_cora_fisher_diagnostic')
    p.add_argument('--output', default='/content/drive/MyDrive/GRIP_cora_diagonal_metric')
    p.add_argument('--raw-data-dir', default='/content/data/')
    p.add_argument('--partition-steps', type=int, default=100)

    p.add_argument('--epoch', type=int, default=1000)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--seeds', default='46,47,48,49,50,51,52,53,54,55')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    seeds = sorted(set(int(value) for value in args.seeds.split(',')))
    if (not seeds or min(seeds) < 0 or len(seeds) < 2
            or args.partition_steps < 1
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
        handle = display('Preparing diagonal sensitivity sweep...', display_id=True) if get_ipython() else None
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
              f'dropout={source_config["dropout"]:g}; {len(VARIANTS)*len(seeds)} fits, two evaluation domains.')
        z, teacher, c0 = (source[key].double().to(args.device) for key in ('Z', 'F', 'C'))
        zdata = identity_graph(z, teacher)
        # Evaluation targets/masks belong to original nodes, not condensed labels.
        zdata.y = data.y
        zdata.train_mask, zdata.val_mask, zdata.test_mask = data.train_mask, data.val_mask, data.test_mask
        datasets = dict(graphless=zdata, gcn_transfer=data)
        hashes = {name: sha(Path(__file__).with_name(name)) for name in
                  ('diagonal_grip.py', 'diagonal_metric.py', 'fisher_grip.py', 'fisher_partition.py', 'fisher_diagnostic.py', 'extend_robust.py')}
        base = dict(source=code_digest(), helpers=hashes, provenance=provenance,
            partition_steps=args.partition_steps,
            coefficient=source_config['mu'], device=args.device,
            normalization='original-GRIP-normalization-in-scaled-space;global-teacher-KL')

        metric_path = out/'condensed'/f'metric_{digest(base)}.pt'
        if metric_path.exists():
            metric = torch.load(metric_path, map_location='cpu', weights_only=True)
        else:
            progress('Estimating global dimension sensitivities...')
            metric = sensitivity(z, weights)
            atomic_torch(metric_path, metric)
        for variant, alpha in zip(VARIANTS, ALPHAS):
            conf = dict(**base, variant=variant, alpha=alpha, metric_sha256=sha(metric_path))
            path = out/'condensed'/f'{digest(conf)}.pt'
            if path.exists():
                artifact = torch.load(path, map_location='cpu', weights_only=True)
            else:
                progress(f'Constructing {variant}...')
                with redirect_stdout(log), redirect_stderr(log):
                    if variant == 'grip':
                        artifact = dict(x=source['C'], y=source['Y'], assign=source['a'], status='original_GRIP')
                    else:
                        seed_everything(0)
                        artifact = condense(z, teacher, metric['score'], alpha, len(c0),
                            source_config['mu'], args.partition_steps)
                    # Diagnostics at actual float32 student inputs, evaluated in float64.
                    artifact['frozen_diagnostics'] = prediction_diagnostics(z, artifact['x'].float().double().to(args.device),
                        teacher, artifact['assign'].to(args.device), weights)
                    atomic_torch(path, artifact)
            diagnostics[variant] = artifact['frozen_diagnostics']
            if alpha:
                print(f'{variant}: dimension weights '
                      f'[{artifact["weight_min"]:.3g}, {artifact["weight_max"]:.3g}]')
            student_config = dict(**conf, epoch=args.epoch, eval_every=args.eval_every,
                dropout=source_config['dropout'], hidden=256, loss='uniform-soft-CE',
                lr=.01, weight_decay=5e-4, artifact_sha256=sha(path),
                checkpoint='independent-first-best-validation-per-domain')
            key = digest(student_config)
            entries.append(dict(id=key, config=student_config, artifact=str(path)))
            atomic_json(out/'manifest.json', dict(entries=entries, arguments=vars(args), frozen_diagnostics=diagnostics))


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
        handle.update('Complete. Diagonal sensitivity comparisons below.')
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
        for variant in VARIANTS[1:]:
            comparisons.append(dict(domain=domain, variant=variant, baseline='grip',
                paired_test=paired_stats(by_variant[variant][:, 1], by_variant['grip'][:, 1]),
                paired_val=paired_stats(by_variant[variant][:, 0], by_variant['grip'][:, 0]),
                bonferroni_test=paired_stats(by_variant[variant][:, 1], by_variant['grip'][:, 1], 8)))
    atomic_json(out/'summary.json', dict(results=summaries, comparisons=comparisons,
        frozen_diagnostics=diagnostics, seeds=seeds, provenance=provenance,
        caveat='Fixed data split, teacher and sensitivity ensemble. All alpha settings reported; no test-based selection. '
        'Pointwise student-seed intervals; eight planned alpha contrasts also have Bonferroni intervals. '
        'Graphless evaluation is distinct from raw-graph GCN transfer. No certified Lipschitz or generalization guarantee from empirical sensitivities.'))
    print('Full results: summary.json | student_runs.csv | condensed/*.pt | runs/*.json')


if __name__ == '__main__':
    main()
