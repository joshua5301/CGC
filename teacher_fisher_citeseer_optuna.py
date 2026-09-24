"""Budgeted sequential Optuna search for Citeseer 3.6 percent."""
import argparse
import optuna
import json
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import torch
from torch_geometric import seed_everything
from diagonal_metric import condense
from fisher_diagnostic import identity_graph, save_csv
from fisher_grip import sha, train_dual
from teacher_fisher_nystrom import teacher_fisher, METRIC_TEMPERATURE
from sweep_distance import atomic_json, atomic_torch, code_digest, data_digest, digest
from src.dataloader import get_dataset, set_dataset
from src.models import GCN
from src.utils import budget, conv_graph_multi

DOMAINS = ('graphless', 'gcn_transfer')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='/content/drive/MyDrive/GRIP_citeseer_teacher_fisher_optuna')
    p.add_argument('--raw-data-dir', default='/content/data/')
    p.add_argument('--trials', type=int, default=40)
    p.add_argument('--selection-seeds', default='79,80,81')
    p.add_argument('--confirmation-seeds', default='82,83,84,85,86,87,88,89,90,91')
    p.add_argument('--epoch', type=int, default=1000)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    def parse(value, cast=float):
        return sorted(set(cast(x) for x in value.split(',')))
    selection, confirmation = parse(args.selection_seeds, int), parse(args.confirmation_seeds, int)
    if (args.trials < 1 or min(selection + confirmation) < 0
            or min(len(selection), len(confirmation)) < 2 or set(selection) & set(confirmation)
            or not 1 <= args.eval_every <= args.epoch):
        p.error('Invalid trial budget, epochs or seeds')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.output).resolve()
    for folder in ('teachers', 'condensed', 'runs', 'models', 'trials'):
        (out/folder).mkdir(parents=True, exist_ok=True)
    logfile = out / f'full_{datetime.now():%Y%m%d_%H%M%S}.log'
    total = args.trials
    print('GPU:', torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU')
    print(f'Results: {out}\nLog: {logfile}')
    print(f'Citeseer .036; tau=1, dropout=.9; {total} settings, {total*len(selection)} selection fits.')
    try:
        from IPython import get_ipython
        from IPython.display import display
        handle = display('Preparing budgeted Optuna search...', display_id=True) if get_ipython() else None
    except ImportError:
        handle = None
    def progress(message):
        if handle:
            handle.update(message)
        else:
            print('\r' + message.ljust(130), end='', flush=True)
    train = SimpleNamespace(dataset_name='citeseer', ratio=.036, device=args.device,
                            raw_data_dir=str(Path(args.raw_data_dir).resolve()) + '/')
    rows, entries, summaries, invalid = [], {}, [], []
    with logfile.open('w', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            seed_everything(0)
            train, data, _, _ = set_dataset(train, get_dataset(train))
            _, _, z = conv_graph_multi(train, data)
        z = z.double()
        zdata = identity_graph(z, torch.nn.functional.one_hot(data.y, train.num_class))
        zdata.y = data.y
        zdata.train_mask, zdata.val_mask, zdata.test_mask = data.train_mask, data.val_mask, data.test_mask
        datasets = dict(graphless=zdata, gcn_transfer=data)
        helpers = {name: sha(Path(__file__).with_name(name)) for name in (
            'teacher_fisher_citeseer_optuna.py', 'teacher_fisher_nystrom.py', 'diagonal_metric.py',
            'fisher_grip.py', 'fisher_diagnostic.py')}
        base = dict(source=code_digest(), helpers=helpers, data=data_digest(data),
                    dataset='citeseer', ratio=.036, kernel='relu', basis=3000,
                    condensation_seed=0, torch=str(torch.__version__), device=args.device,
                    threads=4, metric_temperature=METRIC_TEMPERATURE, optuna_version=optuna.__version__)
        study_config = dict(base=base, selection=selection, confirmation=confirmation,
            epoch=args.epoch, eval_every=args.eval_every, sampler='TPE', sampler_seed=20260924,
            startup_trials=10, objective='mean_gcn_transfer_validation',
            space=dict(gamma=[.001,.01,.1,1.], T=[.1,.2,.5,1.,2.,5.,10.],
                       mu=[0.,.1,.2,.5,1.,2.,5.,10.], alpha=[.05,.95]))
        manifest_path = out/'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text())['study_config'] != study_config:
            raise ValueError('Study configuration changed; use a new output directory')
        atomic_json(manifest_path, dict(study_config=study_config, requested_trials=total))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=20260924, n_startup_trials=10))
        study.enqueue_trial(dict(gamma=.01, T=.2, mu=.1, alpha=.05))
        if list((out/'trials').glob('*.json')) and max(int(x.stem) for x in (out/'trials').glob('*.json')) >= total:
            raise ValueError('Cannot shrink an existing study; retain or increase --trials')
        def fit(entry, artifact, seed, stage):
            conf, key = entry['config'], entry['id']
            path = out/'runs'/f'{key}_{seed}.json'
            if path.exists():
                result = json.loads(path.read_text(encoding='utf-8'))
                if result['config'] != conf or result['seed'] != seed:
                    raise ValueError('Cached run mismatch')
            else:
                with redirect_stdout(log), redirect_stderr(log):
                    seed_everything(seed)
                    graph = identity_graph(artifact['x'].to(args.device), artifact['y'].to(args.device))
                    model = GCN(data.num_features, 256, train.num_class, 2, .9).to(args.device)
                    best = train_dual(model, graph, datasets, args.epoch, args.eval_every)
                    scores = {}
                    for domain, value in best.items():
                        scores[domain] = dict(val=100*value['val'], test=100*value['test'], epoch=value['epoch'])
                        if stage == 'confirmation':
                            atomic_torch(out/'models'/f'{key}_{seed}_{domain}.pt',
                                         {k: v.cpu() for k, v in value['state'].items()})
                    result = dict(config=conf, seed=seed, scores=scores)
                    atomic_json(path, result)
                    del model, graph, best
            for domain, score in result['scores'].items():
                rows.append(dict(id=key, stage=stage, domain=domain, seed=seed,
                    **{k: conf[k] for k in ('gamma', 'T', 'mu', 'alpha')}, **score))
            return result['scores']

        completed = 0
        # Replaying seeded ask/tell from trial zero preserves sampler state without
        # keeping a live SQLite database on Google Drive. Fits and trials are cached.
        for _ in range(total):
            trial = study.ask()
            gamma = trial.suggest_categorical('gamma', study_config['space']['gamma'])
            T = trial.suggest_categorical('T', study_config['space']['T'])
            mu = trial.suggest_categorical('mu', study_config['space']['mu'])
            alpha = trial.suggest_float('alpha', .05, .95, log=True)
            trial_path = out/'trials'/f'{trial.number}.json'
            previous = json.loads(trial_path.read_text()) if trial_path.exists() else None
            if previous is not None and previous['params'] != trial.params:
                raise ValueError('Optuna replay mismatch; preserve software version or use a new directory')
            if previous is not None and previous['invalid']:
                invalid.append(previous)
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                completed += 1
                continue
            tc = dict(**base, gamma=gamma)
            teacher_path = out/'teachers'/f'{digest(tc)}.pt'
            if teacher_path.exists():
                teacher = torch.load(teacher_path, map_location='cpu', weights_only=True)
            else:
                progress(f'Fitting teacher and Fisher once for gamma={gamma:g}...')
                with redirect_stdout(log), redirect_stderr(log):
                    teacher = teacher_fisher(z, data.y, data.train_mask, None, dict(**tc, T=1.))
                    atomic_torch(teacher_path, teacher)
            teacher_hash = sha(teacher_path)
            logits = teacher['logits'].to(args.device)
            probabilities = (logits/T).softmax(1).clamp_min(1e-12)
            probabilities /= probabilities.sum(1, keepdim=True)
            cc = dict(**base, gamma=gamma, T=T, mu=mu, alpha=alpha,
                      m=budget(train), partition_steps=100, teacher_sha256=teacher_hash)
            artifact_path = out/'condensed'/f'{digest(cc)}.pt'
            if artifact_path.exists():
                artifact = torch.load(artifact_path, map_location='cpu', weights_only=True)
            else:
                with redirect_stdout(log), redirect_stderr(log):
                    seed_everything(0)
                    try:
                        artifact = condense(z, probabilities, teacher['score'], alpha, budget(train), mu)
                    except RuntimeError as error:
                        if 'GRIP dropped empty cells' not in str(error):
                            raise
                        invalid.append(dict(gamma=gamma, T=T, mu=mu, alpha=alpha, reason=str(error)))
                        atomic_json(out/'invalid_settings.json', invalid)
                        atomic_json(trial_path, dict(params=trial.params, invalid=True, reason=str(error)))
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                        completed += 1
                        continue
                    atomic_torch(artifact_path, artifact)
            conf = dict(**cc, epoch=args.epoch, eval_every=args.eval_every, dropout=.9,
                hidden=256, loss='uniform-soft-CE', lr=.01, weight_decay=5e-4,
                artifact_sha256=sha(artifact_path), checkpoint='independent-first-best-validation-per-domain')
            key = digest(conf)
            entry = dict(id=key, config=conf, artifact=str(artifact_path))
            entries[key] = entry
            scores = []
            for seed in selection:
                progress(f'Selection {completed+1}/{total}: gamma={gamma:g}, T={T:g}, mu={mu:g}, alpha={alpha:g}, seed={seed}')
                scores.append(fit(entry, artifact, seed, 'selection'))
            for domain in DOMAINS:
                summaries.append(dict(id=key, domain=domain, gamma=gamma, T=T, mu=mu, alpha=alpha,
                    val=float(np.mean([s[domain]['val'] for s in scores])),
                    test=float(np.mean([s[domain]['test'] for s in scores]))))
            completed += 1
            objective = float(np.mean([s['gcn_transfer']['val'] for s in scores]))
            if previous is not None and previous['value'] != objective:
                raise ValueError('Cached trial objective mismatch')
            study.tell(trial, objective)
            atomic_json(trial_path, dict(params=trial.params, invalid=False, value=objective, id=key))
            save_csv(out/'summary.csv', summaries)
            save_csv(out/'student_runs.csv', rows)
        # Selection uses validation only, with a predeclared numerical tie break.
        if not summaries:
            raise RuntimeError('No valid settings; see invalid_settings.json')
        best_row = min((r for r in summaries if r['domain'] == 'gcn_transfer'),
                       key=lambda r: (-r['val'], r['gamma'], r['T'], r['mu'], r['alpha']))
        winners = {d: next(r for r in summaries if r['domain'] == d and r['id'] == best_row['id'])
                   for d in DOMAINS}
        atomic_json(out/'selection.json', dict(winners=winners,
            entries={d: entries[r['id']] for d, r in winners.items()},
            rule='maximum mean GCN validation; ties: ascending gamma,T,mu,alpha; graphless diagnostic only'))
        for key in sorted({r['id'] for r in winners.values()}):
            entry = entries[key]
            artifact = torch.load(entry['artifact'], map_location='cpu', weights_only=True)
            for seed in confirmation:
                progress(f'Confirming selected configuration {key}, seed={seed}...')
                fit(entry, artifact, seed, 'confirmation')
                save_csv(out/'student_runs.csv', rows)
    if handle:
        handle.update('Complete. Validation-selected settings and fresh-seed confirmation below.')
    else:
        print()
    print('domain          gamma      T     mu  alpha   selection val   fresh val +/- sd   fresh test +/- sd')
    report = {}
    for domain, winner in winners.items():
        indexed = {r['seed']: r for r in rows if r['stage'] == 'confirmation'
                   and r['domain'] == domain and r['id'] == winner['id']}
        if set(indexed) != set(confirmation):
            raise ValueError('Incomplete confirmation')
        values = np.array([[indexed[s]['val'], indexed[s]['test']] for s in confirmation])
        mean, sd = values.mean(0), values.std(0, ddof=1)
        report[domain] = dict(selected=winner, val=float(mean[0]), val_std=float(sd[0]),
                              test=float(mean[1]), test_std=float(sd[1]))
        print(f'{domain:14s} {winner["gamma"]:6g} {winner["T"]:6g} {winner["mu"]:6g} {winner["alpha"]:6g}'
              f' {winner["val"]:10.2f}       {mean[0]:.2f} +/- {sd[0]:.2f}      {mean[1]:.2f} +/- {sd[1]:.2f}')
    atomic_json(out/'confirmation.json', dict(results=report, seeds=confirmation,
        caveat='Proposed method only; no baseline superiority claim. Fixed data split and condensation seed. '
        'Independent student seeds do not establish generalization across splits or graphs.'))
    print('Full results: summary.csv | student_runs.csv | selection.json | confirmation.json')
    if invalid:
        print(f'Excluded {len(invalid)} settings with reduced node budget; see invalid_settings.json.')


if __name__ == '__main__':
    main()
