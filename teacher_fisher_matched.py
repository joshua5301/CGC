"""Matched alpha=0 vs .05 ablation using full-sweep confirmation seeds."""
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
from diagonal_metric import condense
from extend_robust import paired_stats
from fisher_diagnostic import identity_graph, save_csv
from fisher_grip import sha, train_dual
from src.dataloader import get_dataset, set_dataset
from src.models import GCN
from src.utils import conv_graph_multi
from sweep_distance import atomic_json, atomic_torch, code_digest, data_digest, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='/content/drive/MyDrive/GRIP_cora_teacher_fisher_full')
    p.add_argument('--output', default='/content/drive/MyDrive/GRIP_cora_teacher_fisher_matched')
    p.add_argument('--raw-data-dir', default='/content/data/')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    source, out = Path(args.source).resolve(), Path(args.output).resolve()
    if source == out:
        raise ValueError('Use a separate output directory')
    manifest = json.loads((source/'manifest.json').read_text())
    selected = json.loads((source/'selection.json').read_text())
    entry = selected['entries']['gcn_transfer']
    conf = entry['config']
    expected = dict(dataset='cora', ratio=.052, gamma=.001, T=1., mu=10., alpha=.05,
                    dropout=.9, metric_temperature=1., loss='uniform-soft-CE')
    if any(conf[k] != v for k, v in expected.items()):
        raise ValueError('Source is not the requested fixed configuration')
    if selected['entries']['graphless']['id'] != entry['id'] or digest(conf) != entry['id']:
        raise ValueError('Source winners differ or configuration digest is invalid')
    if conf['source'] != code_digest() or conf['torch'] != str(torch.__version__) or conf['device'] != args.device:
        raise ValueError('Source code/runtime mismatch')
    for name, value in conf['helpers'].items():
        if sha(Path(__file__).with_name(name)) != value:
            raise ValueError(f'Source helper changed: {name}')
    seeds = manifest['confirmation_seeds']
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or set(seeds) & set(manifest['selection_seeds']):
        raise ValueError('Invalid source seed lists')
    original = {}
    for seed in seeds:
        result = json.loads((source/'runs'/f'{entry["id"]}_{seed}.json').read_text())
        if result['config'] != conf or result['seed'] != seed:
            raise ValueError('Source run configuration mismatch')
        original[seed] = result
    positive_path = source/'condensed'/Path(entry['artifact']).name
    if sha(positive_path) != conf['artifact_sha256']:
        raise ValueError('Source condensation changed')
    teacher_path = source/'teachers'/f'{digest(dict(**manifest["base"], gamma=conf["gamma"]))}.pt'
    if sha(teacher_path) != conf['teacher_sha256']:
        raise ValueError('Source teacher changed')
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for folder in ('runs', 'models', 'condensed'):
        (out/folder).mkdir(parents=True, exist_ok=True)
    logfile = out/f'matched_{datetime.now():%Y%m%d_%H%M%S}.log'
    print(f'Results: {out}\nLog: {logfile}')
    print(f'gamma=.001, T=1, mu=10, tau=1, dropout=.9; reuse alpha=.05, {len(seeds)} new alpha=0 fits.')
    try:
        from IPython import get_ipython
        from IPython.display import display
        handle = display('Preparing matched comparison...', display_id=True) if get_ipython() else None
    except ImportError:
        handle = None
    def progress(message):
        if handle:
            handle.update(message)
        else:
            print('\r'+message.ljust(100), end='', flush=True)
    rows = []
    with logfile.open('w', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            seed_everything(0)
            train = SimpleNamespace(dataset_name='cora', ratio=.052, device=args.device,
                raw_data_dir=str(Path(args.raw_data_dir).resolve())+'/')
            train, data, _, _ = set_dataset(train, get_dataset(train))
            if data_digest(data) != conf['data']:
                raise ValueError('Source data mismatch')
            _, _, z = conv_graph_multi(train, data)
            teacher = torch.load(teacher_path, map_location='cpu', weights_only=True)
            probabilities = (teacher['logits'].to(args.device)/conf['T']).softmax(1).clamp_min(1e-12)
            probabilities /= probabilities.sum(1, keepdim=True)
            baseline = dict(source_config=conf, alpha=0., helpers={name: sha(Path(__file__).with_name(name))
                for name in ('teacher_fisher_matched.py', 'diagonal_metric.py', 'extend_robust.py')})
            key = digest(baseline)
            artifact_path = out/'condensed'/f'{key}.pt'
            if artifact_path.exists():
                artifact = torch.load(artifact_path, map_location='cpu', weights_only=True)
            else:
                seed_everything(0)
                artifact = condense(z, probabilities, teacher['score'], 0., conf['m'], conf['mu'], conf['partition_steps'])
                atomic_torch(artifact_path, artifact)
            positive = torch.load(positive_path, map_location='cpu', weights_only=True)
            if len(positive['x']) != len(artifact['x']):
                raise ValueError('Unequal condensation budgets')
            baseline['artifact_sha256'] = sha(artifact_path)
            key = digest(baseline)
            atomic_json(out/'manifest.json', dict(config=baseline, seeds=seeds, source=str(source)))
            graph = identity_graph(artifact['x'].to(args.device), artifact['y'].to(args.device))
            zdata = identity_graph(z, probabilities)
            zdata.y = data.y
            zdata.train_mask, zdata.val_mask, zdata.test_mask = data.train_mask, data.val_mask, data.test_mask
            datasets = dict(graphless=zdata, gcn_transfer=data)
        for seed in seeds:
            progress(f'Matched alpha=0, seed={seed}...')
            path = out/'runs'/f'{key}_{seed}.json'
            if path.exists():
                result = json.loads(path.read_text())
                if result['config'] != baseline or result['seed'] != seed:
                    raise ValueError('Baseline cache mismatch')
            else:
                with redirect_stdout(log), redirect_stderr(log):
                    seed_everything(seed)
                    model = GCN(data.num_features, conf['hidden'], train.num_class, 2, conf['dropout']).to(args.device)
                    best = train_dual(model, graph, datasets, conf['epoch'], conf['eval_every'])
                    scores = {}
                    for domain, fit in best.items():
                        scores[domain] = dict(val=100*fit['val'], test=100*fit['test'], epoch=fit['epoch'])
                        atomic_torch(out/'models'/f'{key}_{seed}_{domain}.pt', {k: v.cpu() for k, v in fit['state'].items()})
                    result = dict(config=baseline, seed=seed, scores=scores)
                    atomic_json(path, result)
                    del model, best
            for alpha, run in ((0., result), (.05, original[seed])):
                for domain, score in run['scores'].items():
                    rows.append(dict(alpha=alpha, seed=seed, domain=domain, **score))
            save_csv(out/'student_runs.csv', rows)
    if handle:
        handle.update('Complete. Matched alpha comparison below.')
    else:
        print()
    report = {}
    for domain in datasets:
        arrays = {}
        print(f'{domain}: alpha   val +/- sd     test +/- sd')
        for alpha in (0., .05):
            indexed = {r['seed']: r for r in rows if r['domain'] == domain and r['alpha'] == alpha}
            arrays[alpha] = np.array([[indexed[s]['val'], indexed[s]['test']] for s in seeds])
            mean, sd = arrays[alpha].mean(0), arrays[alpha].std(0, ddof=1)
            print(f'  {alpha:g}: {mean[0]:.2f} +/- {sd[0]:.2f}   {mean[1]:.2f} +/- {sd[1]:.2f}')
        val = paired_stats(arrays[.05][:, 0], arrays[0.][:, 0])
        test = paired_stats(arrays[.05][:, 1], arrays[0.][:, 1])
        report[domain] = dict(paired_val=val, paired_test=test,
            bonferroni_test=paired_stats(arrays[.05][:, 1], arrays[0.][:, 1], 2))
        print(f'  Delta (.05 - 0): val {val["mean"]:+.2f}; test {test["mean"]:+.2f} '
              f'[{test["low"]:+.2f}, {test["high"]:+.2f}] pp')
    atomic_json(out/'summary.json', dict(results=report, seeds=seeds, config=baseline,
        caveat='Matched ablation on previously reported confirmation seeds, not a new independent confirmation. '
        'Conditional on fixed split/teacher/condensation seed; alpha=0 is not independently tuned GRIP.'))
    print('Full results: summary.json | student_runs.csv')


if __name__ == '__main__':
    main()
