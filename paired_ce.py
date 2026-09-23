"""Paired student seeds on one saved GRIP condensation; no new selection/sweep."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.data import Data

from sweep_distance import atomic_json, code_digest, data_digest, digest
from src.dataloader import get_dataset, set_dataset
from src.models import GCN
from src.utils import model_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', default='/content/drive/MyDrive/GRIP_cora_weighted_ce_full')
    parser.add_argument('--output', default='/content/drive/MyDrive/GRIP_cora_paired_ce')
    parser.add_argument('--raw-data-dir', default='/content/data/')
    parser.add_argument('--seed-start', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=10)
    args = parser.parse_args()
    if args.repeat < 2 or args.seed_start < 0 or not 1 <= args.eval_every <= args.epoch:
        raise ValueError('Need repeat >= 2, seed-start >= 0, and 1 <= eval-every <= epoch')
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = Path(args.source_dir)
    entries = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    matches = [e for e in entries if all(e['config'].get(k) == v for k, v in
               dict(dataset='cora', ratio=.052, method='grip', gamma=.01, T=2., coefficient=.5, seed=0).items())]
    if len(matches) != 1:
        raise ValueError(f'Expected one saved GRIP configuration, found {len(matches)} in {source}')
    artifact_path = source / 'condensed' / (matches[0]['id'] + '.pt')
    artifact = torch.load(artifact_path, map_location='cpu', weights_only=True)
    if artifact['config'] != matches[0]['config']:
        raise ValueError('Artifact and manifest do not match')
    output = Path(args.output)
    (output / 'runs').mkdir(parents=True, exist_ok=True)
    log_path = output / f'paired_{datetime.now():%Y%m%d_%H%M%S}.log'
    print(f'GPU: {torch.cuda.get_device_name(0)}\nResults: {output}\nLog: {log_path}')
    train_args = SimpleNamespace(dataset_name='cora', ratio=.052, device='cuda',
                                 raw_data_dir=str(Path(args.raw_data_dir).resolve()) + '/',
                                 n_dim=256, lr=.01, weight_decay=5e-4,
                                 epoch=args.epoch, eval_every=args.eval_every)
    with log_path.open('w', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            seed_everything(0)
            train_args, data, data_val, data_test = set_dataset(train_args, get_dataset(train_args))
        fingerprint = digest([data_digest(d) for d in [data, data_val, data_test] if d is not None])
        if fingerprint != artifact['config']['data']:
            raise ValueError('Original data/splits differ from saved condensation')
        count = len(artifact['x'])
        assignment = artifact['assign']
        if assignment.ndim != 1 or len(assignment) != data.num_nodes or assignment.min() < 0 or assignment.max() >= count:
            raise ValueError('Invalid saved assignment')
        counts = torch.bincount(assignment, minlength=count).to('cuda')
        ids = torch.arange(count, device='cuda')
        graph = Data(x=artifact['x'].to('cuda'), y=artifact['y'].to('cuda'),
                     edge_index=torch.stack([ids, ids]), edge_attr=torch.ones(count, device='cuda'),
                     train_mask=torch.ones(count, dtype=torch.bool, device='cuda'))
        config = dict(artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                      source=code_digest(), runner=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      data=fingerprint, epoch=args.epoch, eval_every=args.eval_every,
                      dropout=.9, student='GCN-2-256', lr=.01, weight_decay=5e-4,
                      torch=str(torch.__version__), gpu=torch.cuda.get_device_name(0), threads=4)
        key = digest(config)
        atomic_json(output / 'manifest.json', dict(config=config, artifact=str(artifact_path),
                    seed_start=args.seed_start, repeat=args.repeat))
        rows = []
        for seed in range(args.seed_start, args.seed_start + args.repeat):
            for loss in ['uniform', 'cell-size']:
                path = output / 'runs' / f'{key}_{seed}_{loss}.json'
                if path.exists():
                    row = json.loads(path.read_text(encoding='utf-8'))
                else:
                    # Reset BOTH initialization and dropout RNG for each paired fit.
                    seed_everything(seed)
                    model = GCN(data.num_features, 256, train_args.num_class, 2, .9).to('cuda')
                    with redirect_stdout(log), redirect_stderr(log):
                        print(f'Student seed={seed} loss={loss}', flush=True)
                        val, test = model_training(model, train_args, data, graph, data_val, data_test,
                                                  sample_weight=counts if loss == 'cell-size' else None)
                    row = dict(seed=seed, loss=loss, val=100 * val, test=100 * test)
                    atomic_json(path, row)
                    del model
                rows.append(row)
            print(f'\rCompleted pairs: {seed - args.seed_start + 1}/{args.repeat}', end='', flush=True)
    print()
    # No hyperparameter or seed selection. Test is taken at each fit's best validation epoch.
    import csv
    with (output / 'paired.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['seed', 'loss', 'val', 'test'])
        writer.writeheader()
        writer.writerows(rows)
    print('Loss             Val mean +/- std   Test mean +/- std')
    for loss in ['uniform', 'cell-size']:
        values = np.array([[r['val'], r['test']] for r in rows if r['loss'] == loss])
        mean, std = values.mean(0), values.std(0, ddof=1)
        print(f'{loss:16s} {mean[0]:.2f} +/- {std[0]:.2f}      {mean[1]:.2f} +/- {std[1]:.2f}')
    pairs = np.array([[rows[i+1]['val'] - rows[i]['val'], rows[i+1]['test'] - rows[i]['test']]
                      for i in range(0, len(rows), 2)])
    print('Paired change (weighted - uniform), percentage points:')
    for col, metric in enumerate(['val', 'test']):
        d = pairs[:, col]
        print(f'{metric}: {d.mean():+.2f} +/- {d.std(ddof=1):.2f}; wins/ties/losses: '
              f'{(d > 1e-9).sum()}/{(np.abs(d) <= 1e-9).sum()}/{(d < -1e-9).sum()}')
    print('Seed   Delta val   Delta test')
    for seed, delta in zip(range(args.seed_start, args.seed_start + args.repeat), pairs):
        print(f'{seed:4d}   {delta[0]:+8.2f}   {delta[1]:+9.2f}')


if __name__ == '__main__':
    main()
