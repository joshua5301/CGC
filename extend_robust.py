"""Extend the existing radius sweep without changing its cache identities."""
import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import t


CAPS = [0., .001, .003, .005, .01, .02, .03, .1]


def paired_stats(values, baseline, comparisons=1):
    delta = np.asarray(values, dtype=float) - np.asarray(baseline, dtype=float)
    if delta.ndim != 1 or len(delta) < 2 or not np.isfinite(delta).all():
        raise ValueError('Need at least two finite paired observations')
    mean, std = float(delta.mean()), float(delta.std(ddof=1))
    half = float(t.ppf(1-.05/(2*comparisons), len(delta)-1)*std/np.sqrt(len(delta)))
    return dict(mean=mean, std=std, low=mean-half, high=mean+half,
                wins=int((delta > 1e-9).sum()), ties=int((abs(delta) <= 1e-9).sum()),
                losses=int((delta < -1e-9).sum()))


def report(output, repeats):
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    by_cap = {}
    for entry in manifest:
        config = entry['config']
        cap = config['radius_cap']
        if cap not in CAPS or cap in by_cap:
            raise ValueError('Unexpected or duplicate radius in manifest')
        rows = []
        for seed in range(repeats):
            path = output / 'runs' / f"{entry['id']}_d0.9_r{seed}.json"
            row = json.loads(path.read_text(encoding='utf-8'))
            if row['student_seed'] != seed or row['config'] != config:
                raise ValueError(f'Mismatched paired run: {path}')
            rows.append([100*row['val'], 100*row['test']])
        by_cap[cap] = np.asarray(rows)
    if set(by_cap) != set(CAPS):
        raise ValueError('Missing radius settings')
    baseline = by_cap[0.][3:]
    confirmation = dict(fresh_seeds=list(range(3, repeats)),
                        previous_candidate=.01, confidence_method='paired Student t; pointwise 95%',
                        scope='student seeds only, fixed condensation and dataset split', rows=[])
    print(f'\nFresh seeds 3-{repeats-1}; test at each fit\'s best validation epoch.')
    print('cap    old val  fresh val +/- sd  fresh test +/- sd  paired test delta [95% CI]')
    for cap in CAPS:
        values = by_cap[cap]
        fresh = values[3:]
        mean, std = fresh.mean(0), fresh.std(0, ddof=1)
        stat = paired_stats(fresh[:, 1], baseline[:, 1])
        adjusted = paired_stats(fresh[:, 1], baseline[:, 1], comparisons=len(CAPS)-1)
        confirmation['rows'].append(dict(cap=cap, old_val=float(values[:3, 0].mean()),
            fresh_val=float(mean[0]), fresh_test=float(mean[1]), fresh_test_std=float(std[1]),
            paired_val=paired_stats(fresh[:, 0], baseline[:, 0]), paired_test=stat,
            grid_bonferroni_test=adjusted))
        print(f'{cap:<6g} {values[:3, 0].mean():6.2f}   {mean[0]:5.2f} +/- {std[0]:.2f}     '
              f'{mean[1]:5.2f} +/- {std[1]:.2f}      {stat["mean"]:+.2f} [{stat["low"]:+.2f}, {stat["high"]:+.2f}]')
    # Selection is based only on the original three validation seeds, never test.
    selected = max(CAPS, key=lambda cap: by_cap[cap][:3, 0].mean())
    confirmation['selected_by_old_validation'] = selected
    previous = next(r for r in confirmation['rows'] if r['cap'] == .01)['paired_test']
    print(f'Previously selected cap=0.01, fresh test wins/ties/losses: '
          f'{previous["wins"]}/{previous["ties"]}/{previous["losses"]}')
    print(f'Expanded-grid winner using seeds 0-2 validation only: cap={selected:g}')
    print('CI is pointwise, conditional on fixed data/condensation; grid-adjusted intervals are in confirmation.json.')
    (output / 'confirmation.json').write_text(json.dumps(confirmation, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Full results: {output / "summary.csv"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='/content/drive/MyDrive/GRIP_cora_robust_labels')
    parser.add_argument('--raw-data-dir', default='/content/data/')
    parser.add_argument('--repeat', type=int, default=13, help='total seeds including original 0-2')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--outer-iters', type=int, default=20)
    parser.add_argument('--label-steps', type=int, default=150)
    args = parser.parse_args()
    if args.repeat < 5:
        raise ValueError('Need at least two fresh seeds beyond seeds 0-2')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(Path(__file__).with_name('sweep_distance.py')),
        '--cases', 'cora:0.052', '--methods', 'robust', '--gammas', '0.01',
        '--temperatures', '2', '--baseline-kl', '0.5', '--dropouts', '0.9',
        '--robust-caps', ','.join(map(str, CAPS)), '--repeat', str(args.repeat),
        '--epoch', str(args.epoch), '--eval-every', str(args.eval_every),
        '--outer-iters', str(args.outer_iters), '--robust-label-steps', str(args.label_steps),
        '--raw-data-dir', str(Path(args.raw_data_dir).resolve()), '--output', str(output)]
    log_path = output / f'extended_{datetime.now():%Y%m%d_%H%M%S}.log'
    print(f'Results: {output}\nLog: {log_path}')
    try:
        from IPython import get_ipython
        from IPython.display import display
        status = display('Preparing extended sweep...', display_id=True) if get_ipython() else None
    except ImportError:
        status = None
    tail = deque(maxlen=20)
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                tail.append(line)
                if line.startswith(('Condense ', 'Student ', 'SKIP completed ', 'Fit teacher ')):
                    if status:
                        status.update(line.strip())
                    else:
                        print('\r'+line.strip().ljust(120), end='', flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        print(''.join(tail))
        raise RuntimeError(f'Sweep failed: {log_path}')
    if status:
        status.update('Complete. Fresh-seed comparison below.')
    else:
        print()
    report(output, args.repeat)


if __name__ == '__main__':
    main()
