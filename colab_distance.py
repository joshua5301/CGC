# Paste into a Colab cell after the user's clone/install/data preparation cell.
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import torch
from collections import deque
from IPython.display import display

os.chdir('/content/GRIP')
pull = subprocess.run(['git', 'pull', '--ff-only'], capture_output=True, text=True)
if pull.returncode:
    raise RuntimeError(pull.stdout + pull.stderr)
assert torch.cuda.is_available(), 'Colab에서 GPU 런타임을 선택하세요.'
print('GPU:', torch.cuda.get_device_name(0))
assert Path('/content/drive/MyDrive').is_dir(), '먼저 Google Drive를 mount 하세요.'

PRESET = os.environ.get('GRIP_SWEEP_PRESET', 'pilot')  # pilot / full / rho / weighted
if PRESET not in ('pilot', 'full', 'rho', 'weighted'):
    raise ValueError('GRIP_SWEEP_PRESET must be pilot, full, rho, or weighted')
OUTPUT = '/content/drive/MyDrive/' + ('GRIP_mpnn_rho' if PRESET == 'rho' else 'GRIP_mpnn_identity')
if PRESET == 'weighted':
    OUTPUT = '/content/drive/MyDrive/GRIP_cora_weighted_ce_full'
CASES = 'cora:0.052' if PRESET == 'weighted' else 'cora:0.052,citeseer:0.036'
Path(OUTPUT).mkdir(parents=True, exist_ok=True)
command = [
    sys.executable, '-u', 'sweep_distance.py',
    '--preset', 'pilot' if PRESET == 'rho' else ('full' if PRESET == 'weighted' else PRESET), '--cases', CASES,
    '--methods', 'mpnn' if PRESET == 'rho' else ('grip' if PRESET == 'weighted' else 'mpnn,raw,grip'),
    '--mus', '0.3,1,3,10' if PRESET == 'rho' else '0.1,0.3,1,3,10', '--baseline-kl', '0.1,0.2,0.5,1,2',
    '--dropouts', '0.1,0.5,0.9', '--repeat', '3',
    '--epoch', '1000', '--eval-every', '10',
    '--outer-iters', '20', '--median-iters', '30', '--batch-size', '32',
    '--raw-data-dir', '/content/data/', '--output', OUTPUT, '--device', 'cuda',
]
if PRESET == 'weighted':
    command += ['--student-loss', 'cell-size']
if PRESET == 'rho':
    # Full sweep's best teacher setting shared by mpnn/raw on Cora; Citeseer unchanged.
    # gamma stays at the per-dataset default (.01 for Cora, .1 for Citeseer).
    command += ['--rhos', '0,0.05,0.1,0.2,0.35,0.5', '--temperatures', '0.2']
log_path = Path(OUTPUT) / f'{PRESET}_{datetime.now():%Y%m%d_%H%M%S}.log'
print('Results:', OUTPUT, '\nLog:', log_path)
status = display('실험 준비 중…', display_id=True)
tail = deque(maxlen=20)
with log_path.open('w', encoding='utf-8') as logfile:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    try:
        for line in process.stdout:
            logfile.write(line)
            logfile.flush()
            tail.append(line)
            # Replace one progress item; full epoch/objective logs stay on Drive.
            if line.startswith(('Condense ', 'Student ', 'SKIP completed ', 'Fit teacher ')) or 'condensations,' in line:
                status.update(line.strip())
        code = process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        status.update('중단됨. 같은 셀을 다시 실행하면 저장된 결과에서 재개합니다.')
        raise
if code:
    status.update('실험 실패. 마지막 로그를 확인하세요.')
    print(''.join(tail))
    raise RuntimeError(f'Sweep failed (exit {code}). See {log_path}')

import pandas as pd
status.update('완료 — ' + ('rho별 ' if PRESET == 'rho' else '') + 'validation 최고 설정 (동률 포함).')
results = pd.read_csv(Path(OUTPUT) / 'summary.csv')
group = ['dataset', 'ratio', 'method']
if PRESET == 'rho':
    group += ['rho']
top = results.groupby(group)['val'].transform('max')
# Ignore floating-point aggregation noise when displaying genuine validation ties.
selected = results[results['val'].round(10) == top.round(10)].copy()
selected['test +/- std'] = selected.apply(lambda r: f"{r['test']:.2f} +/- {r['test_std']:.2f}", axis=1)
selected['val'] = selected['val'].map(lambda value: f'{value:.2f}')
columns = ['dataset', 'ratio', 'method', 'gamma', 'T', 'rho', 'coefficient', 'loss', 'dropout', 'val', 'test +/- std']
print(selected[columns].sort_values(group + ['coefficient', 'dropout']).to_string(
    index=False, float_format=lambda value: f'{value:.3g}'))
print('전체 결과: summary.csv | 상세 로그:', log_path.name)
