# Paste into a Colab cell after the user's clone/install/data preparation cell.
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import torch

os.chdir('/content/GRIP')
subprocess.run(['git', 'pull', '--ff-only'], check=True)
assert torch.cuda.is_available(), 'Colab에서 GPU 런타임을 선택하세요.'
print('GPU:', torch.cuda.get_device_name(0))
assert Path('/content/drive/MyDrive').is_dir(), '먼저 Google Drive를 mount 하세요.'

PRESET = os.environ.get('GRIP_SWEEP_PRESET', 'pilot')  # 'full': gamma x T x coefficient 공동 탐색
OUTPUT = '/content/drive/MyDrive/GRIP_distance_identity'
CASES = 'cora:0.052,citeseer:0.036'
Path(OUTPUT).mkdir(parents=True, exist_ok=True)
command = [
    sys.executable, '-u', 'sweep_distance.py',
    '--preset', PRESET, '--cases', CASES,
    '--methods', 'distance,grip',
    '--mus', '0.1,0.3,1,3,10', '--baseline-kl', '0.1,0.2,0.5,1,2',
    '--dropouts', '0.1,0.5,0.9', '--repeat', '3',
    '--epoch', '1000', '--eval-every', '10',
    '--outer-iters', '20', '--median-iters', '30', '--batch-size', '32',
    '--raw-data-dir', '/content/data/', '--output', OUTPUT, '--device', 'cuda',
]
log_path = Path(OUTPUT) / f'{PRESET}_{datetime.now():%Y%m%d_%H%M%S}.log'
print('Results:', OUTPUT, '\nLog:', log_path)
with log_path.open('w', encoding='utf-8') as logfile:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='')
            logfile.write(line)
            logfile.flush()
        code = process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise
if code:
    raise RuntimeError(f'Sweep failed (exit {code}). See {log_path}')

import pandas as pd
display(pd.read_csv(Path(OUTPUT) / 'summary.csv'))
display(pd.read_json(Path(OUTPUT) / 'best.json'))
