# ============ Cell 1: common (single session; cora) =============
# Does the oracle gain grow with density? --label_oracle 1 replaces the condensed labels by the TRUE cell compositions
# (uses all labels - diagnostic ceiling). Paired with the normal labels at the protocol-A val-best condensation of each
# density (relu1, raw, basis 3000, depth 2, T 1, own gamma / mu / dropout), repeat 10. 2.6% reference (group1): 84.0 -> 85.9.
import subprocess, re, json, os, time
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'oracle2'
LOG = f'{LOGDIR}/{TAG}.jsonl'
subprocess.run('git -C /content/CGC pull', shell=True)

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--label_kernel relu1 --conv_depth 2 --feat_norm 0 --teacher_temp 1 --no_hyperpara 1 --lr 0.01 --epoch 1000 "
        "--eval_every 10 --dataset_name cora")
#         ratio   gamma  mu   dropout  wd
CELLS = [(0.013, 0.1,  1.0, 0.9, 5e-4),
         (0.026, 1.0,  5.0, 0.9, 5e-4),
         (0.052, 0.01, 2.0, 0.9, 5e-3)]
REPEAT = 10
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+) .*argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)  \(n-weighted, (\d+) cells\)')
PAT_C = re.compile(r'cell diag: within-var ([\d.]+) \(([\d.]+)% of total\).*size min/p10/med/p90/max (\d+)/(\d+)/(\d+)/(\d+)/(\d+)')

def load():
    return pd.DataFrame([json.loads(l) for l in open(LOG)]) if os.path.exists(LOG) else pd.DataFrame()

def done(r, oracle):
    df = load()
    return len(df) > 0 and ((df.ratio == r) & (df.oracle == oracle)).any()

def run(r, gamma, mu, do, wd, oracle):
    cmd = (f"python main.py {BASE} --ratio {r} --gamma {gamma} --bregman {mu} --label_oracle {oracle} "
           f"--repeat {REPEAT} --down_grid '{do:g};{wd:g}'")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    rows = PAT_D.findall(out); g = PAT_G.search(out); c = PAT_C.search(out)
    if not rows or not g:
        print('FAIL', r, oracle, '\n', out[-2000:]); return
    ex = re.search(r'expert: train ([\d.]+)%\s+val ([\d.]+)%\s+test ([\d.]+)', out)
    do_, wd_, lr_, te, sd, va = rows[0]
    rec = dict(ratio=r, oracle=oracle, gamma=gamma, mu=mu, drop=do, wd=wd, test=float(te), std=float(sd), val=float(va),
               t_test=float(ex[3]) if ex else None, kl_g=float(g[1]), agree=float(g[2]), h_true=float(g[3]), h_q=float(g[4]),
               cells=int(g[5]), size_med=int(c[5]) if c else None, size_p10=int(c[4]) if c else None)
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(f"cora {r:<6g} oracle={oracle}  cells {rec['cells']} (median size {rec['size_med']})  teacher {rec['t_test']}  "
          f"agree {rec['agree']}%  KL {rec['kl_g']:.3f}  H(true) {rec['h_true']:.3f} H(q) {rec['h_q']:.3f}  "
          f"test {te}±{sd} (val {va})  ({round(time.time() - t)}s)")

# ============ Cell 2: run =============
for r, gamma, mu, do, wd in CELLS:
    for oracle in (0, 1):
        if not done(r, oracle):
            run(r, gamma, mu, do, wd, oracle)

# ============ Cell 3: table =============
df = load()
p = df.pivot_table(index='ratio', columns='oracle', values=['test', 'std', 'agree', 'kl_g', 'h_q'], aggfunc='first')
out = pd.DataFrame({'cells': df.groupby('ratio').cells.first(), 'median cell': df.groupby('ratio').size_med.first(),
                    'teacher': df.groupby('ratio').t_test.first(),
                    'agree (normal)': p[('agree', 0)], 'KL (normal)': p[('kl_g', 0)], 'H(q) normal': p[('h_q', 0)], 'H(true)': df.groupby('ratio').h_true.first(),
                    'student normal': p[('test', 0)].round(2).astype(str) + '+-' + p[('std', 0)].round(2).astype(str),
                    'student oracle': p[('test', 1)].round(2).astype(str) + '+-' + p[('std', 1)].round(2).astype(str),
                    'oracle gain': (p[('test', 1)] - p[('test', 0)]).round(2)})
print('##### cora: label-noise ceiling per density (normal = cell-mean teacher posteriors, oracle = true cell compositions), repeat 10')
print(out.to_string())
