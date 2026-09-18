# ============ Cell 1: common (SESSION = 'all' | 'A' arxiv | 'C' cora + citeseer) =============
# Which teacher quantity predicts the student? For every (gamma, T) of the teacher grid we log, on the SAME partition,
#   node level : teacher argmax accuracy and NLL on the pool
#   group level: KL(true cell composition || condensed label q_j), n-weighted; entropies of both
#   student    : downstream GCN val/test (reduced recipe grid, repeat 3)
# plus the label-estimation ceiling: --label_oracle 1 (q_j := true composition, diagnostic only).
# One density per dataset at the main-table condensation (raw space, l1 k-medians, mu 0.3).
SESSION = 'all'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'group1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c label_oracle /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--conv_depth 2 --feat_norm 0 --bregman 0.3 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 "
        "--dropout 0.5 --weight_decay 5e-4")
#          ds          ratio   kernel   gammas
CELLS = {'A': [('arxiv',    0.0025, 'erf',   [1e-4, 1e-3, 1e-2, 1e-1])],
         'C': [('cora',     0.026,  'relu1', [1e-2, 1e-1, 1.0, 10.0, 30.0]),
               ('citeseer', 0.018,  'erf',   [1e-2, 1e-1, 1.0, 10.0, 30.0])]}
CELLS = CELLS['A'] + CELLS['C'] if SESSION == 'all' else CELLS[SESSION]
TEMPS = [1.0, 0.5, 0.25]
ORACLE_GAMMA = {'arxiv': 1e-3, 'cora': 1.0, 'citeseer': 1.0}
DOWN = '0,0.1,0.3,0.5,0.7,0.9;5e-4'
REPEAT = 3
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+)  L1 ([\d.]+)  argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)'
                   r'  \(n-weighted, (\d+) cells\)(?:  \| node teacher on the pool: NLL ([\d.]+)  acc ([\d.]+)%  H ([\d.]+))?')

def load():
    recs = [json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)]
    return pd.DataFrame(recs)

def done(ds, r, gamma, temp, oracle):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.gamma == gamma) & (df.temp == temp)
                            & (df.oracle == oracle)).any()

def run(ds, r, kernel, gamma, temp, oracle=0):
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --gamma {gamma} "
           f"--teacher_temp {temp} --label_oracle {oracle} --repeat {REPEAT} --down_grid '{DOWN}'")
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC')
    out = p.stdout + p.stderr
    rows = PAT_D.findall(out)
    g = PAT_G.search(out)
    if not rows or not g:
        print('FAIL', ds, r, gamma, temp, oracle, '\n', out[-1500:]); return
    ex = re.search(r'expert: train ([\d.]+)%(?:\s+val ([\d.]+)%)?(?:\s+test ([\d.]+))?', out)
    gd = dict(kl_g=float(g[1]), l1_g=float(g[2]), agree=float(g[3]), h_true=float(g[4]), h_q=float(g[5]), cells=int(g[6]),
              nll_n=float(g[7]) if g[7] else None, acc_n=float(g[8]) if g[8] else None, h_n=float(g[9]) if g[9] else None)
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, kernel=kernel, gamma=gamma, temp=temp, oracle=oracle,
                                    drop=float(do_), wd=float(wd_), repeat=REPEAT,
                                    test=float(te), std=float(sd), val=float(va),
                                    t_test=float(ex[3]) if ex and ex[3] else None, **gd)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"{ds:8s} r={r:<7g} g={gamma:<5g} T={temp:<4g}{' ORACLE' if oracle else ''}  "
          f"teacher acc {gd['acc_n']} NLL {gd['nll_n']}  group KL {gd['kl_g']:.4f} agree {gd['agree']:.1f}% "
          f"H(true) {gd['h_true']:.3f} H(q) {gd['h_q']:.3f}  val {best[5]} test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

print(f'{SESSION}: {[(c[0], c[1]) for c in CELLS]}  T {TEMPS}')

# ============ Cell 2: run (done() resumes) =============
for ds, r, kernel, gammas in CELLS:
    for gamma, temp in itertools.product(gammas, TEMPS):
        if not done(ds, r, gamma, temp, 0):
            run(ds, r, kernel, gamma, temp, 0)
    if not done(ds, r, ORACLE_GAMMA[ds], 1.0, 1):
        run(ds, r, kernel, ORACLE_GAMMA[ds], 1.0, 1)

# ============ Cell 3: tables (run where all group1_*.jsonl are present) =============
df = load(); assert len(df), 'no logs'
b = df.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'gamma', 'temp', 'oracle']).head(1)
cols = ['gamma', 'temp', 'acc_n', 'nll_n', 'h_n', 'kl_g', 'l1_g', 'agree', 'h_true', 'h_q', 'val', 'test', 'std']
for ds in b.ds.unique():
    s = b[(b.ds == ds) & (b.oracle == 0)].sort_values(['gamma', 'temp'])
    o = b[(b.ds == ds) & (b.oracle == 1)]
    print(f'\n##### {ds}: teacher grid (node metrics on the pool, group metrics n-weighted over cells, student val-selected)')
    print(s[cols].round(4).to_string(index=False))
    if len(o):
        print(f'      ORACLE labels (true cell composition): test {o.test.iloc[0]:.2f}+-{o["std"].iloc[0]:.2f}  '
              f'vs best of grid {s.test.max():.2f}  (H(true) {o.h_true.iloc[0]:.3f})')
    print('      Spearman rank correlation of student test with:  '
          + '  '.join(f'{m} {s.test.corr(s[m], method="spearman"):+.2f}' for m in ['acc_n', 'nll_n', 'kl_g', 'l1_g', 'agree']))
print('\n##### student test per (ds) x gamma (val-best T)')
print(b[b.oracle == 0].sort_values('val', ascending=False).groupby(['ds', 'gamma']).head(1)
      .pivot_table(index='ds', columns='gamma', values='test').round(2).to_string())
print('\n##### group KL per (ds) x gamma (val-best T)')
print(b[b.oracle == 0].sort_values('val', ascending=False).groupby(['ds', 'gamma']).head(1)
      .pivot_table(index='ds', columns='gamma', values='kl_g').round(4).to_string())
