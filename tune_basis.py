# ============ Cell 1: common (single session) =============
# Nystrom basis of the kernel teacher: k-means centroids of the pool (new default, --basis_mode kmeans) vs uniformly
# sampled rows (legacy, --basis_mode random), paired at the ax1 val-best condensation of each arxiv density (raw space,
# recipe fixed to the selected dropout, repeat 10, condensation seeds 0 and 1). On cora / citeseer 3000 >= N so the two coincide.
SESSION = 'A'
import subprocess, re, json, os, time, glob
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'basis1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c basis_mode /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--conv_depth 2 --feat_norm 0 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 --weight_decay 5e-4")
#          ds        ratio   kernel   gamma  mu   T     dropout     (ax1 val-best per density, raw space)
CELLS = {'A': [('arxiv',  0.0005, 'erf',   1e-3, 0.5, 0.25, 0.7),
               ('arxiv',  0.0025, 'relu1', 1e-4, 0.2, 0.25, 0.3),
               ('arxiv',  0.005,  'erf',   1e-3, 0.5, 1.0,  0.1)]}
CELLS = CELLS['A']
MODES = ['kmeans', 'random']
SEEDS = [0, 1]
REPEAT = 10
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')

def load():
    recs = [json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)]
    return pd.DataFrame(recs)

def done(ds, r, mode, seed):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df['mode'] == mode) & (df.seed == seed)).any()

def run(ds, r, kernel, gamma, mu, temp, do, mode, seed):
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} "
           f"--gamma {gamma} --bregman {mu} --teacher_temp {temp} --basis_mode {mode} --seed {seed} "
           f"--repeat {REPEAT} --down_grid '{do:g};5e-4'")
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC')
    out = p.stdout + p.stderr
    rows = PAT_D.findall(out)
    if not rows:
        print('FAIL', ds, r, mode, seed, '\n', out[-2000:]); return
    ex = re.search(r'expert: train ([\d.]+)%(?:\s+val ([\d.]+)%)?(?:\s+test ([\d.]+))?', out)
    ct = re.search(r'Condensation time: ([0-9.]+)', out); ct = float(ct[1]) if ct else float('nan')
    lo = re.search(r'loss: ([\d.]+)  gnorm', out)
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, mode=mode, seed=seed, kernel=kernel, gamma=gamma, mu=mu, temp=temp,
                                    drop=float(do_), wd=float(wd_), repeat=REPEAT, test=float(te), std=float(sd), val=float(va),
                                    t_train=float(ex[1]) if ex else None, t_test=float(ex[3]) if ex and ex[3] else None,
                                    t_loss=float(lo[1]) if lo else None, cond_s=ct)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"{ds:8s} r={r:<7g} basis={mode:6s} seed={seed}  teacher train {ex[1] if ex else '?'} test {(ex[3] if ex and ex[3] else '-')}  "
          f"val {best[5]} (do={best[0]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s, cond {ct:.0f}s)")

print(f'{SESSION}: {[(c[0], c[1]) for c in CELLS]}  modes {MODES} x seeds {SEEDS}')

# ============ Cell 2: run (done() resumes) =============
for ds, r, kernel, gamma, mu, temp, do in CELLS:
    for mode in MODES:
        for seed in SEEDS:
            if not done(ds, r, mode, seed):
                run(ds, r, kernel, gamma, mu, temp, do, mode, seed)

# ============ Cell 3: table (run where all basis1_*.jsonl are present) =============
df = load(); assert len(df), 'no logs'
b = df.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'mode', 'seed']).head(1)
print('##### student test per (ds, ratio) x basis mode   [ax1 recipe, repeat 10; mean/min/max over condensation seeds 0, 1]')
print(b.pivot_table(index=['ds', 'ratio'], columns='mode', values='test', aggfunc=['mean', 'min', 'max']).round(2).to_string())
print('\n##### teacher test accuracy (train on inductive graphs) per (ds, ratio) x basis mode, mean over seeds')
t = b.assign(t=b.t_test.fillna(b.t_train)).pivot_table(index=['ds', 'ratio'], columns='mode', values='t', aggfunc=['mean', 'min', 'max']).round(2)
print(t.to_string())
print('\n##### teacher training loss and condensation time per mode')
print(b.pivot_table(index=['ds', 'ratio'], columns='mode', values=['t_loss', 'cond_s'], aggfunc='mean').round(3).to_string())
