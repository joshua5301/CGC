# ============ Cell 1: common (SESSION = 'A' arxiv 0.25% | 'B' citeseer 3.6% | 'C' cora 5.2%) =============
# Proximity-to-supervision weights in the cell mean (--prox_tau tau): w_t = exp(-(d_t - d_min) / (tau * median d)), floor 0.1,
# d_t = distance in H to the nearest TRAINING node. Basis (dist_diag): the teacher's accuracy falls with d_t on every dataset
# (feature-distance Q1 -> Q4: cora 88 -> 76, citeseer 81 -> 61, arxiv 79 -> 62). Paired with tau = inf (plain mean) at the
# protocol-A val-best condensation of the weakest high-density cells; tau x T x dropout/wd on val, repeat 5.
SESSION = 'A'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'prox1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c prox_min /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--conv_depth 2 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 --dropout 0.5 --weight_decay 5e-4")
#          ds          ratio   kernel   gamma  fn  mu    temps          wds
CELLS = {'A': ('arxiv',    0.0025, 'relu1', 1e-4, 0, 0.2, [1.0, 0.25],      '5e-4'),
         'B': ('citeseer', 0.036,  'erf',   3.0,  1, 0.2, [1.0, 0.5, 0.25], '5e-4,5e-3'),
         'C': ('cora',     0.052,  'relu1', 0.01, 0, 2.0, [1.0, 0.5],       '5e-4,5e-3')}[SESSION]
TAUS = [0.0, 2.0, 1.0, 0.5, 0.25]          # 0 = no weighting
DOS = '0,0.1,0.3,0.5,0.7,0.9' if SESSION != 'A' else '0,0.1,0.3,0.5,0.7'
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+) .*argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)')
PAT_W = re.compile(r'prox weights: .*?w mean ([\d.]+) .*?effective sample ([\d.]+)%')

def load():
    return pd.DataFrame([json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)])

def done(ds, r, tau, temp):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.tau == tau) & (df.temp == temp)).any()

def run(ds, r, kernel, gamma, fn, mu, temp, tau, wds):
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --gamma {gamma} --feat_norm {fn} "
           f"--bregman {mu} --teacher_temp {temp} --prox_tau {tau} --repeat {REPEAT} --down_grid '{DOS};{wds}'")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    rows = PAT_D.findall(out); g = PAT_G.search(out); w = PAT_W.search(out)
    if not rows or not g:
        print('FAIL', ds, r, tau, temp, '\n', out[-2000:]); return
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, kernel=kernel, gamma=gamma, fn=fn, mu=mu, temp=temp, tau=tau,
                                    drop=float(do_), wd=float(wd_), repeat=REPEAT, test=float(te), std=float(sd), val=float(va),
                                    kl_g=float(g[1]), agree=float(g[2]), h_true=float(g[3]), h_q=float(g[4]),
                                    w_mean=float(w[1]) if w else 1.0, eff=float(w[2]) if w else 100.0)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"{ds:8s} {r:<6g} tau={tau:<4g} T={temp:<4g}" + (f"  w mean {w[1]} eff {w[2]}%" if w else '') +
          f"  agree {g[2]}%  KL {g[1]}  H(q) {g[4]}  val {best[5]} (do={best[0]} wd={best[1]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

ds, r, kernel, gamma, fn, mu, temps, wds = CELLS
print(f'prox1 {SESSION}: {ds} {r:g}  taus {TAUS}  T {temps}')

# ============ Cell 2: run (done() resumes) =============
for tau, temp in itertools.product(TAUS, temps):
    if not done(ds, r, tau, temp):
        run(ds, r, kernel, gamma, fn, mu, temp, tau, wds)

# ============ Cell 3: tables (run where all prox1_*.jsonl are present) =============
df = load()
b = df.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'tau', 'temp']).head(1)
b = b.assign(cell=b.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1))
print('##### student test per (ds, T) x tau   [val-selected dropout/wd, repeat 5]   (tau 0 = plain cell mean)')
print(b.pivot_table(index=['ds', 'temp'], columns='tau', values='cell', aggfunc='first').to_string())
bb = b.sort_values('val', ascending=False).groupby(['ds', 'tau']).head(1)
print('\n##### T selected on val, per ds x tau')
print(bb.assign(cell=bb.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f} (T{x['temp']:g})", axis=1)).pivot_table(index='ds', columns='tau', values='cell', aggfunc='first').to_string())
print('\n##### val-selected over (tau, T) per ds  vs  tau 0')
for d, g in b.groupby('ds'):
    top = g.sort_values('val', ascending=False).iloc[0]; base = g[g.tau == 0].sort_values('val', ascending=False).iloc[0]
    print(f"  {d}: best-val tau {top.tau:g} T {top.temp:g}  test {top.test:.2f}+-{top['std']:.2f}   | tau 0 (T {base.temp:g}): {base.test:.2f}   | oracle over grid {g.test.max():.2f}")
for m, title in [('agree', 'cell argmax agreement [%]'), ('kl_g', 'group KL(true||q)'), ('h_q', 'H(q)'), ('eff', 'effective sample size [%]')]:
    print(f'\n##### {title}   [T 1]')
    print(b[b.temp == 1.0].pivot_table(index='ds', columns='tau', values=m, aggfunc='first').round(3).to_string())
