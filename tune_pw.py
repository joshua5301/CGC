# ============ Cell 1: common (SESSION = 'A' cora 5.2% | 'B' citeseer 3.6%) =============
# Node-weighted l1 partition (--pw_tau): sum_t w_t ||h_t - c_j|| with w_t = exp(-(d_t - d_min) / (tau * median d)), d_t = feature
# distance to the nearest training node. Weights enter the k-means init (faiss sample weights) and the Weiszfeld medians; the
# per-node argmin is unchanged, so this is a plain weighted quantizer (no degenerate minimiser): centres move towards the
# well-labelled region, the far region gets fewer, larger cells. The oracle localisation put the reducible label error in the far
# small cells (cora: 25% of cells = 8% of the pool hold +1.5 of +3.4). Labels: plain cell mean (lab=plain) or the same
# proximity weights in the mean (lab=prox, --prox_tau tau). tau {0.3, 1, 3} x lab x T; dropout/wd on val (wd incl. 1e-4); repeat 5.
SESSION = 'A'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'pw1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c pw_min /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--refine_teacher kernel --conv_depth 2 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 "
        "--dropout 0.5 --weight_decay 5e-4")
CELL = {'A': ('cora',     0.052, 'relu1', 0.01, 0, 2.0, [1.0, 0.5]),
        'B': ('citeseer', 0.036, 'erf',   3.0,  1, 0.2, [0.5, 0.25])}[SESSION]
TAUS = [0.3, 1.0, 3.0]
LABS = ['plain', 'prox']
DOWN = '0,0.1,0.3,0.5,0.7,0.9;1e-4,5e-4,5e-3'
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+) .*argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)')
PAT_W = re.compile(r'partition weights: .*?effective sample ([\d.]+)%')
PAT_A = re.compile(r'cell allocation by label distance.*?Q1:\s+(\d+) cells, mean size\s+([\d.]+), pool\s+([\d.]+)% \| Q2:\s+(\d+) cells, mean size\s+([\d.]+), pool\s+([\d.]+)% \| Q3:\s+(\d+) cells, mean size\s+([\d.]+), pool\s+([\d.]+)% \| Q4:\s+(\d+) cells, mean size\s+([\d.]+), pool\s+([\d.]+)%')
PAT_C = re.compile(r'cell diag: within-var ([\d.]+) \(([\d.]+)% of total\)  cells (\d+)  size min/p10/med/p90/max (\d+)/(\d+)/(\d+)/(\d+)/(\d+)')

def load():
    return pd.DataFrame([json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)])

def done(ds, r, tau, lab, temp):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.tau == tau) & (df.lab == lab) & (df.temp == temp)).any()

def run(ds, r, kernel, gamma, fn, mu, temp, tau, lab):
    extra = (f' --pw_tau {tau}' if tau > 0 else '') + (f' --prox_tau {tau}' if lab == 'prox' else '')
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --gamma {gamma} --feat_norm {fn} "
           f"--bregman {mu} --teacher_temp {temp} --repeat {REPEAT} --down_grid '{DOWN}'{extra}")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    rows = PAT_D.findall(out); g = PAT_G.search(out); w = PAT_W.search(out); a = PAT_A.search(out); c = PAT_C.search(out)
    if not rows or not g or not c or not a:
        print('FAIL', ds, r, tau, lab, temp, '\n', out[-2500:]); return
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, tau=tau, lab=lab, temp=temp, drop=float(do_), wd=float(wd_), repeat=REPEAT,
                                    test=float(te), std=float(sd), val=float(va),
                                    kl_g=float(g[1]), agree=float(g[2]), h_true=float(g[3]), h_q=float(g[4]),
                                    ess=float(w[1]) if w else 100.0,
                                    q1c=int(a[1]), q1s=float(a[2]), q1p=float(a[3]), q2c=int(a[4]), q2s=float(a[5]), q2p=float(a[6]),
                                    q3c=int(a[7]), q3s=float(a[8]), q3p=float(a[9]), q4c=int(a[10]), q4s=float(a[11]), q4p=float(a[12]),
                                    cells=int(c[3]), within=float(c[2]), size_med=int(c[6]), size_max=int(c[8]))) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"{ds:8s} {r:<6g} tau={tau:<4g} lab={lab:5s} T={temp:<4g}" + (f"  ess {w[1]}%" if w else '') +
          f"  cells by d-quartile {a[1]}/{a[4]}/{a[7]}/{a[10]} (mean size {a[2]}/{a[5]}/{a[8]}/{a[11]})  within-var {c[2]}%  agree {g[2]}%  KL {g[1]}"
          f"  val {best[5]} (do={best[0]} wd={best[1]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

ds, r, kernel, gamma, fn, mu, temps = CELL
print(f'pw1 {SESSION}: {ds} {r:g}  tau {TAUS} x lab {LABS} x T {temps}')

# ============ Cell 2: run (done() resumes) =============
for temp in temps:
    if not done(ds, r, 0.0, 'plain', temp):
        run(ds, r, kernel, gamma, fn, mu, temp, 0.0, 'plain')
for tau, lab, temp in itertools.product(TAUS, LABS, temps):
    if not done(ds, r, tau, lab, temp):
        run(ds, r, kernel, gamma, fn, mu, temp, tau, lab)

# ============ Cell 3: tables (run where both pw1_*.jsonl are present) =============
df = load()
b = df.sort_values('val', ascending=False).groupby(['ds', 'tau', 'lab', 'temp']).head(1)
bb = b.sort_values('val', ascending=False).groupby(['ds', 'tau', 'lab']).head(1)            # T on val
bb = bb.assign(cell=bb.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f} (T{x['temp']:g}, val {x['val']:.1f})", axis=1))
print('##### student test per (ds, lab) x tau   [T, dropout, wd on val; repeat 5]   tau 0 = plain GRIP')
print(bb.pivot_table(index=['ds', 'lab'], columns='tau', values='cell', aggfunc='first').to_string())
print('\n##### val-selected over (tau, lab, T) per ds  vs  tau 0')
for d, g in bb.groupby('ds'):
    top = g.sort_values('val', ascending=False).iloc[0]; base = g[g.tau == 0].iloc[0]
    print(f"  {d}: best-val tau {top.tau:g} lab {top.lab} T {top.temp:g}  test {top.test:.2f}+-{top['std']:.2f}   | tau 0: {base.test:.2f}   | oracle over grid {g.test.max():.2f}")
print('\n##### cell allocation by label-distance quartile of the cell (cells / mean size / pool share), per (ds, tau)   [lab plain, T first]')
b1 = b[(b.lab == 'plain') & (b.temp == b.temp.max())]
for _, x in b1.sort_values(['ds', 'tau']).iterrows():
    print(f"  {x.ds:8s} tau {x.tau:<4g} ess {x.ess:5.1f}%   " + '  '.join(f"Q{q}: {int(x[f'q{q}c']):3d} cells / {x[f'q{q}s']:5.1f} / {x[f'q{q}p']:4.1f}%" for q in range(1, 5)) +
          f"   within {x.within:.1f}%  agree {x.agree:.1f}%  KL {x.kl_g:.3f}")
