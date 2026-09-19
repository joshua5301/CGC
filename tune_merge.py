# ============ Cell 1: common (SESSION = 'A' cora 5.2% | 'B' citeseer 3.6%) =============
# Post-hoc cell merging (--merge_lambda): GRIP builds m cells as usual; then cell pairs (knn-nearest centres) are merged
# greedily while  DeltaJ0 + lambda * nbar * DeltaU < 0,  U = sum_j mean_{t in j} u_t.  Merged label = size-weighted mean,
# merged centre = geometric median of the union. Final cells m' <= m.
#   u = const : DeltaU = -1 per merge -> plain J0-greedy agglomeration (control: "fewer cells")
#   u = dist  : distance to the nearest training node (teacher error grows with it) -> "merge the uncertain cells"
# Control 2: plain GRIP at ratio r/2 and r/4 with the same settings (the density curve at the same m').
# lambda {0.3, 1, 3, 10} x u x T; dropout/wd on val; repeat 5.
SESSION = 'A'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'merge1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c merge_knn /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--refine_teacher kernel --conv_depth 2 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 "
        "--dropout 0.5 --weight_decay 5e-4")
CELL = {'A': ('cora',     0.052, 'relu1', 0.01, 0, 2.0, [1.0, 0.5]),
        'B': ('citeseer', 0.036, 'erf',   3.0,  1, 0.2, [0.5, 0.25])}[SESSION]
LAMS = [0.3, 1.0, 3.0, 10.0]
US = ['const', 'dist']
DOWN = '0,0.1,0.3,0.5,0.7,0.9;5e-4,5e-3'
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+) .*argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)')
PAT_M = re.compile(r'merge: lambda [\d.]+ u \w+ knn \d+  cells (\d+) -> (\d+) \((\d+) merges\)  J0 ([\d.]+) -> ([\d.]+)  U ([\d.]+) -> ([\d.]+)')
PAT_C = re.compile(r'cell diag: within-var ([\d.]+) \(([\d.]+)% of total\)  cells (\d+)  size min/p10/med/p90/max (\d+)/(\d+)/(\d+)/(\d+)/(\d+)')

def load():
    return pd.DataFrame([json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)])

def done(ds, r, u, lam, temp):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.u == u) & (df.lam == lam) & (df.temp == temp)).any()

def run(ds, r, kernel, gamma, fn, mu, temp, u, lam):
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --gamma {gamma} --feat_norm {fn} "
           f"--bregman {mu} --teacher_temp {temp} --merge_lambda {lam} --merge_u {u} --repeat {REPEAT} --down_grid '{DOWN}'")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    rows = PAT_D.findall(out); g = PAT_G.search(out); m = PAT_M.search(out); c = PAT_C.search(out)
    if not rows or not g or not c:
        print('FAIL', ds, r, u, lam, temp, '\n', out[-2500:]); return
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, u=u, lam=lam, temp=temp, drop=float(do_), wd=float(wd_), repeat=REPEAT,
                                    test=float(te), std=float(sd), val=float(va),
                                    kl_g=float(g[1]), agree=float(g[2]), h_true=float(g[3]), h_q=float(g[4]),
                                    cells=int(c[3]), within=float(c[2]), size_min=int(c[4]), size_med=int(c[6]), size_max=int(c[8]),
                                    merges=int(m[3]) if m else 0, j0_a=float(m[4]) if m else None, j0_b=float(m[5]) if m else None,
                                    u_a=float(m[6]) if m else None, u_b=float(m[7]) if m else None)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"{ds:8s} {r:<6g} u={u:5s} lam={lam:<4g} T={temp:<4g}  cells {c[3]:>4s}" + (f" ({m[3]} merges, J0 {m[4]}->{m[5]}, U {m[6]}->{m[7]})" if m else '') +
          f"  size {c[4]}/{c[6]}/{c[8]}  within-var {c[2]}%  agree {g[2]}%  KL {g[1]}  val {best[5]} (do={best[0]} wd={best[1]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

ds, r, kernel, gamma, fn, mu, temps = CELL
print(f'merge1 {SESSION}: {ds} {r:g}  lambda {LAMS} x u {US} x T {temps}  + controls at ratio {r / 2:g}, {r / 4:g}')

# ============ Cell 2: run (done() resumes) =============
for temp in temps:                                     # lambda 0 reference (plain GRIP at m)
    if not done(ds, r, 'none', 0.0, temp):
        run(ds, r, kernel, gamma, fn, mu, temp, 'none', 0.0)
for lam, u, temp in itertools.product(LAMS, US, temps):
    if not done(ds, r, u, lam, temp):
        run(ds, r, kernel, gamma, fn, mu, temp, u, lam)
for rr, temp in itertools.product([r / 2, r / 4], temps):   # density-curve control: plain GRIP at fewer cells
    if not done(ds, rr, 'none', 0.0, temp):
        run(ds, rr, kernel, gamma, fn, mu, temp, 'none', 0.0)

# ============ Cell 3: tables (run where both merge1_*.jsonl are present) =============
df = load()
b = df.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'u', 'lam', 'temp']).head(1)
bb = b.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'u', 'lam']).head(1)      # T on val
bb = bb.assign(cell=bb.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f} (T{x['temp']:g}, m'={x['cells']})", axis=1))
print("##### student test per (ds, u) x lambda   [T, dropout, wd on val; repeat 5]   m' = final cells")
print(bb[bb.u != 'none'].pivot_table(index=['ds', 'u'], columns='lam', values='cell', aggfunc='first').to_string())
print('\n##### references: plain GRIP at the same settings (lambda 0), by ratio')
print(bb[bb.u == 'none'].pivot_table(index='ds', columns='ratio', values='cell', aggfunc='first').to_string())
print("\n##### merge vs the density curve: every merged run and the plain-GRIP run with the nearest m'")
ref = bb[bb.u == 'none']
for _, x in bb[bb.u != 'none'].sort_values(['ds', 'u', 'lam']).iterrows():
    rf = ref[ref.ds == x.ds].iloc[(ref[ref.ds == x.ds].cells - x.cells).abs().argsort()[:1]].iloc[0]
    print(f"  {x.ds:8s} u={x.u:5s} lam={x.lam:<4g}  m'={x.cells:4d}  test {x.test:.2f}+-{x['std']:.2f} (val {x.val:.2f})   | plain GRIP m={rf.cells:4d} (ratio {rf.ratio:g}): {rf.test:.2f}+-{rf['std']:.2f}   diff {x.test - rf.test:+.2f}")
print('\n##### val-selected over (u, lambda, T) per ds  vs  lambda 0 at the full budget')
for d, g in bb.groupby('ds'):
    gm = g[(g.ratio == g.ratio.max())]
    top = gm.sort_values('val', ascending=False).iloc[0]; base = gm[gm.u == 'none'].iloc[0]
    print(f"  {d}: best-val u {top.u} lambda {top.lam:g} T {top.temp:g} m'={top.cells}  test {top.test:.2f}+-{top['std']:.2f}   | lambda 0: {base.test:.2f}   | oracle over grid {gm.test.max():.2f}")
print("\n##### partition statistics per (ds, u) x lambda   [T = first]: cells, merges, J0 before -> after, U before -> after, within-cell var %, agreement, group KL, size med/max")
b1 = b[b.temp == b.temp.max()]
for mcol in ['cells', 'merges', 'j0_a', 'j0_b', 'u_a', 'u_b', 'within', 'agree', 'kl_g', 'size_med', 'size_max']:
    print(mcol.ljust(10), b1[b1.u != 'none'].pivot_table(index=['ds', 'u'], columns='lam', values=mcol, aggfunc='first').round(3).to_string().replace('\n', '\n' + ' ' * 10))
