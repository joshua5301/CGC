# ============ Cell 1: common (SESSION = 'A' cora 5.2% | 'B' citeseer 3.6%) =============
# Far-cell-only overlapping windows (--cell_k_mult r --cell_k_far f): the top f fraction of cells by the mean distance of their
# members to the nearest training node get their label (and optionally feature) built from the r x own-size nearest pool nodes
# of their centre; every other cell is untouched. The oracle localisation put the reducible label error in exactly these cells
# (cora: far 25% of cells = 8% of the pool hold +1.5 of +3.4; citeseer: +1.0 of +0.5), and uniform windows (tune_cellk) did
# nothing because widening near cells costs what widening far cells gains. Modes: labels (features stay the medians) / full.
# r {2, 4, 8} at f 0.25, r 4 at f 0.5; x T; dropout/wd on val (wd incl. 1e-4); repeat 5.
SESSION = 'A'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'cellk2'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c cell_k_far /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--refine_teacher kernel --conv_depth 2 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 "
        "--dropout 0.5 --weight_decay 5e-4")
CELL = {'A': ('cora',     0.052, 'relu1', 0.01, 0, 2.0, [1.0, 0.5]),
        'B': ('citeseer', 0.036, 'erf',   3.0,  1, 0.2, [0.5, 0.25])}[SESSION]
GRID = [(2, 0.25), (4, 0.25), (8, 0.25), (4, 0.5)]          # (mult, far fraction)
MODES = ['labels', 'full']
DOWN = '0,0.1,0.3,0.5,0.7,0.9;1e-4,5e-4,5e-3'
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_K = re.compile(r'cell_k: K = [\d.]+ x cell size \(median (\d+), max (\d+)\) on the far [\d.]+% of cells only \((\d+) cells, ([\d.]+)% of the pool, mean size ([\d.]+) -> window ([\d.]+)\)'
                   r'.*?label argmax changed ([\d.]+)%  H ([\d.]+) -> ([\d.]+)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+) .*argmax agree ([\d.]+)%')

def load():
    return pd.DataFrame([json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)])

def done(ds, r, mult, far, mode, temp):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.mult == mult) & (df.far == far) & (df['mode'] == mode) & (df.temp == temp)).any()

def run(ds, r, kernel, gamma, fn, mu, temp, mult, far, mode):
    extra = '' if mult == 0 else f" --cell_k_mult {mult} --cell_k_far {far} --cell_k_labels_only {1 if mode == 'labels' else 0}"
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --gamma {gamma} --feat_norm {fn} "
           f"--bregman {mu} --teacher_temp {temp} --repeat {REPEAT} --down_grid '{DOWN}'{extra}")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    rows = PAT_D.findall(out); k = PAT_K.search(out); g = PAT_G.search(out)
    if not rows or (mult > 0 and not k):
        print('FAIL', ds, r, mult, far, mode, temp, '\n', out[-2500:]); return
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, mult=mult, far=far, mode=mode, temp=temp, drop=float(do_), wd=float(wd_), repeat=REPEAT,
                                    test=float(te), std=float(sd), val=float(va),
                                    far_cells=int(k[3]) if k else 0, far_pool=float(k[4]) if k else 0.0, far_size=float(k[5]) if k else None,
                                    far_window=float(k[6]) if k else None, flipped=float(k[7]) if k else 0.0, h0=float(k[8]) if k else None, h1=float(k[9]) if k else None,
                                    kl_g=float(g[1]) if g else None, agree=float(g[2]) if g else None)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    tag = 'none          ' if mult == 0 else f'r={mult} f={far:<4g} {mode:6s}'
    print(f"{ds:8s} {r:<6g} {tag} T={temp:<4g}" + (f"  far cells {k[3]} ({k[4]}% pool) size {k[5]} -> window {k[6]}  argmax changed {k[7]}%  H {k[8]}->{k[9]}" if k else '') +
          f"  val {best[5]} (do={best[0]} wd={best[1]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

ds, r, kernel, gamma, fn, mu, temps = CELL
print(f'cellk2 {SESSION}: {ds} {r:g}  (mult, far) {GRID} x modes {MODES} x T {temps}')

# ============ Cell 2: run (done() resumes) =============
for temp in temps:
    if not done(ds, r, 0, 0.0, 'none', temp):
        run(ds, r, kernel, gamma, fn, mu, temp, 0, 0.0, 'none')
for (mult, far), mode, temp in itertools.product(GRID, MODES, temps):
    if not done(ds, r, mult, far, mode, temp):
        run(ds, r, kernel, gamma, fn, mu, temp, mult, far, mode)

# ============ Cell 3: tables (run where both cellk2_*.jsonl are present) =============
df = load()
df['key'] = df.apply(lambda x: 'none' if x['mult'] == 0 else f"r{x['mult']:g} f{x['far']:g}", axis=1)
b = df.sort_values('val', ascending=False).groupby(['ds', 'key', 'mode', 'temp']).head(1)
bb = b.sort_values('val', ascending=False).groupby(['ds', 'key', 'mode']).head(1)          # T on val
bb = bb.assign(cell=bb.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f} (T{x['temp']:g}, val {x['val']:.1f})", axis=1))
print('##### student test per (ds, mode) x (mult, far)   [T, dropout, wd on val; repeat 5]')
print(bb.pivot_table(index=['ds', 'mode'], columns='key', values='cell', aggfunc='first').to_string())
print('\n##### val-selected over the grid per ds  vs  none   (oracle far 25%: cora +1.5, citeseer +1.0 at T first)')
for d, g in bb.groupby('ds'):
    top = g.sort_values('val', ascending=False).iloc[0]; base = g[g.key == 'none'].iloc[0]
    print(f"  {d}: best-val {top.key} {top['mode']} T {top.temp:g}  test {top.test:.2f}+-{top['std']:.2f}   | none {base.test:.2f}   | oracle over grid {g.test.max():.2f}")
print('\n##### far-cell windows per (ds, key)   [T first]: far cells, pool share, mean size -> window, label argmax changed, entropy before -> after')
b1 = b[(b.temp == b.temp.max()) & (b['mode'] == 'labels')]
for _, x in b1.sort_values(['ds', 'key']).iterrows():
    print(f"  {x.ds:8s} {x.key:10s} far cells {x.far_cells:4d} ({x.far_pool:4.1f}% pool)  size {x.far_size} -> window {x.far_window}  argmax changed {x.flipped:4.1f}%  H {x.h0} -> {x.h1}")
