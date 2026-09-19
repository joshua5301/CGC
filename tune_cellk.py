# ============ Cell 1: common (single session; cora 5.2%) =============
# Overlapping cells: each of the 140 condensed nodes is built from the K nearest pool nodes of its centre instead of its
# Voronoi cell. K is either fixed (9 ~ the current median cell, 40 ~ the 1.3% cell size) or a MULTIPLE of the cell's own
# Voronoi size (--cell_k_mult r: big cells keep big windows, small cells grow proportionally). Two variants:
#   full         (--cell_k K)                      features AND labels from the K-neighbourhood (covering, not a partition)
#   labels only  (--cell_k K --cell_k_labels_only) labels from the K-neighbourhood, features stay the cell medians
# Motivation: the oracle-label gain grows with density (cora +0.6 / +1.5 / +2.5) because 9-node cells no longer average the
# teacher's errors away; cell-level kNN smoothing (m nearest cells) failed (-2.7 .. -6, boundary bias). Node-level windows cut
# by distance instead. Note: the 'group diag' line is computed BEFORE cell_k (Voronoi labels), so only the student is read here.
import subprocess, re, json, os, time, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'cellk1'
LOG = f'{LOGDIR}/{TAG}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c cell_k_labels_only /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--label_kernel relu1 --gamma 0.01 --bregman 2 --feat_norm 0 --conv_depth 2 --no_hyperpara 1 --lr 0.01 --epoch 1000 "
        "--eval_every 10 --dropout 0.5 --weight_decay 5e-4 --dataset_name cora --ratio 0.052")
KS, MODES, TEMPS = [9, 20, 40, 80], ['full', 'labels'], [1.0, 0.5]
MULTS = [2, 4, 8]                  # window = r x own cell size
DOWN = '0,0.1,0.3,0.5,0.7,0.9;5e-4,5e-3'
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_C = re.compile(r'cell_k: (K=\d+|K = [^)]*\))\S*.*?pool covered ([\d.]+)%  multiplicity ([\d.]+)  label argmax changed ([\d.]+)%  H ([\d.]+) -> ([\d.]+)')

def load():
    return pd.DataFrame([json.loads(l) for l in open(LOG)]) if os.path.exists(LOG) else pd.DataFrame()

def done(K, mode, temp, mult=0):
    df = load()
    if not len(df):
        return False
    if 'mult' not in df.columns:
        df['mult'] = 0
    return bool(((df.K == K) & (df['mode'] == mode) & (df.temp == temp) & (df.mult.fillna(0) == mult)).any())

def run(K, mode, temp, mult=0):
    lab = 1 if mode == 'labels' else 0
    extra = '' if (K == 0 and mult == 0) else (f'--cell_k_mult {mult} --cell_k_labels_only {lab}' if mult > 0 else f'--cell_k {K} --cell_k_labels_only {lab}')
    cmd = f"python main.py {BASE} --teacher_temp {temp} {extra} --repeat {REPEAT} --down_grid '{DOWN}'"
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    rows = PAT_D.findall(out); c = PAT_C.search(out)
    if not rows:
        print('FAIL', K, mode, temp, '\n', out[-2000:]); return
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(K=K, mult=mult, mode=mode, temp=temp, drop=float(do_), wd=float(wd_), repeat=REPEAT,
                                    test=float(te), std=float(sd), val=float(va),
                                    cover=float(c[2]) if c else None, multip=float(c[3]) if c else None,
                                    flipped=float(c[4]) if c else None, h0=float(c[5]) if c else None, h1=float(c[6]) if c else None)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"cora 5.2% K={K:<3d} x{mult:<2g} {mode:6s} T={temp:<4g}" + (f"  covered {c[2]}% mult {c[3]}  label flipped {c[4]}%  H {c[5]}->{c[6]}" if c else '')
          + f"  val {best[5]} (do={best[0]} wd={best[1]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

print('cellk1: overlapping cells on cora 5.2%')

# ============ Cell 2: run =============
for temp in TEMPS:
    if not done(0, 'voronoi', temp):
        run(0, 'voronoi', temp)
for K, mode, temp in itertools.product(KS, MODES, TEMPS):
    if not done(K, mode, temp):
        run(K, mode, temp)
for mult, mode, temp in itertools.product(MULTS, MODES, TEMPS):
    if not done(0, mode, temp, mult):
        run(0, mode, temp, mult)

# ============ Cell 3: table =============
df = load(); df['mult'] = df['mult'].fillna(0) if 'mult' in df.columns else 0
b = df.sort_values('val', ascending=False).groupby(['K', 'mult', 'mode', 'temp']).head(1)
b = b.assign(var=b.apply(lambda x: 'voronoi' if (x.K == 0 and x.mult == 0) else (f"x{x.mult:g} {x['mode']}" if x.mult > 0 else f"K{int(x.K)} {x['mode']}"), axis=1),
             cell=b.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1))
order = ['voronoi'] + [f"K{K} {m}" for K in KS for m in MODES] + [f"x{r:g} {m}" for r in MULTS for m in MODES]
print('##### student test per T x variant   [val-selected dropout/wd, repeat 5]   (1.3% reference 84.4, oracle at 5.2% 86.2)')
print(b.pivot_table(index='temp', columns='var', values='cell', aggfunc='first').reindex(columns=order).to_string())
bb = b.sort_values('val', ascending=False).groupby('var').head(1)
print('\n##### T selected on val')
print(bb.assign(cell=bb.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f} (T{x['temp']:g})", axis=1)).set_index('var').reindex(order).cell.to_string())
print('\n##### coverage / multiplicity / label argmax flipped / H   [T 1]')
print(b[b.temp == 1.0].set_index('var').reindex(order)[['cover', 'multip', 'flipped', 'h0', 'h1']].round(3).to_string())
