# ============ Cell 1: common (SESSION = 'A' cora 5.2% | 'B' citeseer 3.6%) =============
# Removing the nodes far from every labelled node from the pool (--pool_drop_far f: farthest f in feature distance;
# --pool_drop_hop k: hop distance >= k). The teacher is least accurate there (Q4 vs Q1: cora 76 vs 88, citeseer 61 vs 81),
# so at high density their cells carry noisy labels. The budget m stays, i.e. the cells are re-allocated to the near region;
# the student still predicts the far test nodes by extrapolation. Two localising diagnostics: --label_oracle with
# --label_oracle_sel far/near f = true compositions only for the far (near) f of cells -> where does the oracle gain sit?
# Grid: drop feat {0.1, 0.25, 0.5}, hop {>=3, >=2}, x T; oracle all / far 25% / near 25% / far 50% at the first T; repeat 5.
SESSION = 'A'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'drop1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c pool_drop_hop /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--refine_teacher kernel --conv_depth 2 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 "
        "--dropout 0.5 --weight_decay 5e-4")
CELL = {'A': ('cora',     0.052, 'relu1', 0.01, 0, 2.0, [1.0, 0.5]),
        'B': ('citeseer', 0.036, 'erf',   3.0,  1, 0.2, [0.5, 0.25])}[SESSION]
DROPS = [('feat', 0.1), ('feat', 0.25), ('feat', 0.5), ('hop', 3), ('hop', 2)]
ORACLES = [('all', 1.0), ('far', 0.25), ('near', 0.25), ('far', 0.5)]
DOWN = '0,0.1,0.3,0.5,0.7,0.9;5e-4,5e-3'
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+) .*argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)')
PAT_P = re.compile(r'pool drop: .*? -> (\d+) nodes removed, pool (\d+)/(\d+)')
PAT_O = re.compile(r'label_oracle\[\w+ [\d.]+%\]: true compositions for (\d+)/(\d+) cells holding ([\d.]+)% of the pool')
PAT_C = re.compile(r'cell diag: within-var ([\d.]+) \(([\d.]+)% of total\)  cells (\d+)  size min/p10/med/p90/max (\d+)/(\d+)/(\d+)/(\d+)/(\d+)')
PAT_E = re.compile(r'expert: train ([\d.]+)%(?:\s+val ([\d.]+)%)?(?:\s+test ([\d.]+))?')

def load():
    return pd.DataFrame([json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)])

def done(ds, r, mode, val, temp, osel='none', ofrac=0.0):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df['mode'] == mode) & (df.val_ == val) & (df.temp == temp)
                            & (df.osel == osel) & (df.ofrac == ofrac)).any()

def run(ds, r, kernel, gamma, fn, mu, temp, mode, val, osel='none', ofrac=0.0):
    extra = (f' --pool_drop_far {val}' if mode == 'feat' else f' --pool_drop_hop {int(val)}' if mode == 'hop' else '')
    if osel != 'none':
        extra += f' --label_oracle 1 --label_oracle_sel {osel} --label_oracle_frac {ofrac}'
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --gamma {gamma} --feat_norm {fn} "
           f"--bregman {mu} --teacher_temp {temp} --repeat {REPEAT} --down_grid '{DOWN}'{extra}")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    rows = PAT_D.findall(out); g = PAT_G.search(out); pdr = PAT_P.search(out); o = PAT_O.search(out); c = PAT_C.search(out); e = PAT_E.search(out)
    if not rows or not g or not c:
        print('FAIL', ds, r, mode, val, temp, osel, '\n', out[-2500:]); return
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, mode=mode, val_=val, temp=temp, osel=osel, ofrac=ofrac, drop=float(do_), wd=float(wd_),
                                    repeat=REPEAT, test=float(te), std=float(sd), val=float(va),
                                    kl_g=float(g[1]), agree=float(g[2]), h_true=float(g[3]), h_q=float(g[4]),
                                    removed=int(pdr[1]) if pdr else 0, pool=int(pdr[2]) if pdr else None,
                                    o_cells=int(o[1]) if o else None, o_pool=float(o[3]) if o else None,
                                    cells=int(c[3]), within=float(c[2]), size_med=int(c[6]), size_max=int(c[8]),
                                    t_test=float(e[3]) if e and e[3] else None)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    tag = f'{mode}={val:<4g}' if mode != 'none' else 'none      '
    if osel != 'none':
        tag += f' oracle[{osel} {ofrac:g}]'
    print(f"{ds:8s} {r:<6g} {tag} T={temp:<4g}" + (f"  removed {pdr[1]} (pool {pdr[2]})" if pdr else '') + (f"  oracle {o[1]}/{o[2]} cells ({o[3]}% pool)" if o else '') +
          f"  cells {c[3]}  size {c[4]}/{c[6]}/{c[8]}  within-var {c[2]}%  agree {g[2]}%  KL {g[1]}  teacher test {e[3] if e else '?'}"
          f"  val {best[5]} (do={best[0]} wd={best[1]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

ds, r, kernel, gamma, fn, mu, temps = CELL
print(f'drop1 {SESSION}: {ds} {r:g}  drops {DROPS} x T {temps};  oracles {ORACLES} at T {temps[0]}')

# ============ Cell 2: run (done() resumes) =============
for temp in temps:
    if not done(ds, r, 'none', 0.0, temp):
        run(ds, r, kernel, gamma, fn, mu, temp, 'none', 0.0)
for (mode, val), temp in itertools.product(DROPS, temps):
    if not done(ds, r, mode, val, temp):
        run(ds, r, kernel, gamma, fn, mu, temp, mode, val)
for osel, ofrac in ORACLES:
    if not done(ds, r, 'none', 0.0, temps[0], osel, ofrac):
        run(ds, r, kernel, gamma, fn, mu, temps[0], 'none', 0.0, osel, ofrac)

# ============ Cell 3: tables (run where both drop1_*.jsonl are present) =============
df = load()
df['key'] = df.apply(lambda x: ('none' if x['mode'] == 'none' else f"{x['mode']} {x['val_']:g}") + ('' if x.osel == 'none' else f" | oracle {x.osel} {x.ofrac:g}"), axis=1)
b = df.sort_values('val', ascending=False).groupby(['ds', 'key', 'temp']).head(1)
bb = b.sort_values('val', ascending=False).groupby(['ds', 'key']).head(1)          # T on val
print('##### student test per ds x setting   [T, dropout, wd on val; repeat 5]')
for d, g in bb.groupby('ds'):
    base = g[g.key == 'none'].iloc[0]
    print(f'--- {d}   (none: {base.test:.2f}+-{base["std"]:.2f}, val {base.val:.2f})')
    for _, x in g.sort_values('key').iterrows():
        print(f"  {x.key:28s} T{x.temp:<5g} pool {str(int(x.pool)) if pd.notna(x.pool) else 'all':>6s} (removed {int(x.removed):5d})  cells {x.cells:4d} size med/max {x.size_med}/{x.size_max}  "
              f"within {x.within:5.1f}%  agree {x.agree:5.1f}%  KL {x.kl_g:.3f}  val {x.val:.2f}  test {x.test:.2f}+-{x['std']:.2f}  ({x.test - base.test:+.2f})")
print('\n##### val-selected over the drop settings (oracle rows excluded) vs none')
for d, g in bb[bb.osel == 'none'].groupby('ds'):
    top = g.sort_values('val', ascending=False).iloc[0]; base = g[g.key == 'none'].iloc[0]
    print(f"  {d}: best-val {top.key} T {top.temp:g}  test {top.test:.2f}+-{top['std']:.2f}   | none {base.test:.2f}   | oracle over grid {g.test.max():.2f}")
print('\n##### oracle localisation (T first): gain of true labels in all / far / near cells')
o = b[(b.osel != 'none')]
for d, g in o.groupby('ds'):
    base = b[(b.ds == d) & (b.key == 'none') & (b.temp == g.temp.iloc[0])].iloc[0]
    for _, x in g.sort_values('key').iterrows():
        print(f"  {d:8s} {x.key:24s} cells {x.o_cells}/{x.cells} ({x.o_pool:.1f}% of pool)  test {x.test:.2f}+-{x['std']:.2f}  gain {x.test - base.test:+.2f}")
