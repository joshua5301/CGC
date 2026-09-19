# ============ Cell 1: common (SESSION = 'A' cora 5.2% | 'B' citeseer 3.6% gamma 3 | 'C' citeseer 3.6% gamma 30) =============
SESSION = 'A'
# Label variance reduction for small cells (--label_knn m): the condensed label of cell j = size-weighted mean of the
# cell-mean posteriors of its m nearest cells (centre distance), features unchanged. Motivation: the oracle-label gain on
# cora grows with density (+0.6 / +1.5 / +2.5 at 1.3 / 2.6 / 5.2%) while cell agreement drops (96.7 -> 90.8%), i.e. cells of
# ~9 nodes no longer average the teacher's errors away. Tested at the two weakest high-density cells, paired with m = 1,
# plus the oracle ceiling; m x T x dropout/wd on val, repeat 5.
import subprocess, re, json, os, time, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'lknn1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c label_knn_w /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--conv_depth 2 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 --dropout 0.5 --weight_decay 5e-4")
#         ds         ratio  kernel   gamma fn  mu    temps              (protocol-A val-best condensation; citeseer also gamma 30 = its oracle region)
CELLS = {'A': [('cora',     0.052, 'relu1', 0.01, 0, 2.0, [1.0, 0.5])],
         'B': [('citeseer', 0.036, 'erf',   3.0,  1, 0.2, [1.0, 0.5, 0.25])],
         'C': [('citeseer', 0.036, 'erf',   30.0, 1, 2.0, [1.0, 0.5, 0.25])]}[SESSION]
MS = [1, 2, 4, 8]
DOWN = '0,0.1,0.3,0.5,0.7,0.9;5e-4,5e-3'
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+) .*argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)')
PAT_K = re.compile(r'label_knn: m=(\d+) .*argmax changed ([\d.]+)% of cells  H ([\d.]+) -> ([\d.]+)')

def load():
    import glob
    recs = [json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)]
    return pd.DataFrame(recs)

def done(**key):
    df = load()
    if not len(df):
        return False
    m = pd.Series(True, index=df.index)
    for k, v in key.items():
        m &= (df[k] == v) if k in df.columns else False
    return bool(m.any())

def run(ds, r, kernel, gamma, fn, mu, temp, m, oracle=0, w='size'):
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --gamma {gamma} --feat_norm {fn} "
           f"--bregman {mu} --teacher_temp {temp} --label_knn {m} --label_knn_w {w} --label_oracle {oracle} "
           f"--repeat {REPEAT} --down_grid '{DOWN}'")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    rows = PAT_D.findall(out); g = PAT_G.search(out); kk = PAT_K.search(out)
    if not rows or not g:
        print('FAIL', ds, r, gamma, temp, m, oracle, '\n', out[-2000:]); return
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, kernel=kernel, gamma=gamma, fn=fn, mu=mu, temp=temp, m=m, w=w, oracle=oracle,
                                    drop=float(do_), wd=float(wd_), repeat=REPEAT, test=float(te), std=float(sd), val=float(va),
                                    kl_g=float(g[1]), agree=float(g[2]), h_true=float(g[3]), h_q=float(g[4]),
                                    flipped=float(kk[2]) if kk else 0.0)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"{ds:8s} {r:<6g} g={gamma:<4g} T={temp:<4g} m={m} {w:7s}{' ORACLE' if oracle else ''}  agree {g[2]}%  KL {g[1]}  H(q) {g[4]}"
          + (f"  flipped {kk[2]}%" if kk else '') + f"  val {best[5]} (do={best[0]} wd={best[1]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

print(f'lknn1 {SESSION}: label kNN smoothing  {CELLS}')

# ============ Cell 2: run =============
for ds, r, kernel, gamma, fn, mu, temps in CELLS:
    for m, temp in itertools.product(MS, temps):
        if not done(ds=ds, ratio=r, gamma=gamma, temp=temp, m=m, w='size', oracle=0):
            run(ds, r, kernel, gamma, fn, mu, temp, m)
    for temp in temps[:1]:                                   # uniform weights at m 4, and the oracle ceiling, at T 1
        if not done(ds=ds, ratio=r, gamma=gamma, temp=temp, m=4, w='uniform', oracle=0):
            run(ds, r, kernel, gamma, fn, mu, temp, 4, 0, 'uniform')
        if not done(ds=ds, ratio=r, gamma=gamma, temp=temp, m=1, oracle=1):
            run(ds, r, kernel, gamma, fn, mu, temp, 1, 1)

# ============ Cell 3: tables (run where all lknn1_*.jsonl are present) =============
import glob
df = load()
b = df.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'gamma', 'temp', 'm', 'w', 'oracle']).head(1)
b = b.assign(var=b.apply(lambda x: 'ORACLE' if x.oracle else f"m{int(x.m)}" + ('' if x.w == 'size' else ' unif'), axis=1),
             cell=b.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1))
print('##### student test per (ds, gamma, T) x m   [val-selected dropout/wd, repeat 5]')
print(b.pivot_table(index=['ds', 'gamma', 'temp'], columns='var', values='cell', aggfunc='first').to_string())
bb = b[b.oracle == 0].sort_values('val', ascending=False).groupby(['ds', 'gamma', 'var']).head(1)
print('\n##### T selected on val, per (ds, gamma) x m')
print(bb.assign(cell=bb.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f} (T{x['temp']:g})", axis=1))
        .pivot_table(index=['ds', 'gamma'], columns='var', values='cell', aggfunc='first').to_string())
print('\n##### val-selected over (gamma, T, m) per ds  vs  m = 1 at its own val-best T  vs  oracle')
for ds, g in b.groupby('ds'):
    top = g[g.oracle == 0].sort_values('val', ascending=False).iloc[0]
    base = g[(g.oracle == 0) & (g.m == 1) & (g.w == 'size')].sort_values('val', ascending=False).iloc[0]
    orc = g[g.oracle == 1].test.max()
    print(f"  {ds}: best-val g{top.gamma:g} T{top.temp:g} {top['var']}  test {top.test:.2f}+-{top['std']:.2f}   | m=1 best-val g{base.gamma:g} T{base.temp:g}: {base.test:.2f}   | oracle {orc:.2f}")
for mtr, title in [('agree', 'cell argmax agreement [%]'), ('kl_g', 'group KL(true||q)'), ('h_q', 'H(q)'), ('flipped', 'cells whose argmax the smoothing changed [%]')]:
    print(f'\n##### {title}   [T 1]')
    print(b[b.temp == 1.0].pivot_table(index=['ds', 'gamma'], columns='var', values=mtr, aggfunc='first').round(3).to_string())
