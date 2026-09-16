# ============ Cell 1: common (identical in all three sessions except SESSION) =============
# Teacher-consistent representative refinement (--refine_c lam): x_j moves from the cell mean to
# argmin CE(ybar_j, f(x)) + lam*||x - hbar_j||^2/s under the kernel teacher f. Paired against the
# plain cell mean (refine off). relu1 teacher, basis 3000, Euclidean k-means + KL (mu=1),
# feat_norm 1 on cora/citeseer/flickr, val-selected downstream recipe.
SESSION = 'C'          # 'A' / 'B' / 'C'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'refine'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 220)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c refine_centres /content/CGC/scr/label_solve.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --expert_basis 3000 --label_kernel relu1 "
        "--bregman 1 --no_hyperpara 1 --lr 0.01 --epoch 1000 --dropout 0.5 --weight_decay 5e-4")
CELLS = {'A': [('arxiv', 0.0005), ('arxiv', 0.0025)],
         'B': [('reddit', 0.0005), ('reddit', 0.001)],
         'C': [('citeseer', 0.009), ('citeseer', 0.018), ('cora', 0.013), ('cora', 0.026), ('flickr', 0.001)]}[SESSION]
FEATNORM = {'cora': 1, 'citeseer': 1, 'flickr': 1}
LAMS   = [-1, 0.3, 1.0, 3.0]          # -1 = off (plain cell mean)
GAMMAS = [1e-3, 1e-2]
DOWN   = '0,0.3,0.5,0.7;0,1e-4,5e-4,2e-3'
REPEAT = 2
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+): ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')

def load():
    recs = [json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)]
    return pd.DataFrame(recs)

def done(ds, r, lam, gamma):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.lam == lam) & (df.gamma == gamma)).any()

def run(ds, r, lam, gamma):
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --feat_norm {FEATNORM.get(ds, 0)} "
           f"--refine_c {lam} --gamma {gamma} --repeat {REPEAT} --down_grid '{DOWN}'")
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC')
    out = p.stdout + p.stderr
    rows = PAT_D.findall(out)
    if not rows:
        print('FAIL', ds, r, lam, gamma, '\n', out[-1500:]); return
    rf = re.search(r'refine_c:.*', out); rf = rf[0] if rf else ''
    cd = re.search(r'cell diag.*', out); cd = cd[0] if cd else ''
    with open(LOG, 'a') as f:
        for do_, wd_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, lam=lam, gamma=gamma, drop=float(do_), wd=float(wd_),
                                    repeat=REPEAT, test=float(te), std=float(sd), val=float(va),
                                    refine=rf, diag=cd)) + '\n')
    best = max(rows, key=lambda x: float(x[4]))
    print(f"{ds:8s} r={r:<7g} lam={lam:<4g} g={gamma:<5g}  best val {best[4]} "
          f"(do={best[0]} wd={best[1]}) test {best[2]}±{best[3]}  ({round(time.time() - t)}s)  {rf}")

# ============ Cell 2: paired grid (8 condensations per cell) =============
for ds, r in CELLS:
    for gamma, lam in itertools.product(GAMMAS, LAMS):
        if not done(ds, r, lam, gamma):
            run(ds, r, lam, gamma)

# ============ Cell 3: tables (any session) =============
df = load(); assert len(df), 'no logs found in Drive'
best = df.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'lam']).head(1)
best = best.assign(cell=best.apply(lambda x: f"{x['test']:.1f}±{x['std']:.1f} (γ={x['gamma']:g})", axis=1))
print('##### val-selected test per (dataset, ratio): columns = refine lam (-1 = off)')
print(best.pivot_table(index=['ds', 'ratio'], columns='lam', values='cell', aggfunc='first').to_string())
w = best.pivot_table(index=['ds', 'ratio'], columns='lam', values='test')
if -1 in w.columns:
    for lam in [c for c in w.columns if c != -1]:
        w[f'd{lam:g}'] = (w[lam] - w[-1]).round(2)
    print('\n##### delta vs off (test, val-selected gamma/recipe)')
    print(w.round(2).to_string())
    print('\nmean delta:', {f'lam={c[1:]}': round(w[c].mean(), 2) for c in w.columns if str(c).startswith('d')})
print('\n##### refine diagnostics: KL(ybar || f(c)) before -> after, mean shift (in within-cell radii)')
d = df.drop_duplicates(['ds', 'ratio', 'lam', 'gamma'])
print(d[d.lam >= 0].pivot_table(index=['ds', 'ratio', 'gamma'], columns='lam', values='refine',
                                aggfunc=lambda v: v.iloc[0].replace('refine_c: ', '')[:60]).to_string())
