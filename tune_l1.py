# ============ Cell 1: common (identical in all three sessions except SESSION) =============
# Paired test of the clustering objective: l2 (squared distance + mu*KL, mean centres) vs
# l1 (distance sum + mu*KL = the Lipschitz bound, geometric-median centres). Euclidean space,
# relu1 teacher, basis 3000, feat_norm 1 on cora/citeseer/flickr, val-selected downstream recipe.
SESSION = 'C'          # 'A' / 'B' / 'C'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'l1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 220)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c l1_assign /content/CGC/scr/label_solve.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --expert_basis 3000 --label_kernel relu1 "
        "--no_hyperpara 1 --lr 0.01 --epoch 1000 --dropout 0.5 --weight_decay 5e-4")
CELLS = {'A': [('arxiv', 0.0005), ('arxiv', 0.0025)],
         'B': [('reddit', 0.0005), ('reddit', 0.001)],
         'C': [('citeseer', 0.009), ('citeseer', 0.018), ('citeseer', 0.036), ('cora', 0.026), ('flickr', 0.005)]}[SESSION]
FEATNORM = {'cora': 1, 'citeseer': 1, 'flickr': 1}
OBJS   = ['l2', 'l1']
GAMMAS = [1e-3, 1e-2]
MUS    = [0.0, 1.0]
DOWN   = '0,0.3,0.5,0.7;0,1e-4,5e-4,2e-3'
REPEAT = 2
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+): ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')

def load():
    recs = [json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)]
    return pd.DataFrame(recs)

def done(ds, r, obj, gamma, mu):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.obj == obj)
                            & (df.gamma == gamma) & (df.mu == mu)).any()

def run(ds, r, obj, gamma, mu):
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --feat_norm {FEATNORM.get(ds, 0)} "
           f"--cluster_obj {obj} --gamma {gamma} --bregman {mu} --repeat {REPEAT} --down_grid '{DOWN}'")
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC')
    out = p.stdout + p.stderr
    rows = PAT_D.findall(out)
    if not rows:
        print('FAIL', ds, r, obj, gamma, mu, '\n', out[-1500:]); return
    cd = re.search(r'cell diag.*', out); cd = cd[0] if cd else ''
    with open(LOG, 'a') as f:
        for do_, wd_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, obj=obj, gamma=gamma, mu=mu, drop=float(do_), wd=float(wd_),
                                    repeat=REPEAT, test=float(te), std=float(sd), val=float(va), diag=cd)) + '\n')
    best = max(rows, key=lambda x: float(x[4]))
    print(f"{ds:8s} r={r:<7g} {obj} g={gamma:<5g} mu={mu:g}  best val {best[4]} "
          f"(do={best[0]} wd={best[1]}) test {best[2]}±{best[3]}  ({round(time.time() - t)}s)  {cd[:110]}")

# ============ Cell 2: paired grid (8 condensations per cell) =============
for ds, r in CELLS:
    for gamma, mu, obj in itertools.product(GAMMAS, MUS, OBJS):
        if not done(ds, r, obj, gamma, mu):
            run(ds, r, obj, gamma, mu)

# ============ Cell 3: tables (any session) =============
df = load(); assert len(df), 'no logs found in Drive'
best = df.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'obj', 'mu']).head(1)
best = best.assign(cell=best.apply(lambda x: f"{x['test']:.1f}±{x['std']:.1f} (γ={x['gamma']:g})", axis=1))
print('##### val-selected test per (dataset, ratio, mu): columns = objective')
print(best.pivot_table(index=['ds', 'ratio', 'mu'], columns='obj', values='cell', aggfunc='first').to_string())
w = best.pivot_table(index=['ds', 'ratio', 'mu'], columns='obj', values='test')
if set(OBJS) <= set(w.columns):
    w['delta'] = (w['l1'] - w['l2']).round(2)
    print('\n##### delta = l1 - l2 (test acc, val-selected gamma/recipe)')
    print(w.round(2).to_string())
    print(f"\nmean delta: mu=0 {w.xs(0.0, level='mu')['delta'].mean():+.2f}   mu=1 {w.xs(1.0, level='mu')['delta'].mean():+.2f}")
sel = df.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'obj']).head(1)
print('\n##### fully val-selected (gamma, mu, recipe) per objective')
print(sel.pivot_table(index=['ds', 'ratio'], columns='obj', values='test').round(2).to_string())
print('\n##### cell diag (within-var, size quantiles, cells<10) per condensation')
d = df.drop_duplicates(['ds', 'ratio', 'obj', 'gamma', 'mu'])
d = d.assign(diag=d.diag.str.replace('cell diag: ', '').str[:120])
print(d.pivot_table(index=['ds', 'ratio', 'gamma', 'mu'], columns='obj', values='diag', aggfunc='first').to_string())
