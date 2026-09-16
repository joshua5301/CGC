# ============ Cell 1: common (identical in all three sessions except SESSION) =============
# FINAL main-table run, two stages.
#   stage 1: gamma x mu grid, repeat 2, 16 downstream recipes  -> pick (gamma, mu, dropout, wd) by val
#   stage 2: the val-best config at repeat 10 (single recipe)   -> main-table cell
#            + the fixed GCond/ClustGDD recipe (dropout 0.5, wd 5e-4; arxiv wd 0) at its own val-best (gamma, mu),
#              repeat 10                                          -> "Ours (fixed recipe)" row
# Method: H = A^2 X, relu1 kernel teacher (basis 3000, rkhs, gamma), Euclidean k-means + mu*KL refinement,
# x' = cell mean, y' = cell mean posterior, A' = I. Preprocessing as GCond/GEOM/ClustGDD (see FEATNORM).
SESSION = 'C'          # 'A' / 'B' / 'C'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'final'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 220)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c "pred.dbl" /content/CGC/scr/label_solve.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --expert_basis 3000 --label_kernel relu1 "
        "--no_hyperpara 1 --lr 0.01 --epoch 1000 --dropout 0.5 --weight_decay 5e-4")
RATIOS = {'cora': [0.013, 0.026, 0.052], 'citeseer': [0.009, 0.018, 0.036],
          'arxiv': [0.0005, 0.0025, 0.005], 'flickr': [0.001, 0.005, 0.01],
          'reddit': [0.0005, 0.001, 0.002]}
FEATNORM = {'cora': 1, 'citeseer': 1, 'flickr': 1}     # <- set cora to 0 if the cora_fn cell says so
FIXED_WD = {'arxiv': 0.0}                              # GCond recipe: wd 5e-4 everywhere except arxiv (0)
MINE = {'A': ['arxiv'], 'B': ['reddit'], 'C': ['cora', 'citeseer', 'flickr']}[SESSION]
GAMMAS = [1e-3, 1e-2, 3e-2]
MUS    = [0.0, 1.0]
DOWN1  = '0,0.3,0.5,0.7;0,1e-4,5e-4,2e-3'
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+): ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')

def load():
    recs = [json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)]
    return pd.DataFrame(recs)

def done(ds, r, gamma, mu, stage, down=None):
    df = load()
    if not len(df):
        return False
    m = (df.ds == ds) & (df.ratio == r) & (df.gamma == gamma) & (df.mu == mu) & (df.stage == stage)
    if down is not None:
        m &= (df.down == down)
    return m.any()

def run(ds, r, gamma, mu, down, repeat, stage):
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --feat_norm {FEATNORM.get(ds, 0)} "
           f"--gamma {gamma} --bregman {mu} --repeat {repeat} --down_grid '{down}'")
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC')
    out = p.stdout + p.stderr
    rows = PAT_D.findall(out)
    if not rows:
        print('FAIL', ds, r, gamma, mu, stage, '\n', out[-1500:]); return
    ex = re.search(r'expert:.*', out); ex = ex[0] if ex else ''
    cd = re.search(r'cell diag.*', out); cd = cd[0] if cd else ''
    with open(LOG, 'a') as f:
        for do_, wd_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, gamma=gamma, mu=mu, drop=float(do_), wd=float(wd_),
                                    repeat=repeat, stage=stage, down=down, test=float(te), std=float(sd),
                                    val=float(va), expert=ex, diag=cd, fn=FEATNORM.get(ds, 0))) + '\n')
    best = max(rows, key=lambda x: float(x[4]))
    print(f"{ds:8s} r={r:<7g} g={gamma:<5g} mu={mu:g} [{stage}]  val {best[4]} "
          f"(do={best[0]} wd={best[1]}) test {best[2]}±{best[3]}  ({round(time.time() - t)}s)  {ex[:45]}")

def pick(ds, r, fixed=False):
    """val-best stage-1 row for a cell; fixed=True restricts to the GCond recipe."""
    s1 = load(); s1 = s1[(s1.stage == 'stage1') & (s1.ds == ds) & (s1.ratio == r)]
    if fixed:
        s1 = s1[(s1['drop'] == 0.5) & (s1.wd == FIXED_WD.get(ds, 5e-4))]
    return None if not len(s1) else s1.sort_values('val', ascending=False).iloc[0]

# ============ Cell 2: stage 1 - gamma x mu grid, repeat 2, 16 recipes (6 condensations per cell) =============
for ds in MINE:
    for r in RATIOS[ds]:
        for gamma, mu in itertools.product(GAMMAS, MUS):
            if not done(ds, r, gamma, mu, 'stage1'):
                run(ds, r, gamma, mu, DOWN1, 2, 'stage1')

# ============ Cell 3: stage 2 - repeat 10 at the val-best config and at the fixed recipe =============
for ds in MINE:
    for r in RATIOS[ds]:
        b = pick(ds, r)
        if b is not None:
            down = f"{b['drop']:g};{b['wd']:g}"
            if not done(ds, r, b.gamma, b.mu, 'final', down):
                run(ds, r, b.gamma, b.mu, down, 10, 'final')
        f = pick(ds, r, fixed=True)
        if f is not None:
            down = f"0.5;{FIXED_WD.get(ds, 5e-4):g}"
            if not done(ds, r, f.gamma, f.mu, 'fixed', down):
                run(ds, r, f.gamma, f.mu, down, 10, 'fixed')

# ============ Cell 4: tables (any session) =============
df = load(); assert len(df), 'no logs found in Drive'
s1 = df[df.stage == 'stage1']
print('##### stage 1: val-selected test per (ds, ratio, mu)  [repeat 2]')
b1 = s1.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'mu']).head(1)
b1 = b1.assign(cell=b1.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f} (g={x['gamma']:g} do={x['drop']:g} wd={x['wd']:g})", axis=1))
print(b1.pivot_table(index=['ds', 'ratio'], columns='mu', values='cell', aggfunc='first').to_string())
for stage, title in [('final', 'MAIN TABLE - Ours (val-selected recipe), repeat 10'),
                     ('fixed', 'Ours (fixed GCond/ClustGDD recipe: dropout 0.5, wd 5e-4 / arxiv 0), repeat 10')]:
    s = df[df.stage == stage]
    if not len(s):
        continue
    s = s.assign(cell=s.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1),
                 cfg=s.apply(lambda x: f"g={x['gamma']:g} mu={x['mu']:g} do={x['drop']:g} wd={x['wd']:g}", axis=1))
    print(f'\n##### {title}')
    print(s.pivot_table(index='ds', columns='ratio', values='cell', aggfunc='first').to_string())
    print(s.pivot_table(index='ds', columns='ratio', values='cfg', aggfunc='first').to_string())
fin, fix = df[df.stage == 'final'], df[df.stage == 'fixed']
if len(fin) and len(fix):
    d = (fin.set_index(['ds', 'ratio']).test - fix.set_index(['ds', 'ratio']).test).round(2)
    print('\n##### val-selected minus fixed recipe (test)'); print(d.unstack('ratio').to_string())
