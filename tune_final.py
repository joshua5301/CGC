# ============ Cell 1: common (identical in all sessions except SESSION) =============
# FINAL main-table run, two stages.
#   stage 1: gamma x mu grid, 6 dropouts (wd fixed 5e-4; repeat 1 on arxiv/reddit, 2 on the small graphs)
#            -> pick (gamma, mu, dropout, wd) by val
#   stage 2: the val-best config at repeat 10 (single recipe)   -> main-table cell
#            + the fixed GCond/ClustGDD recipe (dropout 0.5, wd 5e-4, all datasets) at its own val-best (gamma, mu),
#              repeat 10                                          -> "Ours (fixed recipe)" row
# Method: H = A^2 X, relu1 kernel teacher (basis 3000, rkhs, gamma), Euclidean k-means + mu*KL refinement,
# x' = cell mean, y' = cell mean posterior, A' = I. No feature preprocessing (see FEATNORM).
# Sessions: 'A'/'B'/'C' = whole dataset group, or per cell 'A1'..'A3' (arxiv densities), 'B1'..'B3' (reddit),
#           'C1' (cora), 'C2' (citeseer), 'C3' (flickr). Logs merge on Drive; every cell resumes via done().
SESSION = 'C'
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
FEATNORM = {}                                          # raw features everywhere (cora_fn / citeseer_fn cells)
WD = 5e-4                                              # weight decay fixed (GCN/GCond default); only dropout is selected on val
GROUPS = {'A': [('arxiv', r) for r in RATIOS['arxiv']], 'B': [('reddit', r) for r in RATIOS['reddit']],
          'C': [(d, r) for d in ['cora', 'citeseer', 'flickr'] for r in RATIOS[d]]}
for i, r in enumerate(RATIOS['arxiv'], 1): GROUPS[f'A{i}'] = [('arxiv', r)]
for i, r in enumerate(RATIOS['reddit'], 1): GROUPS[f'B{i}'] = [('reddit', r)]
for i, d in enumerate(['cora', 'citeseer', 'flickr'], 1): GROUPS[f'C{i}'] = [(d, r) for r in RATIOS[d]]
CELLS = GROUPS[SESSION]
GAMMAS = [1e-4, 1e-3, 1e-2, 1e-1]
MUS    = [0.2, 0.5, 1.0, 2.0, 5.0]
DOWN1  = f'0,0.1,0.3,0.5,0.7,0.9;{WD:g}'
REP1   = lambda ds: 1 if ds in ('arxiv', 'reddit') else 2      # stage-1 repeats (val sets: 30k / 24k vs 500)
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
        s1 = s1[(s1['drop'] == 0.5) & (s1.wd == WD)]
    return None if not len(s1) else s1.sort_values('val', ascending=False).iloc[0]

# ============ Cell 2: stage 1 - gamma x mu grid x 6 dropouts (20 condensations per cell; done() skips logged ones) =============
for ds, r in CELLS:
    for gamma, mu in itertools.product(GAMMAS, MUS):
        if not done(ds, r, gamma, mu, 'stage1'):
            run(ds, r, gamma, mu, DOWN1, REP1(ds), 'stage1')

# ============ Cell 2b: selected config per cell (edge = at a grid boundary) =============
print('##### selected config per cell (edge = at a grid boundary)')
for ds, r in CELLS:
    b = pick(ds, r)
    if b is not None:
        edge = ' '.join(k for k, v in [('gamma', b.gamma in (min(GAMMAS), max(GAMMAS))), ('mu', b.mu in (min(MUS), max(MUS))),
                                       ('do', b['drop'] in (0.0, 0.9)), ('wd', False)] if v)
        print(f"{ds} {r:g}: gamma {b.gamma:g} mu {b.mu:g} do {b['drop']:g} wd {b.wd:g}  val {b.val:.2f} test {b.test:.2f}  edge: {edge or '-'}")

# ============ Cell 3: stage 2 - repeat 10 at the val-best config and at the fixed recipe =============
for ds, r in CELLS:
    b = pick(ds, r)
    if b is not None:
        down = f"{b['drop']:g};{b['wd']:g}"
        if not done(ds, r, b.gamma, b.mu, 'final', down):
            run(ds, r, b.gamma, b.mu, down, 10, 'final')
    f = pick(ds, r, fixed=True)
    if f is not None:
        down = f"0.5;{WD:g}"
        if not done(ds, r, f.gamma, f.mu, 'fixed', down):
            run(ds, r, f.gamma, f.mu, down, 10, 'fixed')

# ============ Cell 4: tables (any session) =============
df = load(); assert len(df), 'no logs found in Drive'
s1 = df[df.stage == 'stage1']
print('##### stage 1: val-selected config per (ds, ratio)')
b1 = s1.sort_values('val', ascending=False).groupby(['ds', 'ratio']).head(1)
print(b1[['ds', 'ratio', 'gamma', 'mu', 'drop', 'wd', 'val', 'test', 'std']].sort_values(['ds', 'ratio']).to_string(index=False))
print('\n##### stage 1: val-selected test per (ds, ratio) x mu')
bm = s1.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'mu']).head(1)
print(bm.pivot_table(index=['ds', 'ratio'], columns='mu', values='test').round(2).to_string())
print('\n##### stage 1: val-selected test per (ds, ratio) x gamma')
bg = s1.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'gamma']).head(1)
print(bg.pivot_table(index=['ds', 'ratio'], columns='gamma', values='test').round(2).to_string())
for stage, title in [('final', 'MAIN TABLE - Ours (val-selected recipe), repeat 10'),
                     ('fixed', 'Ours (fixed GCond/ClustGDD recipe: dropout 0.5, wd 5e-4), repeat 10')]:
    s = df[df.stage == stage]
    if not len(s):
        continue
    s = s.sort_values('val', ascending=False).groupby(['ds', 'ratio']).head(1)   # several repeat-10 runs per cell -> keep the val-best
    s = s.assign(cell=s.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1),
                 cfg=s.apply(lambda x: f"g={x['gamma']:g} mu={x['mu']:g} do={x['drop']:g} wd={x['wd']:g}", axis=1))
    print(f'\n##### {title}')
    print(s.pivot_table(index='ds', columns='ratio', values='cell', aggfunc='first').to_string())
    print(s.pivot_table(index='ds', columns='ratio', values='cfg', aggfunc='first').to_string())
top = lambda st: df[df.stage == st].sort_values('val', ascending=False).groupby(['ds', 'ratio']).head(1)
fin, fix = top('final'), top('fixed')
if len(fin) and len(fix):
    d = (fin.set_index(['ds', 'ratio']).test - fix.set_index(['ds', 'ratio']).test).round(2)
    print('\n##### val-selected minus fixed recipe (test)'); print(d.unstack('ratio').to_string())
