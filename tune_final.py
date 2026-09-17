# ============ Cell 1: common (identical in all nine sessions except SESSION) =============
# FINAL main-table run, nine sessions (3 accounts x 3), one condensation cell (or one small dataset) per session:
#   A1 A2 A3 = arxiv 0.05 / 0.25 / 0.5%     B1 B2 B3 = reddit 0.05 / 0.1 / 0.2%
#   C1 = cora (3 densities)   C2 = citeseer (3 densities)   C3 = flickr (3 densities)
# Each session is self-contained: stage 1 (full grid) -> stage 2 (repeat 10) need only this session's log.
# Cell 4 (tables) merges every final3_*.jsonl it finds in LOGDIR: to build the 15-cell table, put the nine files
# from the three accounts into one Drive folder (download/upload or a shared-folder shortcut) and run Cell 4 there.
#
# Method (fixed): H = A^2 X, relu1 kernel teacher (rkhs prior, gamma), distance-sum k-medians + mu*KL refinement
# (--cluster_obj l1: geometric-median centres), x' = cell median, y' = cell mean posterior, A' = I.
# Stage 1 grid per cell (36 condensations; 72 on cora/citeseer/flickr): clustering space {raw, nngp} x teacher basis
#   {1000, 3000} x mu {0.3, 1, 3} x gamma {1e-4, 1e-3, 1e-2} x feature normalisation {0, 1} (small graphs only:
#   row-normalise cora/citeseer, StandardScaler flickr, as GCond/GEOM/ClustGDD); each run evaluates
#   wd {5e-4, 5e-3} x dropout {0, .1, .3, .5, .8}
#   with repeat 3 -> pick everything by val.
# Stage 2: val-best config at repeat 10 (main table) + fixed recipe (dropout 0.5, wd 5e-4) at its own val-best
#   condensation config, repeat 10 ("Ours (fixed recipe)").
SESSION = 'A1'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'final3'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c "l1_assign" /content/CGC/scr/label_solve.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --label_kernel relu1 --cluster_obj l1 "
        "--no_hyperpara 1 --lr 0.01 --epoch 1000 --dropout 0.5 --weight_decay 5e-4")
RATIOS = {'cora': [0.013, 0.026, 0.052], 'citeseer': [0.009, 0.018, 0.036],
          'arxiv': [0.0005, 0.0025, 0.005], 'flickr': [0.001, 0.005, 0.01],
          'reddit': [0.0005, 0.001, 0.002]}
GROUPS = {}
for i, r in enumerate(RATIOS['arxiv'], 1): GROUPS[f'A{i}'] = [('arxiv', r)]
for i, r in enumerate(RATIOS['reddit'], 1): GROUPS[f'B{i}'] = [('reddit', r)]
for i, d in enumerate(['cora', 'citeseer', 'flickr'], 1): GROUPS[f'C{i}'] = [(d, r) for r in RATIOS[d]]
GROUPS['A'] = GROUPS['A1'] + GROUPS['A2'] + GROUPS['A3']; GROUPS['B'] = GROUPS['B1'] + GROUPS['B2'] + GROUPS['B3']
GROUPS['C'] = GROUPS['C1'] + GROUPS['C2'] + GROUPS['C3']
CELLS = GROUPS[SESSION]
SPACES = ['last', 'nngp']            # clustering space: raw features / Nystrom features of the relu1 kernel
BASES  = [1000, 3000]                # teacher inducing points (min(N, basis) on cora/citeseer)
MUS    = [0.3, 1.0, 3.0]
GAMMAS = [1e-4, 1e-3, 1e-2]
FNS    = {'cora': [0, 1], 'citeseer': [0, 1], 'flickr': [0, 1]}   # feature normalisation axis; others: 0 (not implemented)
WDS    = [5e-4, 5e-3]
DOS    = [0, 0.1, 0.3, 0.5, 0.8]
DOWN1  = ','.join(f'{d:g}' for d in DOS) + ';' + ','.join(f'{w:g}' for w in WDS)
REP1   = 3
FIXED  = (0.5, 5e-4)                 # GCond / ClustGDD downstream recipe (dropout, wd)
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+): ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
KEYS = ['space', 'basis', 'gamma', 'mu', 'fn']

def load():
    recs = [json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)]
    return pd.DataFrame(recs)

def done(ds, r, cfg, stage, down=None):
    df = load()
    if not len(df):
        return False
    m = (df.ds == ds) & (df.ratio == r) & (df.stage == stage)
    for k, v in zip(KEYS, cfg):
        m &= (df[k] == v)
    if down is not None:
        m &= (df.down == down)
    return m.any()

def run(ds, r, cfg, down, repeat, stage):
    space, basis, gamma, mu, fn = cfg
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --cluster_feat {space} --expert_basis {basis} "
           f"--gamma {gamma} --bregman {mu} --feat_norm {fn} --repeat {repeat} --down_grid '{down}'")
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC')
    out = p.stdout + p.stderr
    rows = PAT_D.findall(out)
    if not rows:
        print('FAIL', ds, r, cfg, stage, '\n', out[-1500:]); return
    ex = re.search(r'expert:.*', out); ex = ex[0] if ex else ''
    cd = re.search(r'cell diag.*', out); cd = cd[0] if cd else ''
    with open(LOG, 'a') as f:
        for do_, wd_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, space=space, basis=basis, gamma=gamma, mu=mu, fn=fn,
                                    drop=float(do_), wd=float(wd_), repeat=repeat, stage=stage, down=down,
                                    test=float(te), std=float(sd), val=float(va), expert=ex, diag=cd)) + '\n')
    best = max(rows, key=lambda x: float(x[4]))
    print(f"{ds:8s} r={r:<7g} {space:4s} b={basis:<4d} g={gamma:<5g} mu={mu:<3g} fn={fn} [{stage}]  val {best[4]} "
          f"(do={best[0]} wd={best[1]}) test {best[2]}±{best[3]}  ({round(time.time() - t)}s)  {ex[:45]}")

def pick(ds, r, fixed=False):
    """val-best stage-1 row of a cell; fixed=True restricts to the GCond recipe."""
    s1 = load()
    if not len(s1):
        return None
    s1 = s1[(s1.stage == 'stage1') & (s1.ds == ds) & (s1.ratio == r)]
    if fixed:
        s1 = s1[(s1['drop'] == FIXED[0]) & (s1.wd == FIXED[1])]
    return None if not len(s1) else s1.sort_values('val', ascending=False).iloc[0]

cfg_of = lambda b: (b.space, int(b.basis), float(b.gamma), float(b.mu), int(b.fn))
grid_of = lambda ds: list(itertools.product(SPACES, BASES, GAMMAS, MUS, FNS.get(ds, [0])))
print(f'{SESSION}: cells {CELLS}, ' + ', '.join(f'{ds} {len(grid_of(ds))}' for ds in dict.fromkeys(d for d, _ in CELLS))
      + f' condensations per cell, {len(DOS) * len(WDS)} recipes x repeat {REP1} each')

# ============ Cell 2: stage 1 - full grid (36 / 72 condensations per cell; done() skips logged ones) =============
for ds, r in CELLS:
    for cfg in grid_of(ds):
        if not done(ds, r, cfg, 'stage1'):
            run(ds, r, cfg, DOWN1, REP1, 'stage1')
    b = pick(ds, r)
    print(f"--> {ds} {r:g}: {cfg_of(b)} do {b['drop']:g} wd {b.wd:g}  val {b.val:.2f} test {b.test:.2f}")

# ============ Cell 3: stage 2 - repeat 10 at the val-best config and at the fixed recipe =============
for ds, r in CELLS:
    b = pick(ds, r)
    if b is not None:
        down = f"{b['drop']:g};{b.wd:g}"
        if not done(ds, r, cfg_of(b), 'final', down):
            run(ds, r, cfg_of(b), down, 10, 'final')
    f = pick(ds, r, fixed=True)
    if f is not None:
        down = f"{FIXED[0]:g};{FIXED[1]:g}"
        if not done(ds, r, cfg_of(f), 'fixed', down):
            run(ds, r, cfg_of(f), down, 10, 'fixed')

# ============ Cell 4: tables (run where all nine final3_*.jsonl files are present) =============
df = load(); assert len(df), 'no logs found in LOGDIR'
s1 = df[df.stage == 'stage1']
print('##### stage 1: val-selected config per (ds, ratio)   [repeat 3]')
b1 = s1.sort_values('val', ascending=False).groupby(['ds', 'ratio']).head(1)
print(b1[['ds', 'ratio'] + KEYS + ['drop', 'wd', 'val', 'test', 'std']].sort_values(['ds', 'ratio']).to_string(index=False))
for ax in KEYS + ['drop', 'wd']:
    bx = s1.sort_values('val', ascending=False).groupby(['ds', 'ratio', ax]).head(1)
    print(f'\n##### stage 1: val-selected test per (ds, ratio) x {ax}')
    print(bx.pivot_table(index=['ds', 'ratio'], columns=ax, values='test').round(2).to_string())
top = lambda st: df[df.stage == st].sort_values('val', ascending=False).groupby(['ds', 'ratio']).head(1)
for stage, title in [('final', 'MAIN TABLE - Ours (val-selected), repeat 10'),
                     ('fixed', 'Ours (fixed GCond/ClustGDD recipe: dropout 0.5, wd 5e-4), repeat 10')]:
    s = top(stage)
    if not len(s):
        continue
    s = s.assign(cell=s.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1),
                 cfg=s.apply(lambda x: f"{x['space']} b={x['basis']} g={x['gamma']:g} mu={x['mu']:g} fn={x['fn']} do={x['drop']:g} wd={x['wd']:g}", axis=1))
    print(f'\n##### {title}')
    print(s.pivot_table(index='ds', columns='ratio', values='cell', aggfunc='first').to_string())
    print(s.pivot_table(index='ds', columns='ratio', values='cfg', aggfunc='first').to_string())
fin, fix = top('final'), top('fixed')
if len(fin) and len(fix):
    d = (fin.set_index(['ds', 'ratio']).test - fix.set_index(['ds', 'ratio']).test).round(2)
    print('\n##### val-selected minus fixed recipe (test)'); print(d.unstack('ratio').to_string())
# default-config row: raw space, basis 3000, gamma 1e-3, mu 1 (only dropout / wd selected on val), stage-1 repeats
d0 = s1[(s1.space == 'last') & (s1.basis == 3000) & (s1.gamma == 1e-3) & (s1.mu == 1.0) & (s1.fn == 0)]
if len(d0):
    d0 = d0.sort_values('val', ascending=False).groupby(['ds', 'ratio']).head(1)
    d0 = d0.assign(cell=d0.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f} (do={x['drop']:g} wd={x['wd']:g})", axis=1))
    print('\n##### Ours, default condensation config (raw space, basis 3000, gamma 1e-3, mu 1, no feat norm), recipe val-selected')
    print(d0.pivot_table(index='ds', columns='ratio', values='cell', aggfunc='first').to_string())
    if len(fin):
        print('\n##### val-selected condensation minus default (test)')
        print((fin.set_index(['ds', 'ratio']).test - d0.set_index(['ds', 'ratio']).test).round(2).unstack('ratio').to_string())
