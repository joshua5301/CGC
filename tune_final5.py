# ============ Cell 1: common (SESSION = 'A' arxiv | 'B' reddit | 'C1' cora | 'C2' citeseer | 'C3' flickr) =============
# Settles the open axes in one overnight run, per cell, by coordinate search on val:
#   stage 1  large graphs: kernel {erf, relu1} x space {raw, nngp} x gamma {1e-4 .. 1e-1}            at mu 1,   T 1   (16)
#            small graphs: kernel {erf, relu1} x space {raw, nngp} x fn {0,1} x gamma {1e-2 .. 30}   at mu 0.3, T 1   (40)
#   stage 2  at the val-best of stage 1: the other mu values (large {0.3, 3}, small {1})
#   stage 3  at the val-best (kernel, space, [fn], gamma, mu): T {0.5, 0.25}; also T {0.5, 0.25} at the stage-1 mu
# Fixed: basis 3000, l1 k-medians, depth 2, probe_tol 1e-6 (LBFGS stops when converged, ce_steps 1000 is a cap),
#        wd {5e-4, 5e-3} x dropout {0,.1,.3,.5,.7,.9}, repeat 3. Stage 4 (morning): repeat 10 at the val-best + fixed recipe.
SESSION = 'A'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'final5'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c probe_tol /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --expert_basis 3000 --conv_depth 2 "
        "--no_hyperpara 1 --lr 0.01 --epoch 1000 --dropout 0.5 --weight_decay 5e-4")
RATIOS = {'cora': [0.013, 0.026, 0.052], 'citeseer': [0.009, 0.018, 0.036],
          'arxiv': [0.0005, 0.0025, 0.005], 'flickr': [0.001, 0.005, 0.01],
          'reddit': [0.0005, 0.001, 0.002]}
GROUPS = {'A': [('arxiv', r) for r in RATIOS['arxiv']], 'B': [('reddit', r) for r in RATIOS['reddit']],
          'C1': [('cora', r) for r in RATIOS['cora']], 'C2': [('citeseer', r) for r in RATIOS['citeseer']],
          'C3': [('flickr', r) for r in RATIOS['flickr']]}
CELLS = GROUPS[SESSION]
LARGE = lambda ds: ds in ('arxiv', 'reddit')
KERNELS = ['erf', 'relu1']
SPACES  = ['last', 'nngp']
GAMMAS  = lambda ds: [1e-4, 1e-3, 1e-2, 1e-1] if LARGE(ds) else [1e-2, 1e-1, 1.0, 10.0, 30.0]
FNS     = lambda ds: [0] if LARGE(ds) else [0, 1]
MU1     = lambda ds: 1.0 if LARGE(ds) else 0.3            # stage-1 mu
MU2     = lambda ds: [0.3, 3.0] if LARGE(ds) else [1.0]   # stage-2 alternatives
TEMPS   = [0.5, 0.25]
WDS, DOS = [5e-4, 5e-3], [0, 0.1, 0.3, 0.5, 0.7, 0.9]
DOWN1 = ','.join(f'{d:g}' for d in DOS) + ';' + ','.join(f'{w:g}' for w in WDS)
REP1, FIXED = 3, (0.5, 5e-4)
KEYS = ['kernel', 'space', 'fn', 'gamma', 'mu', 'temp']
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')

def load():
    recs = []
    for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl'):
        for l in open(f):
            d = json.loads(l)
            if all(k in d for k in KEYS):
                recs.append(d)
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
    kernel, space, fn, gamma, mu, temp = cfg
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --cluster_feat {space} "
           f"--feat_norm {fn} --gamma {gamma} --bregman {mu} --teacher_temp {temp} --repeat {repeat} --down_grid '{down}'")
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC')
    out = p.stdout + p.stderr
    rows = PAT_D.findall(out)
    if not rows:
        print('FAIL', ds, r, cfg, stage, '\n', out[-1500:]); return
    ex = re.search(r'expert: train [\d.]+%\s+(?:val ([\d.]+)%\s+)?test ([\d.]+)', out)
    en = re.search(r'posterior entropy ([\d.]+) -> ([\d.]+) nats', out)
    pb = re.findall(r'probe: lbfgs (\d+) iters', out)
    cd = re.search(r'cell diag.*', out); cd = cd[0] if cd else ''
    ct = re.search(r'Condensation time: ([0-9.]+)', out); ct = float(ct[1]) if ct else float('nan')
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, kernel=kernel, space=space, fn=fn, gamma=gamma, mu=mu, temp=temp,
                                    drop=float(do_), wd=float(wd_), repeat=repeat, stage=stage, down=down,
                                    test=float(te), std=float(sd), val=float(va),
                                    t_val=float(ex[1]) if ex and ex[1] else None, t_test=float(ex[2]) if ex else None,
                                    ent_after=float(en[2]) if en else None, lbfgs_iters=int(pb[-1]) if pb else None,
                                    diag=cd, cond_s=ct)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"{ds:8s} r={r:<7g} {kernel:5s} {space:4s} fn={fn} g={gamma:<5g} mu={mu:<3g} T={temp:<4g} [{stage}]  "
          f"teacher {ex[2] if ex else '?'}  val {best[5]} (do={best[0]} wd={best[1]}) test {best[3]}±{best[4]}  "
          f"({round(time.time() - t)}s, cond {ct:.0f}s)")

def pick(ds, r, fixed=False, **fix):
    s1 = load()
    if not len(s1):
        return None
    s1 = s1[(s1.stage == 'stage1') & (s1.ds == ds) & (s1.ratio == r)]
    for k, v in fix.items():
        s1 = s1[s1[k] == v]
    if fixed:
        s1 = s1[(s1['drop'] == FIXED[0]) & (s1.wd == FIXED[1])]
    return None if not len(s1) else s1.sort_values('val', ascending=False).iloc[0]

cfg_of = lambda b: (b.kernel, b.space, int(b.fn), float(b.gamma), float(b.mu), float(b.temp))
print(f'{SESSION}: cells {CELLS}')

# ============ Cell 2: stage 1-3 coordinate search (all logged as stage1; done() resumes) =============
for ds, r in CELLS:
    for kernel, space, fn, gamma in itertools.product(KERNELS, SPACES, FNS(ds), GAMMAS(ds)):        # stage 1
        cfg = (kernel, space, fn, gamma, MU1(ds), 1.0)
        if not done(ds, r, cfg, 'stage1'):
            run(ds, r, cfg, DOWN1, REP1, 'stage1')
    b = pick(ds, r, temp=1.0, mu=MU1(ds))
    k1, s1_, f1, g1 = b.kernel, b.space, int(b.fn), float(b.gamma)
    for mu in MU2(ds):                                                                              # stage 2
        cfg = (k1, s1_, f1, g1, mu, 1.0)
        if not done(ds, r, cfg, 'stage1'):
            run(ds, r, cfg, DOWN1, REP1, 'stage1')
    m_best = float(pick(ds, r, temp=1.0, kernel=k1, space=s1_, fn=f1, gamma=g1).mu)
    for mu in sorted({m_best, MU1(ds)}):                                                            # stage 3
        for T in TEMPS:
            cfg = (k1, s1_, f1, g1, mu, T)
            if not done(ds, r, cfg, 'stage1'):
                run(ds, r, cfg, DOWN1, REP1, 'stage1')
    b = pick(ds, r)
    print(f"--> {ds} {r:g}: {cfg_of(b)} do {b['drop']:g} wd {b.wd:g}  val {b.val:.2f} test {b.test:.2f}")

# ============ Cell 3 (morning): repeat 10 at the val-best config and at the fixed recipe =============
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
    if b is not None:                                   # ablation: no KL refinement (mu = 0) at the val-best config
        cfg0 = cfg_of(b)[:4] + (0.0, cfg_of(b)[5])
        down = f"{b['drop']:g};{b.wd:g}"
        if not done(ds, r, cfg0, 'mu0', down):
            run(ds, r, cfg0, down, 10, 'mu0')

# ============ Cell 4: tables (run where all final5_*.jsonl files are present) =============
df = load(); assert len(df), 'no logs found in LOGDIR'
s1 = df[df.stage == 'stage1']
print('##### stage 1: val-selected config per (ds, ratio)   [repeat 3]')
b1 = s1.sort_values('val', ascending=False).groupby(['ds', 'ratio']).head(1)
print(b1[['ds', 'ratio'] + KEYS + ['drop', 'wd', 'val', 'test', 'std', 't_test', 'ent_after', 'lbfgs_iters', 'cond_s']].sort_values(['ds', 'ratio']).to_string(index=False))
for ax in KEYS + ['drop', 'wd']:
    bx = s1.sort_values('val', ascending=False).groupby(['ds', 'ratio', ax]).head(1)
    print(f'\n##### stage 1: val-selected test per (ds, ratio) x {ax}')
    print(bx.pivot_table(index=['ds', 'ratio'], columns=ax, values='test').round(2).to_string())
print('\n##### teacher test accuracy per (ds, kernel, fn) x gamma')
print(s1.drop_duplicates(['ds', 'kernel', 'fn', 'gamma']).pivot_table(index=['ds', 'kernel', 'fn'], columns='gamma', values='t_test').round(1).to_string())
print('\n##### condensation time [s] (median) per ds x ratio x kernel x space')
print(s1.drop_duplicates(['ds', 'ratio'] + KEYS).pivot_table(index=['ds', 'ratio'], columns=['kernel', 'space'], values='cond_s', aggfunc='median').round(0).to_string())
top = lambda st: df[df.stage == st].sort_values('val', ascending=False).groupby(['ds', 'ratio']).head(1)
for stage, title in [('final', 'MAIN TABLE - Ours (val-selected), repeat 10'),
                     ('fixed', 'Ours (fixed GCond/ClustGDD recipe: dropout 0.5, wd 5e-4), repeat 10')]:
    s = top(stage)
    if not len(s):
        continue
    s = s.assign(cell=s.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1),
                 cfg=s.apply(lambda x: f"{x['kernel']} {x['space']} fn={x['fn']} g={x['gamma']:g} mu={x['mu']:g} T={x['temp']:g} do={x['drop']:g} wd={x['wd']:g}", axis=1))
    print(f'\n##### {title}')
    print(s.pivot_table(index='ds', columns='ratio', values='cell', aggfunc='first').to_string())
    print(s.pivot_table(index='ds', columns='ratio', values='cfg', aggfunc='first').to_string())
m0 = top('mu0')
if len(m0):
    m0 = m0.assign(cell=m0.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1))
    print('\n##### ablation: mu = 0 (no KL refinement) at the val-best config, repeat 10')
    print(m0.pivot_table(index='ds', columns='ratio', values='cell', aggfunc='first').to_string())
d0 = s1[(s1.kernel == 'erf') & (s1.space == 'last') & (s1.fn == 0) & (s1.temp == 1.0)]
d0 = d0[((d0.ds.isin(['arxiv', 'reddit'])) & (d0.gamma == 1e-3) & (d0.mu == 1.0)) | ((~d0.ds.isin(['arxiv', 'reddit'])) & (d0.gamma == 1.0) & (d0.mu == 0.3))]
if len(d0):
    d0 = d0.sort_values('val', ascending=False).groupby(['ds', 'ratio']).head(1)
    d0 = d0.assign(cell=d0.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f} (do={x['drop']:g} wd={x['wd']:g})", axis=1))
    print('\n##### default condensation config (erf, raw, fn 0, T 1; large: gamma 1e-3 mu 1, small: gamma 1 mu 0.3), recipe val-selected, stage-1 repeats')
    print(d0.pivot_table(index='ds', columns='ratio', values='cell', aggfunc='first').to_string())
fin, fix = top('final'), top('fixed')
if len(fin) and len(fix):
    d = (fin.set_index(['ds', 'ratio']).test - fix.set_index(['ds', 'ratio']).test).round(2)
    print('\n##### val-selected minus fixed recipe (test)'); print(d.unstack('ratio').to_string())
