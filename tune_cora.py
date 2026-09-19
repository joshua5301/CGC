# ============ Cell 1: common (cora sweep; three sessions: SESSION = 'C1' / 'C2' / 'C3' = 1.3 / 2.6 / 5.2%) =============
# Same protocol as the citeseer / arxiv sweeps with the cora-specific axes fixed:
#   fixed   raw space, basis 3000 (>= N: full kernel), depth 2, fn 0 (fn 1 was -4), lr 0.01, random basis
#   stage 1 (Cell 2)  kernel {erf, relu1} x gamma {0.01, 0.1, 1, 10} x T {2, 1, 0.5} x mu {0.2, 0.5, 1, 2, 5}   = 120 condensations
#                     each: dropout {0,.1,.3,.5,.7,.9} x wd {5e-4, 5e-3} on val, repeat 3, eval every 10 epochs (~1 min)
#                     (T 2 = flatter labels than the teacher mean, new; T 0.25 dropped: sharpening never helped cora)
#   stage 2 (Cell 3)  around the top-5 val configs: T 4 if T 2 won there, depth {1, 3}, space nngp               <= 20 more
#   stage 3 (Cell 4)  repeat 10 at the top-3 val configs (own recipe), fixed recipe (do .5, wd 5e-4), condensation seeds {1, 2}
#   Cell 5            POOLED selection: one config per dataset by mean val over the three densities (3x the val evidence;
#                     cora / citeseer val = 500 nodes cannot rank 120 configs per density), repeat 10 at this density
#   Cell 6            tables: marginals, kernel x gamma, mu x T, selection noise, main rows, pooled rows
SESSION = 'C1'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'cora1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run("grep -c \"default='random', choices=\\['kmeans', 'random'\\]\" /content/CGC/scr/para.py", shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale (random basis default)'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --expert_basis 3000 --nngp_basis 3000 "
        "--feat_norm 0 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 "
        "--dropout 0.5 --weight_decay 5e-4 --dataset_name cora")
RATIOS  = [0.013, 0.026, 0.052]
RATIO   = {'C1': 0.013, 'C2': 0.026, 'C3': 0.052}[SESSION]
KERNELS = ['erf', 'relu1']
GAMMAS  = [0.01, 0.1, 1.0, 10.0]
TEMPS   = [2.0, 1.0, 0.5]
MUS     = [0.2, 0.5, 1.0, 2.0, 5.0]
DEPTHS  = [1, 3]
WDS     = [5e-4, 5e-3]
DOS     = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
DOWN1   = ','.join(f'{d:g}' for d in DOS) + ';' + ','.join(f'{w:g}' for w in WDS)
REP1, TOPK, FIXED = 3, 3, (0.5, 5e-4)
KEYS = ['kernel', 'gamma', 'temp', 'mu', 'space', 'depth', 'seed']
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')

def load():
    recs = []
    for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl'):
        for l in open(f):
            d = json.loads(l)
            if all(k in d for k in KEYS):
                recs.append(d)
    return pd.DataFrame(recs)

def done(cfg, stage, down=None, ratio=None):
    df = load()
    if not len(df):
        return False
    m = (df.ratio == (RATIO if ratio is None else ratio)) & (df.stage == stage)
    for k, v in zip(KEYS, cfg):
        m &= (df[k] == v)
    if down is not None:
        m &= (df.down == down)
    return m.any()

def run(cfg, down, repeat, stage):
    kernel, gamma, temp, mu, space, depth, seed = cfg
    cmd = (f"python main.py {BASE} --ratio {RATIO} --label_kernel {kernel} --gamma {gamma} --teacher_temp {temp} "
           f"--bregman {mu} --cluster_feat {space} --conv_depth {depth} --seed {seed} --repeat {repeat} --down_grid '{down}'")
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC')
    out = p.stdout + p.stderr
    rows = PAT_D.findall(out)
    if not rows:
        print('FAIL', cfg, stage, '\n', out[-1500:]); return
    ex = re.search(r'expert: train ([\d.]+)%\s+val ([\d.]+)%\s+test ([\d.]+)', out)
    en = re.search(r'posterior entropy ([\d.]+) -> ([\d.]+) nats', out)
    gd = re.search(r'group diag: KL\(true\|\|q\) ([\d.]+) .*argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)', out)
    cd = re.search(r'cell diag.*', out); cd = cd[0] if cd else ''
    ct = re.search(r'Condensation time: ([0-9.]+)', out); ct = float(ct[1]) if ct else float('nan')
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds='cora', ratio=RATIO, kernel=kernel, gamma=gamma, temp=temp, mu=mu, space=space, depth=depth, seed=seed,
                                    drop=float(do_), wd=float(wd_), repeat=repeat, stage=stage, down=down,
                                    test=float(te), std=float(sd), val=float(va),
                                    t_val=float(ex[2]) if ex else None, t_test=float(ex[3]) if ex else None,
                                    ent_after=float(en[2]) if en else None,
                                    kl_g=float(gd[1]) if gd else None, agree=float(gd[2]) if gd else None,
                                    h_q=float(gd[4]) if gd else None, diag=cd, cond_s=ct)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"cora {RATIO:g} {kernel:5s} g={gamma:<4g} T={temp:<3g} mu={mu:<3g} {space:4s} d={depth} s={seed} [{stage}]  "
          f"teacher {ex[3] if ex else '?'}  agree {gd[2] if gd else '?'}%  val {best[5]} (do={best[0]} wd={best[1]}) "
          f"test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

def top(k=1, fixed=False, stage='stage1', ratio=None):
    s1 = load()
    if not len(s1):
        return s1
    s1 = s1[(s1.stage == stage) & (s1.ratio == (RATIO if ratio is None else ratio))]
    if fixed:
        s1 = s1[(s1['drop'] == FIXED[0]) & (s1.wd == FIXED[1])]
    s1 = s1.sort_values('val', ascending=False).drop_duplicates(KEYS)
    return s1.head(k)

def pooled(k=5):
    """configs present at all three densities, ranked by mean val over densities (each at its own val-best recipe)"""
    s1 = load()
    s1 = s1[s1.stage == 'stage1'].sort_values('val', ascending=False).drop_duplicates(['ratio'] + KEYS)
    g = s1.groupby(KEYS).agg(n=('ratio', 'nunique'), val=('val', 'mean'), test=('test', 'mean'))
    g = g[g.n == len(RATIOS)].sort_values('val', ascending=False)
    return g.head(k).reset_index()

cfg_of = lambda b: (b.kernel, float(b.gamma), float(b.temp), float(b.mu), b.space, int(b.depth), int(b.seed))
GRID = [(k, g, T, mu, 'last', 2, 0) for k, g, T, mu in itertools.product(KERNELS, GAMMAS, TEMPS, MUS)]
print(f'{SESSION}: cora {RATIO:g}, stage 1 = {len(GRID)} condensations x {len(DOS) * len(WDS)} recipes x repeat {REP1}')

# ============ Cell 2: stage 1 - full factorial (done() skips logged configs) =============
for cfg in GRID:
    if not done(cfg, 'stage1'):
        run(cfg, DOWN1, REP1, 'stage1')
b = top(1).iloc[0]
print(f"--> cora {RATIO:g}: {cfg_of(b)} do {b['drop']:g} wd {b.wd:g}  val {b.val:.2f} test {b.test:.2f}")

# ============ Cell 3: stage 2 - neighbourhood of the top-5: T 4 (if T 2), depth {1, 3}, space nngp (logged as stage1) =============
for _, b in top(5).iterrows():
    k_, g_, T_, mu_, sp_, d_, s_ = cfg_of(b)
    cfgs = [(k_, g_, T_, mu_, sp_, dp, 0) for dp in DEPTHS] + [(k_, g_, T_, mu_, 'nngp', d_, 0)]
    if T_ == 2.0:
        cfgs.append((k_, g_, 4.0, mu_, sp_, d_, 0))
    for cfg in cfgs:
        if not done(cfg, 'stage1'):
            run(cfg, DOWN1, REP1, 'stage1')
b = top(1).iloc[0]
print(f"--> cora {RATIO:g} after stage 2: {cfg_of(b)} do {b['drop']:g} wd {b.wd:g}  val {b.val:.2f} test {b.test:.2f}")

# ============ Cell 4: stage 3 - repeat 10 at the top-3 (own recipe), fixed recipe, condensation seeds =============
for _, b in top(TOPK).iterrows():
    down = f"{b['drop']:g};{b.wd:g}"
    if not done(cfg_of(b), 'final', down):
        run(cfg_of(b), down, 10, 'final')
f = top(1, fixed=True)
if len(f):
    f = f.iloc[0]; dfx = f"{FIXED[0]:g};{FIXED[1]:g}"
    if not done(cfg_of(f), 'fixed', dfx):
        run(cfg_of(f), dfx, 10, 'fixed')
b = top(1).iloc[0]; down = f"{b['drop']:g};{b.wd:g}"
for seed in (1, 2):
    cfg = cfg_of(b)[:-1] + (seed,)
    if not done(cfg, 'seed', down):
        run(cfg, down, 10, 'seed')

# ============ Cell 5: pooled selection (needs stage 1 of all three sessions in LOGDIR) - repeat 10 at this density =============
pb = pooled(5)
print('##### top-5 configs by mean val over the three densities (test = mean of the repeat-3 stage-1 values)')
print(pb.to_string(index=False))
if len(pb):
    cfg = cfg_of(pb.iloc[0])
    r = top(1000)
    mine = r[(r[KEYS] == pd.Series(dict(zip(KEYS, cfg)))).all(axis=1)].iloc[0]     # this density's val-best recipe for it
    down = f"{mine['drop']:g};{mine.wd:g}"
    if not done(cfg, 'pooled', down):
        run(cfg, down, 10, 'pooled')

# ============ Cell 6: tables (run where all three cora1_C*.jsonl files are present) =============
df = load(); assert len(df), 'no logs found in LOGDIR'
s1 = df[df.stage == 'stage1']
one = s1.sort_values('val', ascending=False).drop_duplicates(['ratio'] + KEYS)
print('##### stage 1+2: val-best config per ratio   [repeat 3]')
print(one.groupby('ratio').head(1)[['ratio'] + KEYS + ['drop', 'wd', 'val', 'test', 'std', 't_test', 'agree', 'ent_after']].to_string(index=False))
for ax in ['kernel', 'gamma', 'temp', 'mu', 'space', 'depth', 'drop', 'wd']:
    src = one if ax not in ('drop', 'wd') else s1.sort_values('val', ascending=False)
    print(f'\n##### val-selected test per ratio x {ax}')
    print(src.groupby(['ratio', ax]).head(1).pivot_table(index='ratio', columns=ax, values='test').round(2).to_string())
print('\n##### val-selected test per ratio x kernel x gamma')
print(one.groupby(['ratio', 'kernel', 'gamma']).head(1).pivot_table(index=['ratio', 'kernel'], columns='gamma', values='test').round(2).to_string())
print('\n##### val-selected test per ratio x mu x temp')
print(one.groupby(['ratio', 'mu', 'temp']).head(1).pivot_table(index=['ratio', 'mu'], columns='temp', values='test').round(2).to_string())
print('\n##### teacher test accuracy per kernel x gamma (depth 2)')
print(s1[s1.depth == 2].drop_duplicates(['kernel', 'gamma']).pivot_table(index='kernel', columns='gamma', values='t_test').round(1).to_string())
print('\n##### selection noise per ratio: val-best test vs oracle, Spearman(val, test) over configs, top-3 by val')
for r, g in one.groupby('ratio'):
    g = g.sort_values('val', ascending=False)
    print(f"  {r:g}: val-best test {g.test.iloc[0]:.2f}  oracle {g.test.max():.2f} ({g.loc[g.test.idxmax(), KEYS].to_dict()})  "
          f"rho {g.val.corr(g.test, method='spearman'):+.2f}  top-3 test {g.test.iloc[:3].round(2).tolist()}  mean of top-10 test {g.test.iloc[:10].mean():.2f}")
print('\n##### pooled ranking (mean val over densities), top-5, with per-density repeat-3 test')
pb = pooled(5)
per = one.set_index(KEYS)
for _, row in pb.iterrows():
    key = tuple(row[k] for k in KEYS)
    tests = [per.loc[key].set_index('ratio').test.get(r, float('nan')) if isinstance(per.loc[key], pd.DataFrame) else float('nan') for r in RATIOS]
    print(f"  {dict(zip(KEYS, key))}  mean val {row.val:.2f}  test per density {[round(t, 2) for t in tests]}")
for stage, title in [('final', 'MAIN ROWS - top-3 val configs at repeat 10 (first row per ratio = val-best)'),
                     ('pooled', 'POOLED ROW - one config for all densities (mean-val best), repeat 10'),
                     ('fixed', 'fixed recipe (dropout 0.5, wd 5e-4) at its val-best config, repeat 10'),
                     ('seed', 'condensation seeds 1, 2 at the val-best config, repeat 10')]:
    s = df[df.stage == stage].sort_values(['ratio', 'val'], ascending=[True, False])
    if not len(s):
        continue
    s = s.assign(cell=s.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f} (val {x['val']:.1f})", axis=1),
                 cfg=s.apply(lambda x: f"{x['kernel']} g={x['gamma']:g} T={x['temp']:g} mu={x['mu']:g} {x['space']} d={x['depth']} s={x['seed']} do={x['drop']:g} wd={x['wd']:g}", axis=1))
    print(f'\n##### {title}')
    print(s[['ratio', 'cell', 'cfg']].to_string(index=False))
