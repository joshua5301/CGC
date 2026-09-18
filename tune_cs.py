# ============ Cell 1: common (citeseer deep sweep; three sessions: SESSION = 'C1' / 'C2' / 'C3' = 0.9 / 1.8 / 3.6%) =============
# Everything learned so far about citeseer, swept in one protocol per density:
#   stage 1 (Cell 2)  full factorial  kernel/gamma {erf: 1, 3, 10, 30 | relu1: 0.3, 1, 3, 10}  x  T {1, 0.5, 0.25}
#                     x fn {0, 1} x mu {0.2, 0.5, 1, 2, 5}   at raw space, basis 3000, depth 2   = 240 condensations
#                     each: dropout {0,.1,.3,.5,.7,.9} x wd {1e-4, 5e-4, 5e-3} on val, repeat 3, eval every 10 epochs (~65 s)
#   stage 2 (Cell 3)  around the top-5 val configs: space nngp, basis {500, 1000}, depth {1, 3}, lr {5e-3, 2e-2}   = 35 more
#                     (nngp space was -0.2..-0.6 on citeseer in final5, so it is explored around the winners, not in the factorial)
#   stage 3 (Cell 4)  repeat 10 at the top-3 val configs (own recipe), at the fixed recipe (do .5, wd 5e-4),
#                     and condensation seeds {1, 2} at the val-best                                 ~ 8 runs
#   Cell 5            tables: marginals per axis, teacher accuracy, selection noise (val-test gap, oracle max), main rows
SESSION = 'C1'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'cs1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c sk_anneal /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --expert_basis 3000 --nngp_basis 3000 "
        "--no_hyperpara 1 --epoch 1000 --eval_every 10 --dropout 0.5 --weight_decay 5e-4 --dataset_name citeseer")
RATIO = {'C1': 0.009, 'C2': 0.018, 'C3': 0.036}[SESSION]
KG      = [('erf', g) for g in (1.0, 3.0, 10.0, 30.0)] + [('relu1', g) for g in (0.3, 1.0, 3.0, 10.0)]
TEMPS   = [1.0, 0.5, 0.25]
FNS     = [0, 1]
MUS     = [0.2, 0.5, 1.0, 2.0, 5.0]
BASES   = [500, 1000]            # stage 2; stage 1 uses 3000 (= all nodes)
DEPTHS  = [1, 3]                 # stage 2; stage 1 uses 2
WDS     = [1e-4, 5e-4, 5e-3]
DOS     = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
DOWN1   = ','.join(f'{d:g}' for d in DOS) + ';' + ','.join(f'{w:g}' for w in WDS)
REP1, TOPK, FIXED = 3, 3, (0.5, 5e-4)
KEYS = ['kernel', 'gamma', 'temp', 'fn', 'space', 'mu', 'basis', 'depth', 'lr', 'seed']
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')

def load():
    recs = []
    for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl'):
        for l in open(f):
            d = json.loads(l)
            if all(k in d for k in KEYS):
                recs.append(d)
    return pd.DataFrame(recs)

def done(cfg, stage, down=None):
    df = load()
    if not len(df):
        return False
    m = (df.ratio == RATIO) & (df.stage == stage)
    for k, v in zip(KEYS, cfg):
        m &= (df[k] == v)
    if down is not None:
        m &= (df.down == down)
    return m.any()

def run(cfg, down, repeat, stage):
    kernel, gamma, temp, fn, space, mu, basis, depth, lr, seed = cfg
    cmd = (f"python main.py {BASE} --ratio {RATIO} --label_kernel {kernel} --gamma {gamma} --teacher_temp {temp} "
           f"--feat_norm {fn} --cluster_feat {space} --bregman {mu} --expert_basis {basis} --nngp_basis {basis} "
           f"--conv_depth {depth} --lr {lr} --seed {seed} "
           f"--repeat {repeat} --down_grid '{down}'")
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
            f.write(json.dumps(dict(ds='citeseer', ratio=RATIO, kernel=kernel, gamma=gamma, temp=temp, fn=fn, space=space,
                                    mu=mu, basis=basis, depth=depth, lr=lr, seed=seed,
                                    drop=float(do_), wd=float(wd_), repeat=repeat, stage=stage, down=down,
                                    test=float(te), std=float(sd), val=float(va),
                                    t_val=float(ex[2]) if ex else None, t_test=float(ex[3]) if ex else None,
                                    ent_after=float(en[2]) if en else None,
                                    kl_g=float(gd[1]) if gd else None, agree=float(gd[2]) if gd else None,
                                    h_q=float(gd[4]) if gd else None, diag=cd, cond_s=ct)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"citeseer {RATIO:g} {kernel:5s} g={gamma:<4g} T={temp:<4g} fn={fn} {space:4s} mu={mu:<3g} b={basis} d={depth} lr={lr:g} s={seed} [{stage}]  "
          f"teacher {ex[3] if ex else '?'}  agree {gd[2] if gd else '?'}%  val {best[5]} (do={best[0]} wd={best[1]}) "
          f"test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

def top(k=1, fixed=False, stage='stage1', **fix):
    s1 = load()
    if not len(s1):
        return s1
    s1 = s1[(s1.stage == stage) & (s1.ratio == RATIO)]
    for kk, v in fix.items():
        s1 = s1[s1[kk] == v]
    if fixed:
        s1 = s1[(s1['drop'] == FIXED[0]) & (s1.wd == FIXED[1])]
    # one row per condensation config (its val-best recipe), then the k best configs by val
    s1 = s1.sort_values('val', ascending=False).drop_duplicates(KEYS)
    return s1.head(k)

cfg_of = lambda b: (b.kernel, float(b.gamma), float(b.temp), int(b.fn), b.space, float(b.mu), int(b.basis), int(b.depth), float(b.lr), int(b.seed))
GRID = [(k, g, T, fn, 'last', mu, 3000, 2, 0.01, 0) for (k, g), T, fn, mu in itertools.product(KG, TEMPS, FNS, MUS)]
print(f'{SESSION}: citeseer {RATIO:g}, stage 1 = {len(GRID)} condensations x {len(DOS) * len(WDS)} recipes x repeat {REP1}')

# ============ Cell 2: stage 1 - full factorial (done() skips logged configs) =============
for cfg in GRID:
    if not done(cfg, 'stage1'):
        run(cfg, DOWN1, REP1, 'stage1')
b = top(1).iloc[0]
print(f"--> citeseer {RATIO:g}: {cfg_of(b)} do {b['drop']:g} wd {b.wd:g}  val {b.val:.2f} test {b.test:.2f}")

# ============ Cell 3: stage 2 - neighbourhood of the top-5: space nngp, basis {500, 1000}, depth {1, 3}, lr {5e-3, 2e-2} (logged as stage1) =============
for _, b in top(5).iterrows():
    k_, g_, T_, fn_, sp_, mu_, b_, d_, lr_, s_ = cfg_of(b)
    for cfg in ([(k_, g_, T_, fn_, 'nngp', mu_, b_, d_, lr_, 0)]
                + [(k_, g_, T_, fn_, sp_, mu_, bs, d_, lr_, 0) for bs in BASES]
                + [(k_, g_, T_, fn_, sp_, mu_, b_, dp, lr_, 0) for dp in DEPTHS]
                + [(k_, g_, T_, fn_, sp_, mu_, b_, d_, lr, 0) for lr in (5e-3, 2e-2)]):
        if not done(cfg, 'stage1'):
            run(cfg, DOWN1, REP1, 'stage1')
b = top(1).iloc[0]
print(f"--> citeseer {RATIO:g} after stage 2: {cfg_of(b)} do {b['drop']:g} wd {b.wd:g}  val {b.val:.2f} test {b.test:.2f}")

# ============ Cell 4: stage 3 - repeat 10 at the top-3 (own recipe), fixed recipe, condensation seeds =============
for rank, (_, b) in enumerate(top(TOPK).iterrows()):
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

# ============ Cell 5: tables (run where all three cs1_C*.jsonl files are present) =============
df = load(); assert len(df), 'no logs found in LOGDIR'
s1 = df[df.stage == 'stage1']
one = s1.sort_values('val', ascending=False).drop_duplicates(['ratio'] + KEYS)      # val-best recipe per config
print('##### stage 1+2: val-best config per ratio   [repeat 3]')
print(one.groupby('ratio').head(1)[['ratio'] + KEYS + ['drop', 'wd', 'val', 'test', 'std', 't_test', 'agree', 'ent_after']].to_string(index=False))
for ax in ['kernel', 'gamma', 'temp', 'fn', 'space', 'mu', 'basis', 'depth', 'lr', 'drop', 'wd']:
    src = one if ax not in ('drop', 'wd') else s1.sort_values('val', ascending=False)
    bx = src.groupby(['ratio', ax]).head(1)
    print(f'\n##### val-selected test per ratio x {ax}')
    print(bx.pivot_table(index='ratio', columns=ax, values='test').round(2).to_string())
print('\n##### val-selected test per ratio x kernel x gamma')
print(one.groupby(['ratio', 'kernel', 'gamma']).head(1).pivot_table(index=['ratio', 'kernel'], columns='gamma', values='test').round(2).to_string())
print('\n##### teacher test accuracy per kernel x gamma x fn (basis 3000, depth 2)')
print(s1[(s1.basis == 3000) & (s1.depth == 2)].drop_duplicates(['kernel', 'gamma', 'fn']).pivot_table(index=['kernel', 'fn'], columns='gamma', values='t_test').round(1).to_string())
print('\n##### val-selected test per ratio x mu x temp (interaction)')
print(one.groupby(['ratio', 'mu', 'temp']).head(1).pivot_table(index=['ratio', 'mu'], columns='temp', values='test').round(2).to_string())
print('\n##### selection noise per ratio: val-best test vs oracle (max test over configs), Spearman(val, test) over configs, top-3 by val')
for r, g in one.groupby('ratio'):
    g = g.sort_values('val', ascending=False)
    print(f"  {r:g}: val-best test {g.test.iloc[0]:.2f}  oracle {g.test.max():.2f} ({g.loc[g.test.idxmax(), KEYS].to_dict()})  "
          f"rho {g.val.corr(g.test, method='spearman'):+.2f}  top-3 test {g.test.iloc[:3].round(2).tolist()}  mean of top-10 test {g.test.iloc[:10].mean():.2f}")
tops = lambda st: df[df.stage == st].sort_values('val', ascending=False).groupby('ratio')
for stage, title in [('final', 'MAIN ROWS - top-3 val configs at repeat 10 (first row per ratio = val-best)'),
                     ('fixed', 'fixed recipe (dropout 0.5, wd 5e-4) at its val-best config, repeat 10'),
                     ('seed', 'condensation seeds 1, 2 at the val-best config, repeat 10')]:
    s = df[df.stage == stage].sort_values(['ratio', 'val'], ascending=[True, False])
    if not len(s):
        continue
    s = s.assign(cell=s.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f} (val {x['val']:.1f})", axis=1),
                 cfg=s.apply(lambda x: f"{x['kernel']} g={x['gamma']:g} T={x['temp']:g} fn={x['fn']} {x['space']} mu={x['mu']:g} b={x['basis']} d={x['depth']} lr={x['lr']:g} s={x['seed']} do={x['drop']:g} wd={x['wd']:g}", axis=1))
    print(f'\n##### {title}')
    print(s[['ratio', 'cell', 'cfg']].to_string(index=False))
