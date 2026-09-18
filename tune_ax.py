# ============ Cell 1: common (arxiv sweep; three sessions: SESSION = 'A1' / 'A2' / 'A3' = 0.05 / 0.25 / 0.5%) =============
# Same protocol as the citeseer sweep (tune_cs.py) with the arxiv-specific axes fixed:
#   fixed   raw space, basis 3000 (k-means centroids), depth 2, fn 0, wd 5e-4, lr 0.01
#   stage 1 (Cell 2)  kernel {erf, relu1} x gamma {1e-4, 1e-3, 1e-2} x T {1, 0.5, 0.25} x mu {0.2, 0.5, 1, 2, 5}  = 90 condensations
#                     each: dropout {0,.1,.3,.5,.7} on val, repeat 3, eval every 10 epochs (~3-4 min)
#                     (gamma 1e-1 dropped: worst at every density by >= 0.6 in final5 / group1; dropout 0.9: -4 on large graphs)
#   stage 2 (Cell 3)  repeat 10 at the top-3 val configs (own dropout), at the fixed recipe (do .5), condensation seeds {1, 2}
#   Cell 4            tables: marginals, kernel x gamma, mu x T, selection noise, main rows
SESSION = 'A1'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'ax1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c basis_mode /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--conv_depth 2 --feat_norm 0 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 "
        "--dropout 0.5 --weight_decay 5e-4 --dataset_name arxiv")
RATIO = {'A1': 0.0005, 'A2': 0.0025, 'A3': 0.005}[SESSION]
KERNELS = ['erf', 'relu1']
GAMMAS  = [1e-4, 1e-3, 1e-2]
TEMPS   = [1.0, 0.5, 0.25]
MUS     = [0.2, 0.5, 1.0, 2.0, 5.0]
DOS     = [0, 0.1, 0.3, 0.5, 0.7]
DOWN1   = ','.join(f'{d:g}' for d in DOS) + ';5e-4'
REP1, TOPK, FIXED = 3, 3, 0.5
KEYS = ['kernel', 'gamma', 'temp', 'mu', 'seed']
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
    kernel, gamma, temp, mu, seed = cfg
    cmd = (f"python main.py {BASE} --ratio {RATIO} --label_kernel {kernel} --gamma {gamma} --teacher_temp {temp} "
           f"--bregman {mu} --seed {seed} --repeat {repeat} --down_grid '{down}'")
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
            f.write(json.dumps(dict(ds='arxiv', ratio=RATIO, kernel=kernel, gamma=gamma, temp=temp, mu=mu, seed=seed,
                                    drop=float(do_), wd=float(wd_), repeat=repeat, stage=stage, down=down,
                                    test=float(te), std=float(sd), val=float(va),
                                    t_val=float(ex[2]) if ex else None, t_test=float(ex[3]) if ex else None,
                                    ent_after=float(en[2]) if en else None,
                                    kl_g=float(gd[1]) if gd else None, agree=float(gd[2]) if gd else None,
                                    h_q=float(gd[4]) if gd else None, diag=cd, cond_s=ct)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"arxiv {RATIO:g} {kernel:5s} g={gamma:<6g} T={temp:<4g} mu={mu:<3g} s={seed} [{stage}]  "
          f"teacher {ex[3] if ex else '?'}  agree {gd[2] if gd else '?'}%  val {best[5]} (do={best[0]}) "
          f"test {best[3]}±{best[4]}  ({round(time.time() - t)}s, cond {ct:.0f}s)")

def top(k=1, fixed=False, stage='stage1'):
    s1 = load()
    if not len(s1):
        return s1
    s1 = s1[(s1.stage == stage) & (s1.ratio == RATIO)]
    if fixed:
        s1 = s1[s1['drop'] == FIXED]
    s1 = s1.sort_values('val', ascending=False).drop_duplicates(KEYS)
    return s1.head(k)

cfg_of = lambda b: (b.kernel, float(b.gamma), float(b.temp), float(b.mu), int(b.seed))
GRID = [(k, g, T, mu, 0) for k, g, T, mu in itertools.product(KERNELS, GAMMAS, TEMPS, MUS)]
print(f'{SESSION}: arxiv {RATIO:g}, stage 1 = {len(GRID)} condensations x {len(DOS)} dropouts x repeat {REP1}')

# ============ Cell 2: stage 1 - full factorial (done() skips logged configs) =============
for cfg in GRID:
    if not done(cfg, 'stage1'):
        run(cfg, DOWN1, REP1, 'stage1')
b = top(1).iloc[0]
print(f"--> arxiv {RATIO:g}: {cfg_of(b)} do {b['drop']:g}  val {b.val:.2f} test {b.test:.2f}")

# ============ Cell 3: stage 2 - repeat 10 at the top-3 (own dropout), fixed recipe, condensation seeds =============
for _, b in top(TOPK).iterrows():
    down = f"{b['drop']:g};5e-4"
    if not done(cfg_of(b), 'final', down):
        run(cfg_of(b), down, 10, 'final')
f = top(1, fixed=True)
if len(f):
    f = f.iloc[0]
    if not done(cfg_of(f), 'fixed', f'{FIXED:g};5e-4'):
        run(cfg_of(f), f'{FIXED:g};5e-4', 10, 'fixed')
b = top(1).iloc[0]; down = f"{b['drop']:g};5e-4"
for seed in (1, 2):
    cfg = cfg_of(b)[:-1] + (seed,)
    if not done(cfg, 'seed', down):
        run(cfg, down, 10, 'seed')

# ============ Cell 4: tables (run where all three ax1_A*.jsonl files are present) =============
df = load(); assert len(df), 'no logs found in LOGDIR'
s1 = df[df.stage == 'stage1']
one = s1.sort_values('val', ascending=False).drop_duplicates(['ratio'] + KEYS)
print('##### stage 1: val-best config per ratio   [repeat 3]')
print(one.groupby('ratio').head(1)[['ratio'] + KEYS + ['drop', 'val', 'test', 'std', 't_test', 'agree', 'ent_after', 'cond_s']].to_string(index=False))
for ax in ['kernel', 'gamma', 'temp', 'mu', 'drop']:
    src = one if ax != 'drop' else s1.sort_values('val', ascending=False)
    print(f'\n##### val-selected test per ratio x {ax}')
    print(src.groupby(['ratio', ax]).head(1).pivot_table(index='ratio', columns=ax, values='test').round(2).to_string())
print('\n##### val-selected test per ratio x kernel x gamma')
print(one.groupby(['ratio', 'kernel', 'gamma']).head(1).pivot_table(index=['ratio', 'kernel'], columns='gamma', values='test').round(2).to_string())
print('\n##### val-selected test per ratio x mu x temp')
print(one.groupby(['ratio', 'mu', 'temp']).head(1).pivot_table(index=['ratio', 'mu'], columns='temp', values='test').round(2).to_string())
print('\n##### teacher test accuracy per kernel x gamma')
print(s1.drop_duplicates(['kernel', 'gamma']).pivot_table(index='kernel', columns='gamma', values='t_test').round(2).to_string())
print('\n##### selection noise per ratio: val-best test vs oracle, Spearman(val, test) over configs, top-3 by val')
for r, g in one.groupby('ratio'):
    g = g.sort_values('val', ascending=False)
    print(f"  {r:g}: val-best test {g.test.iloc[0]:.2f}  oracle {g.test.max():.2f} ({g.loc[g.test.idxmax(), KEYS].to_dict()})  "
          f"rho {g.val.corr(g.test, method='spearman'):+.2f}  top-3 test {g.test.iloc[:3].round(2).tolist()}  mean of top-10 test {g.test.iloc[:10].mean():.2f}")
for stage, title in [('final', 'MAIN ROWS - top-3 val configs at repeat 10 (first row per ratio = val-best)'),
                     ('fixed', 'fixed recipe (dropout 0.5, wd 5e-4) at its val-best config, repeat 10'),
                     ('seed', 'condensation seeds 1, 2 at the val-best config, repeat 10')]:
    s = df[df.stage == stage].sort_values(['ratio', 'val'], ascending=[True, False])
    if not len(s):
        continue
    s = s.assign(cell=s.apply(lambda x: f"{x['test']:.2f}+-{x['std']:.2f} (val {x['val']:.2f})", axis=1),
                 cfg=s.apply(lambda x: f"{x['kernel']} g={x['gamma']:g} T={x['temp']:g} mu={x['mu']:g} s={x['seed']} do={x['drop']:g}", axis=1))
    print(f'\n##### {title}')
    print(s[['ratio', 'cell', 'cfg']].to_string(index=False))
