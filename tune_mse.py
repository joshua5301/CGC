# ============ Cell 1: common (SESSION = 'A' arxiv | 'B' reddit | 'C1' cora | 'C2' citeseer | 'C3' flickr) =============
# Closed-form teacher sweep: kernel ridge regression (--teacher_loss mse) on the erf NNGP kernel = the infinite-width
# network trained with MSE. KRR outputs are scores, so their posteriors are set by an ENTROPY TARGET (--teacher_ent,
# nats; temperature solved by bisection) instead of a temperature - scale-free, comparable across teachers.
# Coordinate search on val per cell:
#   stage 1  large graphs: gamma {1e-4 .. 1e-1} x ent {0.3, 0.6, 0.9}                 at mu 1     (12)
#            small graphs: fn {0,1} x gamma {1e-2 .. 30} x ent {0.3, 0.6, 0.9}       at mu 0.3   (30)
#   stage 2  at the val-best: the other mu values (large {0.3, 3}, small {1})
# Fixed: erf, raw space, basis 3000, l1 k-medians, depth 2; wd {5e-4, 5e-3} x dropout {0,.1,.3,.5,.7,.9}, repeat 3.
# Stage 3 (Cell 3): repeat 10 at the val-best + fixed recipe + mu=0 ablation.
SESSION = 'A'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'mse1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c teacher_loss /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --teacher_loss mse --label_kernel erf --kernel_prior rkhs "
        "--cluster_obj l1 --cluster_feat last --expert_basis 3000 --conv_depth 2 "
        "--no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 --dropout 0.5 --weight_decay 5e-4")
RATIOS = {'cora': [0.013, 0.026, 0.052], 'citeseer': [0.009, 0.018, 0.036],
          'arxiv': [0.0005, 0.0025, 0.005], 'flickr': [0.001, 0.005, 0.01],
          'reddit': [0.0005, 0.001, 0.002]}
GROUPS = {'A': [('arxiv', r) for r in RATIOS['arxiv']], 'B': [('reddit', r) for r in RATIOS['reddit']],
          'C1': [('cora', r) for r in RATIOS['cora']], 'C2': [('citeseer', r) for r in RATIOS['citeseer']],
          'C3': [('flickr', r) for r in RATIOS['flickr']]}
CELLS = GROUPS[SESSION]
LARGE = lambda ds: ds in ('arxiv', 'reddit')
GAMMAS = lambda ds: [1e-4, 1e-3, 1e-2, 1e-1] if LARGE(ds) else [1e-2, 1e-1, 1.0, 10.0, 30.0]
FNS    = lambda ds: [0] if LARGE(ds) else [0, 1]
ENTS   = [0.3, 0.6, 0.9]
MU1    = lambda ds: 1.0 if LARGE(ds) else 0.3
MU2    = lambda ds: [0.3, 3.0] if LARGE(ds) else [1.0]
WDS, DOS = [5e-4, 5e-3], [0, 0.1, 0.3, 0.5, 0.7, 0.9]
DOWN1 = ','.join(f'{d:g}' for d in DOS) + ';' + ','.join(f'{w:g}' for w in WDS)
REP1, FIXED = 3, (0.5, 5e-4)
KEYS = ['fn', 'gamma', 'ent', 'mu']
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
    fn, gamma, ent, mu = cfg
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --feat_norm {fn} --gamma {gamma} "
           f"--teacher_ent {ent} --bregman {mu} --repeat {repeat} --down_grid '{down}'")
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC')
    out = p.stdout + p.stderr
    rows = PAT_D.findall(out)
    if not rows:
        print('FAIL', ds, r, cfg, stage, '\n', out[-1500:]); return
    ex = re.search(r'expert: train ([\d.]+)%(?:\s+val ([\d.]+)%)?(?:\s+test ([\d.]+))?', out)
    en = re.search(r'posterior entropy ([\d.]+) -> ([\d.]+) nats at T=([\d.]+)', out)
    cd = re.search(r'cell diag.*', out); cd = cd[0] if cd else ''
    ct = re.search(r'Condensation time: ([0-9.]+)', out); ct = float(ct[1]) if ct else float('nan')
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, fn=fn, gamma=gamma, ent=ent, mu=mu,
                                    drop=float(do_), wd=float(wd_), repeat=repeat, stage=stage, down=down,
                                    test=float(te), std=float(sd), val=float(va),
                                    t_train=float(ex[1]) if ex else None, t_val=float(ex[2]) if ex and ex[2] else None, t_test=float(ex[3]) if ex and ex[3] else None,
                                    ent_raw=float(en[1]) if en else None, temp=float(en[3]) if en else None,
                                    diag=cd, cond_s=ct)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"{ds:8s} r={r:<7g} fn={fn} g={gamma:<5g} ent={ent:<3g} mu={mu:<3g} [{stage}]  teacher {(ex[3] or ex[1]) if ex else '?'}  "
          f"T={en[3] if en else '?'}  val {best[5]} (do={best[0]} wd={best[1]}) test {best[3]}±{best[4]}  "
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

cfg_of = lambda b: (int(b.fn), float(b.gamma), float(b.ent), float(b.mu))
print(f'{SESSION}: cells {CELLS}')

# ============ Cell 2: stage 1-2 coordinate search (logged as stage1; done() resumes) =============
for ds, r in CELLS:
    for fn, gamma, ent in itertools.product(FNS(ds), GAMMAS(ds), ENTS):                    # stage 1
        cfg = (fn, gamma, ent, MU1(ds))
        if not done(ds, r, cfg, 'stage1'):
            run(ds, r, cfg, DOWN1, REP1, 'stage1')
    b = pick(ds, r, mu=MU1(ds))
    for mu in MU2(ds):                                                                      # stage 2
        cfg = (int(b.fn), float(b.gamma), float(b.ent), mu)
        if not done(ds, r, cfg, 'stage1'):
            run(ds, r, cfg, DOWN1, REP1, 'stage1')
    b = pick(ds, r)
    print(f"--> {ds} {r:g}: {cfg_of(b)} do {b['drop']:g} wd {b.wd:g}  val {b.val:.2f} test {b.test:.2f}")

# ============ Cell 3: repeat 10 at the val-best config, at the fixed recipe, and with mu = 0 =============
for ds, r in CELLS:
    b = pick(ds, r)
    if b is None:
        continue
    down = f"{b['drop']:g};{b.wd:g}"
    if not done(ds, r, cfg_of(b), 'final', down):
        run(ds, r, cfg_of(b), down, 10, 'final')
    f = pick(ds, r, fixed=True)
    if f is not None:
        dfx = f"{FIXED[0]:g};{FIXED[1]:g}"
        if not done(ds, r, cfg_of(f), 'fixed', dfx):
            run(ds, r, cfg_of(f), dfx, 10, 'fixed')
    cfg0 = cfg_of(b)[:3] + (0.0,)
    if not done(ds, r, cfg0, 'mu0', down):
        run(ds, r, cfg0, down, 10, 'mu0')

# ============ Cell 4: tables (run where all mse1_*.jsonl files are present) =============
df = load(); assert len(df), 'no logs found in LOGDIR'
s1 = df[df.stage == 'stage1']
print('##### stage 1: val-selected config per (ds, ratio)   [repeat 3]')
b1 = s1.sort_values('val', ascending=False).groupby(['ds', 'ratio']).head(1)
print(b1[['ds', 'ratio'] + KEYS + ['temp', 'drop', 'wd', 'val', 'test', 'std', 't_test', 'cond_s']].sort_values(['ds', 'ratio']).to_string(index=False))
for ax in KEYS + ['drop', 'wd']:
    bx = s1.sort_values('val', ascending=False).groupby(['ds', 'ratio', ax]).head(1)
    print(f'\n##### stage 1: val-selected test per (ds, ratio) x {ax}')
    print(bx.pivot_table(index=['ds', 'ratio'], columns=ax, values='test').round(2).to_string())
print('\n##### teacher test accuracy per (ds, fn) x gamma')
print(s1.drop_duplicates(['ds', 'fn', 'gamma']).pivot_table(index=['ds', 'fn'], columns='gamma', values='t_test').round(1).to_string())
print('\n##### condensation time [s] (median over configs) per ds x ratio')
print(s1.drop_duplicates(['ds', 'ratio'] + KEYS).groupby(['ds', 'ratio']).cond_s.median().round(1).to_string())
top = lambda st: df[df.stage == st].sort_values('val', ascending=False).groupby(['ds', 'ratio']).head(1)
for stage, title in [('final', 'MAIN TABLE (mse teacher) - val-selected, repeat 10'),
                     ('fixed', 'fixed GCond/ClustGDD recipe (dropout 0.5, wd 5e-4), repeat 10'),
                     ('mu0', 'ablation: mu = 0 (no KL refinement), repeat 10')]:
    s = top(stage)
    if not len(s):
        continue
    s = s.assign(cell=s.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1),
                 cfg=s.apply(lambda x: f"fn={x['fn']} g={x['gamma']:g} ent={x['ent']:g} mu={x['mu']:g} do={x['drop']:g} wd={x['wd']:g}", axis=1))
    print(f'\n##### {title}')
    print(s.pivot_table(index='ds', columns='ratio', values='cell', aggfunc='first').to_string())
    if stage == 'final':
        print(s.pivot_table(index='ds', columns='ratio', values='cfg', aggfunc='first').to_string())
