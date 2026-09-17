# ============ Cell 1: common (citeseer final run; three sessions: SESSION = 'C1' / 'C2' / 'C3' = 0.9 / 1.8 / 3.6%) =============
# citeseer needs a much stronger teacher regularisation than the large graphs (gamma 10-30) and, because such a teacher
# has flat posteriors, a per-node temperature T on the posteriors before cell averaging. Grid per cell (120 condensations):
#   erf kernel x gamma {1e-2, 1e-1, 1, 10, 30} x T {1, 0.5, 0.25} x feat_norm {0, 1} x mu {0.3, 1}
#   space raw, basis 3000 (= all nodes), l1 k-medians, depth 2; each run evaluates wd {5e-4, 5e-3} x 6 dropouts, repeat 3.
# Stage 2: val-best config at repeat 10 (main table) + fixed recipe (dropout 0.5, wd 5e-4) at its val-best config, repeat 10.
SESSION = 'C1'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'final4'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c "temp=args.teacher_temp" /content/CGC/main.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 --conv_depth 2 "
        "--no_hyperpara 1 --lr 0.01 --epoch 1000 --dropout 0.5 --weight_decay 5e-4 --dataset_name citeseer")
RATIO = {'C1': 0.009, 'C2': 0.018, 'C3': 0.036}[SESSION]
KERNELS = ['erf']                     # relu1 collapses at gamma >= 30 (teacher 68.6); erf stays at 72 up to gamma 100
GAMMAS  = [1e-2, 1e-1, 1.0, 10.0, 30.0]
TEMPS   = [1.0, 0.5, 0.25]
FNS     = [0, 1]
MUS     = [0.3, 1.0]
WDS     = [5e-4, 5e-3]
DOS     = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
DOWN1   = ','.join(f'{d:g}' for d in DOS) + ';' + ','.join(f'{w:g}' for w in WDS)
REP1    = 3
FIXED   = (0.5, 5e-4)
KEYS = ['kernel', 'gamma', 'temp', 'fn', 'mu']
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
    kernel, gamma, temp, fn, mu = cfg
    cmd = (f"python main.py {BASE} --ratio {RATIO} --label_kernel {kernel} --gamma {gamma} --teacher_temp {temp} "
           f"--feat_norm {fn} --bregman {mu} --repeat {repeat} --down_grid '{down}'")
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC')
    out = p.stdout + p.stderr
    rows = PAT_D.findall(out)
    if not rows:
        print('FAIL', cfg, stage, '\n', out[-1500:]); return
    ex = re.search(r'expert: train [\d.]+%\s+val ([\d.]+)%\s+test ([\d.]+)', out)
    en = re.search(r'posterior entropy ([\d.]+) -> ([\d.]+) nats', out)
    cd = re.search(r'cell diag.*', out); cd = cd[0] if cd else ''
    ct = re.search(r'Condensation time: ([0-9.]+)', out); ct = float(ct[1]) if ct else float('nan')
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds='citeseer', ratio=RATIO, kernel=kernel, gamma=gamma, temp=temp, fn=fn, mu=mu,
                                    drop=float(do_), wd=float(wd_), repeat=repeat, stage=stage, down=down,
                                    test=float(te), std=float(sd), val=float(va),
                                    t_val=float(ex[1]) if ex else None, t_test=float(ex[2]) if ex else None,
                                    ent_before=float(en[1]) if en else None, ent_after=float(en[2]) if en else None,
                                    diag=cd, cond_s=ct)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"citeseer {RATIO:g} {kernel:5s} g={gamma:<4g} T={temp:<4g} fn={fn} mu={mu:<3g} [{stage}]  teacher {ex[2] if ex else '?'}  "
          f"val {best[5]} (do={best[0]} wd={best[1]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

def pick(fixed=False):
    s1 = load()
    if not len(s1):
        return None
    s1 = s1[(s1.stage == 'stage1') & (s1.ratio == RATIO)]
    if fixed:
        s1 = s1[(s1['drop'] == FIXED[0]) & (s1.wd == FIXED[1])]
    return None if not len(s1) else s1.sort_values('val', ascending=False).iloc[0]

cfg_of = lambda b: (b.kernel, float(b.gamma), float(b.temp), int(b.fn), float(b.mu))
GRID = list(itertools.product(KERNELS, GAMMAS, TEMPS, FNS, MUS))
print(f'{SESSION}: citeseer {RATIO:g}, {len(GRID)} condensations, {len(DOS) * len(WDS)} recipes x repeat {REP1} each')

# ============ Cell 2: stage 1 - full grid (done() skips logged configs) =============
for cfg in GRID:
    if not done(cfg, 'stage1'):
        run(cfg, DOWN1, REP1, 'stage1')
b = pick()
print(f"--> citeseer {RATIO:g}: {cfg_of(b)} do {b['drop']:g} wd {b.wd:g}  val {b.val:.2f} test {b.test:.2f}")

# ============ Cell 3: stage 2 - repeat 10 at the val-best config and at the fixed recipe =============
b = pick()
if b is not None:
    down = f"{b['drop']:g};{b.wd:g}"
    if not done(cfg_of(b), 'final', down):
        run(cfg_of(b), down, 10, 'final')
f = pick(fixed=True)
if f is not None:
    down = f"{FIXED[0]:g};{FIXED[1]:g}"
    if not done(cfg_of(f), 'fixed', down):
        run(cfg_of(f), down, 10, 'fixed')

# ============ Cell 4: tables (run where all three final4_C*.jsonl files are present) =============
df = load(); assert len(df), 'no logs found in LOGDIR'
s1 = df[df.stage == 'stage1']
print('##### stage 1: val-selected config per ratio   [repeat 3]')
b1 = s1.sort_values('val', ascending=False).groupby('ratio').head(1)
print(b1[['ratio'] + KEYS + ['drop', 'wd', 'val', 'test', 'std', 't_test', 'ent_after']].to_string(index=False))
for ax in KEYS + ['drop', 'wd']:
    bx = s1.sort_values('val', ascending=False).groupby(['ratio', ax]).head(1)
    print(f'\n##### stage 1: val-selected test per ratio x {ax}')
    print(bx.pivot_table(index='ratio', columns=ax, values='test').round(2).to_string())
print('\n##### stage 1: val-selected test per ratio x kernel x gamma')
bg = s1.sort_values('val', ascending=False).groupby(['ratio', 'kernel', 'gamma']).head(1)
print(bg.pivot_table(index=['ratio', 'kernel'], columns='gamma', values='test').round(2).to_string())
print('\n##### teacher test accuracy per kernel x gamma x fn (ratio-independent)')
print(s1.drop_duplicates(['kernel', 'gamma', 'fn']).pivot_table(index=['kernel', 'fn'], columns='gamma', values='t_test').round(1).to_string())
top = lambda st: df[df.stage == st].sort_values('val', ascending=False).groupby('ratio').head(1)
for stage, title in [('final', 'MAIN TABLE - Ours (val-selected), repeat 10'),
                     ('fixed', 'Ours (fixed GCond/ClustGDD recipe: dropout 0.5, wd 5e-4), repeat 10')]:
    s = top(stage)
    if not len(s):
        continue
    s = s.assign(cell=s.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1),
                 cfg=s.apply(lambda x: f"{x['kernel']} g={x['gamma']:g} T={x['temp']:g} fn={x['fn']} mu={x['mu']:g} do={x['drop']:g} wd={x['wd']:g}", axis=1))
    print(f'\n##### {title}')
    print(s[['ratio', 'cell', 'cfg']].to_string(index=False))
