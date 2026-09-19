# ============ Cell 1: common (SESSION = 'A' arxiv 0.25% | 'B' reddit 0.1%) =============
# Nystrom basis size on the large graphs (cora / citeseer: 3000 >= N already). Stage 1 fits the teacher only
# (--teacher_only): basis {3000, 10000, 20000} x gamma {own, 10x} -> teacher val/test, time, peak memory. Stage 2 trains
# students (repeat 5) at every (basis, gamma) whose teacher is within 0.3 of the best or is the 3000 reference.
SESSION = 'A'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'basis2'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c "peak GPU mem" /content/CGC/main.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --refine_teacher kernel "
        "--conv_depth 2 --feat_norm 0 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 --weight_decay 5e-4")
#          ds        ratio   kernel  gamma  mu   T     dropouts               (protocol-A / final5 val-best)
CELL = {'A': ('arxiv',  0.0025, 'erf',  1e-4, 0.2, 0.25, '0,0.1,0.3,0.5,0.7'),
        'B': ('reddit', 0.001,  'erf',  1e-4, 1.0, 1.0,  '0,0.1,0.3,0.5,0.7')}[SESSION]
BASES = [3000, 10000, 20000]
GMULT = [1.0, 10.0]
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_T = re.compile(r'teacher_prop\[refine\]: .*?val ([\d.]+)%  test ([\d.]+)%  H ([\d.]+)')
PAT_TR = re.compile(r'teacher_prop\[refine\]: .*?train ([\d.]+)%  \(inductive')
PAT_E = re.compile(r'expert: train ([\d.]+)%(?:\s+val ([\d.]+)%)?(?:\s+test ([\d.]+))?')
PAT_M = re.compile(r'teacher_only: .*?elapsed ([\d.]+) s  peak GPU mem ([\d.]+) GB')

def load():
    return pd.DataFrame([json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)])

def done(**key):
    df = load()
    if not len(df):
        return False
    m = pd.Series(True, index=df.index)
    for k, v in key.items():
        m &= (df[k] == v) if k in df.columns else False
    return bool(m.any())

ds, r, kernel, gamma0, mu, temp, dos = CELL

def teacher(basis, gamma):
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --gamma {gamma} --bregman {mu} "
           f"--teacher_temp 1 --expert_basis {basis} --teacher_only 1")
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    t = PAT_T.search(out); m = PAT_M.search(out); e = PAT_E.search(out)
    if not m:
        print('FAIL teacher', basis, gamma, '\n', out[-1500:]); return
    tv = float(t[1]) if t else (float(e[2]) if e and e[2] else None); tt = float(t[2]) if t else (float(e[3]) if e and e[3] else None)
    tr_ = PAT_TR.search(out)
    rec = dict(stage='teacher', ds=ds, ratio=r, basis=basis, gamma=gamma, t_val=tv, t_test=tt, t_train=float(tr_[1]) if tr_ else (float(e[1]) if e else None),
               secs=float(m[1]), mem=float(m[2]))
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(f"  [teacher] basis {basis:6d} gamma {gamma:<6g}  train {rec['t_train']}  val {tv}  test {tt}   {rec['secs']:.0f} s  peak {rec['mem']:.1f} GB")

def student(basis, gamma):
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --gamma {gamma} --bregman {mu} "
           f"--teacher_temp {temp} --expert_basis {basis} --repeat {REPEAT} --down_grid '{dos};5e-4'")
    t0 = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    rows = PAT_D.findall(out); e = PAT_E.search(out)
    ct = re.search(r'Condensation time: ([0-9.]+)', out); ct = float(ct[1]) if ct else float('nan')
    if not rows:
        print('FAIL student', basis, gamma, '\n', out[-2000:]); return
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(stage='student', ds=ds, ratio=r, basis=basis, gamma=gamma, temp=temp, drop=float(do_), wd=float(wd_),
                                    repeat=REPEAT, test=float(te), std=float(sd), val=float(va),
                                    t_test=float(e[3]) if e and e[3] else (float(e[1]) if e else None), cond_s=ct)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"{ds} {r:g} basis {basis:6d} gamma {gamma:<6g}  val {best[5]} (do={best[0]}) test {best[3]}±{best[4]}  cond {ct:.0f} s  ({round(time.time() - t0)} s)")

print(f'basis2 {SESSION}: {ds} {r:g}  bases {BASES} x gamma {[gamma0 * g for g in GMULT]}')

# ============ Cell 2: stage 1 - teacher only =============
for basis, gm in itertools.product(BASES, GMULT):
    if not done(stage='teacher', basis=basis, gamma=gamma0 * gm):
        teacher(basis, gamma0 * gm)
t = load(); t = t[(t.stage == 'teacher') & (t.ds == ds)]
print(t.pivot_table(index='basis', columns='gamma', values=['t_val', 't_test', 'secs', 'mem']).round(2).to_string())

# ============ Cell 3: stage 2 - students where the teacher is competitive =============
t = load(); t = t[(t.stage == 'teacher') & (t.ds == ds)]
key = 't_val' if t.t_val.notna().all() else 't_train'
best = t[key].max()
for _, row in t.iterrows():
    if row[key] >= best - 0.3 or row.basis == 3000:
        if not done(stage='student', basis=int(row.basis), gamma=float(row.gamma)):
            student(int(row.basis), float(row.gamma))

# ============ Cell 4: tables (run where both basis2_*.jsonl are present) =============
df = load()
t = df[df.stage == 'teacher']
print('##### teacher accuracy (val / test; train for reddit) per (ds, basis) x gamma, with time and peak memory')
print(t.pivot_table(index=['ds', 'basis'], columns='gamma', values=['t_val', 't_test', 't_train', 'secs', 'mem']).round(2).to_string())
s = df[df.stage == 'student']
if len(s):
    b = s.sort_values('val', ascending=False).groupby(['ds', 'basis', 'gamma']).head(1)
    b = b.assign(cell=b.apply(lambda x: f"{x['test']:.2f}+-{x['std']:.2f} (val {x['val']:.2f}, cond {x['cond_s']:.0f}s)", axis=1))
    print('\n##### student test per (ds, basis) x gamma   [val-selected dropout, repeat 5]')
    print(b.pivot_table(index=['ds', 'basis'], columns='gamma', values='cell', aggfunc='first').to_string())
