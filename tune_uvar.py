# ============ Cell 1: common (SESSION = 'A' cora 5.2% | 'B' citeseer 3.6%) =============
# Uncertainty-aware partition (--uvar_lambda): the CE decomposition adds a label-estimation term sum_j mean_{t in j} s_t to the
# l1 objective, s_t = teacher-error variance proxy of node t from its distance to the nearest training node (per-quartile
# coefficient V fitted on VAL labels: far nodes ~2-3x the variance of near nodes, avg_diag). Nodes with large s_t are pushed
# into large cells (their error is averaged away), certain nodes keep small cells. Labels stay plain cell means.
# lambda {0, 0.3, 1, 3, 10} x T, dropout/wd on val, repeat 5; oracle-label ceiling as reference.
SESSION = 'A'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'uvar1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c uvar_bins /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--refine_teacher kernel --conv_depth 2 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 "
        "--dropout 0.5 --weight_decay 5e-4")
CELL = {'A': ('cora',     0.052, 'relu1', 0.01, 0, 2.0, [1.0, 0.5]),
        'B': ('citeseer', 0.036, 'erf',   3.0,  1, 0.2, [1.0, 0.5, 0.25])}[SESSION]
LAMS = [0.0, 0.3, 1.0, 3.0, 10.0]
DOWN = '0,0.1,0.3,0.5,0.7,0.9;5e-4,5e-3'
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+) .*argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)')
PAT_U = re.compile(r'uvar: lambda [\d.]+  sum_j mean s_j = ([\d.]+) \(uniform cells would give ([\d.]+)\)  cell size min/med/max (\d+)/(\d+)/(\d+)')
PAT_F = re.compile(r'uvar fit .*?V = \[([^\]]*)\]')
PAT_C = re.compile(r'cell diag: within-var ([\d.]+) \(([\d.]+)% of total\).*size min/p10/med/p90/max (\d+)/(\d+)/(\d+)/(\d+)/(\d+)')

def load():
    return pd.DataFrame([json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)])

def done(ds, r, lam, temp, oracle=0):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.lam == lam) & (df.temp == temp) & (df.oracle == oracle)).any()

def run(ds, r, kernel, gamma, fn, mu, temp, lam, oracle=0):
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --gamma {gamma} --feat_norm {fn} "
           f"--bregman {mu} --teacher_temp {temp} --uvar_lambda {lam} --label_oracle {oracle} --repeat {REPEAT} --down_grid '{DOWN}'")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    rows = PAT_D.findall(out); g = PAT_G.search(out); u = PAT_U.search(out); fv = PAT_F.search(out); c = PAT_C.search(out)
    if not rows or not g:
        print('FAIL', ds, r, lam, temp, oracle, '\n', out[-2500:]); return
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, lam=lam, temp=temp, oracle=oracle, drop=float(do_), wd=float(wd_), repeat=REPEAT,
                                    test=float(te), std=float(sd), val=float(va),
                                    kl_g=float(g[1]), agree=float(g[2]), h_true=float(g[3]), h_q=float(g[4]),
                                    obj=float(u[1]) if u else None, obj_uniform=float(u[2]) if u else None,
                                    size_min=int(u[3]) if u else (int(c[3]) if c else None), size_med=int(u[4]) if u else (int(c[5]) if c else None),
                                    size_max=int(u[5]) if u else (int(c[7]) if c else None), within=float(c[2]) if c else None,
                                    V=fv[1] if fv else None)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"{ds:8s} {r:<6g} lam={lam:<4g} T={temp:<4g}{' ORACLE' if oracle else ''}" + (f"  obj {u[1]} (unif {u[2]})  size {u[3]}/{u[4]}/{u[5]}" if u else (f"  size {c[3]}/{c[5]}/{c[7]}" if c else '')) +
          f"  within-var {c[2] if c else '?'}%  agree {g[2]}%  KL {g[1]}  val {best[5]} (do={best[0]} wd={best[1]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

ds, r, kernel, gamma, fn, mu, temps = CELL
print(f'uvar1 {SESSION}: {ds} {r:g}  lambda {LAMS} x T {temps}')

# ============ Cell 2: run (done() resumes) =============
for lam, temp in itertools.product(LAMS, temps):
    if not done(ds, r, lam, temp):
        run(ds, r, kernel, gamma, fn, mu, temp, lam)
if not done(ds, r, 0.0, temps[0], 1):
    run(ds, r, kernel, gamma, fn, mu, temps[0], 0.0, 1)

# ============ Cell 3: tables (run where both uvar1_*.jsonl are present) =============
df = load()
b = df.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'lam', 'temp', 'oracle']).head(1)
b = b.assign(cell=b.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1))
bn = b[b.oracle == 0]
print('##### student test per (ds, T) x lambda   [val-selected dropout/wd, repeat 5]')
print(bn.pivot_table(index=['ds', 'temp'], columns='lam', values='cell', aggfunc='first').to_string())
bb = bn.sort_values('val', ascending=False).groupby(['ds', 'lam']).head(1)
print('\n##### T selected on val, per ds x lambda   (oracle-label ceiling in the last column)')
t = bb.assign(cell=bb.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f} (T{x['temp']:g})", axis=1)).pivot_table(index='ds', columns='lam', values='cell', aggfunc='first')
t['oracle'] = b[b.oracle == 1].groupby('ds').test.first().round(2)
print(t.to_string())
print('\n##### val-selected over (lambda, T) per ds  vs  lambda 0')
for d, g in bn.groupby('ds'):
    top = g.sort_values('val', ascending=False).iloc[0]; base = g[g.lam == 0].sort_values('val', ascending=False).iloc[0]
    print(f"  {d}: best-val lambda {top.lam:g} T {top.temp:g}  test {top.test:.2f}+-{top['std']:.2f}   | lambda 0 (T {base.temp:g}): {base.test:.2f}   | oracle over grid {g.test.max():.2f}")
print('\n##### partition statistics per (ds) x lambda   [T 1]: objective sum_j mean s_j (uniform reference), cell size min/med/max, within-cell posterior var, cell agreement, group KL')
for m in ['obj', 'obj_uniform', 'size_min', 'size_med', 'size_max', 'within', 'agree', 'kl_g']:
    print(m.ljust(12), bn[bn.temp == 1.0].pivot_table(index='ds', columns='lam', values=m, aggfunc='first').round(3).to_string().replace('\n', '\n' + ' ' * 12))
print('\n##### fitted V per distance quartile (val labels):', bn.groupby('ds').V.first().to_dict())
