# ============ Cell 1: common (SESSION = 'all' | 'C1' / 'C2' / 'C3' = cora 1.3 / 2.6 / 5.2%) =============
# Posterior propagation (--teacher_prop k --prop_alpha a [--prop_seed 1]): the teacher's node posteriors are smoothed on the
# original graph, P <- (1-a) F + a A_hat P (k steps; with prop_seed the training rows are clamped to their labels), before
# the KL refinement and the cell averaging. Paired against no propagation at the cora1 val-best (T <= 1) condensation per
# density, with T {1, 0.5} since propagation flattens the posteriors. Logs teacher val/test before and after, cell agreement, student.
SESSION = 'all'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'prop1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c prop_alpha /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--conv_depth 2 --feat_norm 0 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 "
        "--dropout 0.5 --weight_decay 5e-4 --dataset_name cora")
#          ratio   kernel   gamma  mu      (cora1 val-best with T <= 1, structural axes fixed)
CELLS = {'C1': [(0.013, 'relu1', 0.1, 1.0)], 'C2': [(0.026, 'relu1', 1.0, 5.0)], 'C3': [(0.052, 'relu1', 0.01, 2.0)]}
CELLS = CELLS['C1'] + CELLS['C2'] + CELLS['C3'] if SESSION == 'all' else CELLS[SESSION]
#          (k, alpha, seed)
PROPS = [(0, 0.0, 0), (1, 0.5, 0), (2, 0.5, 0), (2, 0.8, 0), (2, 0.5, 1), (2, 0.8, 1), (3, 0.8, 1)]
TEMPS = [1.0, 0.5]
DOWN = '0,0.1,0.3,0.5,0.7,0.9;5e-4'
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_P = re.compile(r'teacher_prop\[(before|refine)\]: .*?val ([\d.]+)%  test ([\d.]+)%  H ([\d.]+)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+) .*argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)')

def load():
    recs = [json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)]
    return pd.DataFrame(recs)

def done(r, prop, temp):
    df = load()
    return len(df) > 0 and ((df.ratio == r) & (df.k == prop[0]) & (df.alpha == prop[1]) & (df.seed_train == prop[2]) & (df.temp == temp)).any()

def run(r, kernel, gamma, mu, prop, temp):
    k, alpha, sd = prop
    cmd = (f"python main.py {BASE} --ratio {r} --label_kernel {kernel} --gamma {gamma} --bregman {mu} --teacher_temp {temp} "
           f"--teacher_prop {k} --prop_alpha {alpha} --prop_seed {sd} --repeat {REPEAT} --down_grid '{DOWN}'")
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC')
    out = p.stdout + p.stderr
    rows = PAT_D.findall(out)
    g = PAT_G.search(out)
    if not rows or not g:
        print('FAIL', r, prop, temp, '\n', out[-2000:]); return
    pr = dict((m[1], (float(m[2]), float(m[3]), float(m[4]))) for m in PAT_P.finditer(out))
    ex = re.search(r'expert: train ([\d.]+)%\s+val ([\d.]+)%\s+test ([\d.]+)', out)
    tb = pr.get('before', (float(ex[2]), float(ex[3]), float('nan')) if ex else (None, None, None))
    ta = pr.get('refine', tb)
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sdv, va in rows:
            f.write(json.dumps(dict(ds='cora', ratio=r, kernel=kernel, gamma=gamma, mu=mu, k=k, alpha=alpha, seed_train=sd, temp=temp,
                                    drop=float(do_), wd=float(wd_), repeat=REPEAT, test=float(te), std=float(sdv), val=float(va),
                                    t_val0=tb[0], t_test0=tb[1], t_val=ta[0], t_test=ta[1], h_t=ta[2],
                                    kl_g=float(g[1]), agree=float(g[2]), h_true=float(g[3]), h_q=float(g[4]))) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"cora {r:<6g} k={k} a={alpha:<3g} seed={sd} T={temp:<4g}  teacher val/test {tb[0]}/{tb[1]} -> {ta[0]}/{ta[1]}  "
          f"agree {g[2]}%  H(q) {g[4]}  val {best[5]} (do={best[0]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

print(f'{SESSION}: {CELLS}  props {PROPS}  T {TEMPS}')

# ============ Cell 2: run (done() resumes) =============
for r, kernel, gamma, mu in CELLS:
    for prop, temp in itertools.product(PROPS, TEMPS):
        if not done(r, prop, temp):
            run(r, kernel, gamma, mu, prop, temp)

# ============ Cell 3: tables =============
df = load(); assert len(df), 'no logs'
b = df.sort_values('val', ascending=False).groupby(['ratio', 'k', 'alpha', 'seed_train', 'temp']).head(1)
b = b.assign(prop=b.apply(lambda x: 'none' if x.k == 0 else f"k{int(x.k)} a{x.alpha:g}" + (' seed' if x.seed_train else ''), axis=1),
             cell=b.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1))
order = ['none'] + [f"k{k} a{a:g}" + (' seed' if s else '') for k, a, s in PROPS if k > 0]
for temp in TEMPS:
    print(f'\n##### student test per ratio x propagation   [T {temp:g}, val-selected dropout, wd 5e-4, repeat 5]')
    print(b[b.temp == temp].pivot_table(index='ratio', columns='prop', values='cell', aggfunc='first')[order].to_string())
print('\n##### student test per ratio x propagation, T selected on val')
bb = b.sort_values('val', ascending=False).groupby(['ratio', 'prop']).head(1)
print(bb.pivot_table(index='ratio', columns='prop', values='cell', aggfunc='first')[order].to_string())
print(bb.pivot_table(index='ratio', columns='prop', values='temp', aggfunc='first')[order].to_string())
for m, title in [('t_test', 'teacher TEST accuracy after propagation (before = none column)'),
                 ('t_val', 'teacher VAL accuracy after propagation'),
                 ('agree', 'cell argmax agreement with the true composition [%]'), ('kl_g', 'group KL(true||q)'),
                 ('h_q', 'H(q) (H(true) ~ 0.5)')]:
    print(f'\n##### {title}   [T 1]')
    print(b[b.temp == 1.0].pivot_table(index='ratio', columns='prop', values=m, aggfunc='first')[order].round(3).to_string())
