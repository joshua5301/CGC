# ============ Cell 1: common (SESSION = 'A' cora | 'B' citeseer; three densities each) =============
# Diagnostic: the GRIP partition is kept, the condensed labels are replaced by the Dirichlet-multinomial posterior predictive
# from the in-cell TRAINING labels,  q_j = (k_j + alpha r) / (n_j^L + alpha)  (--dir_alpha alpha, --dir_prior r):
#   global  : r = global training class frequencies -> cells without a training label get r (no teacher at all)
#   teacher : r = the teacher's cell mean -> the teacher is the prior, the in-cell training labels correct it
# Reported per run: cells with 0/1/2+ training labels, pool share of unlabelled cells, leave-one-out CE / accuracy of the
# training labels under the model vs the teacher's cell labels on the same nodes, and the student.
# Per-density protocol-A settings (raw space, basis 3000, depth 2); alpha {1, 3, 10} x prior; dropout/wd on val; repeat 5.
SESSION = 'A'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'dir1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c dir_prior /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--refine_teacher kernel --conv_depth 2 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 "
        "--dropout 0.5 --weight_decay 5e-4")
#                 ds        ratio  kernel   gamma fn  mu   T      (protocol-A val-best per density)
CELLS = {'A': [('cora',     0.013, 'relu1', 0.1,  0, 1.0, 1.0),
               ('cora',     0.026, 'relu1', 0.1,  0, 1.0, 2.0),
               ('cora',     0.052, 'relu1', 0.01, 0, 2.0, 1.0)],
         'B': [('citeseer', 0.009, 'erf',   10.0, 1, 0.2, 0.25),
               ('citeseer', 0.018, 'erf',   10.0, 1, 0.2, 0.25),
               ('citeseer', 0.036, 'erf',   3.0,  1, 0.2, 0.25)]}[SESSION]
ALPHAS = [1.0, 3.0, 10.0]
PRIORS = ['global', 'teacher']
DOWN = '0,0.1,0.3,0.5,0.7,0.9;1e-4,5e-4,5e-3'
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+) .*argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)')
PAT_R = re.compile(r'dirichlet labels: .*?cells with 0/1/2\+ training labels (\d+)/(\d+)/(\d+) \(pool share of unlabelled cells ([\d.]+)%\)  '
                   r'LOO train-label CE ([\d.]+) acc ([\d.]+)%  \| teacher cell labels on the same: CE ([\d.]+) acc ([\d.]+)%  \| label argmax changed ([\d.]+)%  H ([\d.]+) -> ([\d.]+)')

def load():
    return pd.DataFrame([json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)])

def done(ds, r, alpha, prior):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.alpha == alpha) & (df.prior == prior)).any()

def run(ds, r, kernel, gamma, fn, mu, temp, alpha, prior):
    extra = '' if alpha == 0 else f' --dir_alpha {alpha} --dir_prior {prior}'
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --gamma {gamma} --feat_norm {fn} "
           f"--bregman {mu} --teacher_temp {temp} --repeat {REPEAT} --down_grid '{DOWN}'{extra}")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    rows = PAT_D.findall(out); g = PAT_G.search(out); d = PAT_R.search(out)
    if not rows or not g or (alpha > 0 and not d):
        print('FAIL', ds, r, alpha, prior, '\n', out[-2500:]); return
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, alpha=alpha, prior=prior, temp=temp, drop=float(do_), wd=float(wd_), repeat=REPEAT,
                                    test=float(te), std=float(sd), val=float(va),
                                    kl_g=float(g[1]), agree=float(g[2]), h_true=float(g[3]), h_q=float(g[4]),
                                    c0=int(d[1]) if d else None, c1=int(d[2]) if d else None, c2=int(d[3]) if d else None, unl_pool=float(d[4]) if d else None,
                                    loo_ce=float(d[5]) if d else None, loo_acc=float(d[6]) if d else None, t_ce=float(d[7]) if d else None, t_acc=float(d[8]) if d else None,
                                    flipped=float(d[9]) if d else None, h0=float(d[10]) if d else None, h1=float(d[11]) if d else None)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    tag = 'teacher mean        ' if alpha == 0 else f'alpha={alpha:<3g} {prior:8s}'
    print(f"{ds:8s} {r:<6g} {tag}" + (f"  cells 0/1/2+ labels {d[1]}/{d[2]}/{d[3]} (unlabelled cells hold {d[4]}% of pool)  LOO CE {d[5]} acc {d[6]}% (teacher {d[7]} / {d[8]}%)  argmax changed {d[9]}%  H {d[10]}->{d[11]}" if d else '') +
          f"  agree {g[2]}%  KL {g[1]}  val {best[5]} (do={best[0]} wd={best[1]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

print(f'dir1 {SESSION}: {[(c[0], c[1]) for c in CELLS]}  alpha {ALPHAS} x prior {PRIORS}')

# ============ Cell 2: run (done() resumes) =============
for ds, r, kernel, gamma, fn, mu, temp in CELLS:
    if not done(ds, r, 0.0, 'none'):
        run(ds, r, kernel, gamma, fn, mu, temp, 0.0, 'none')
    for alpha, prior in itertools.product(ALPHAS, PRIORS):
        if not done(ds, r, alpha, prior):
            run(ds, r, kernel, gamma, fn, mu, temp, alpha, prior)

# ============ Cell 3: tables (run where both dir1_*.jsonl are present) =============
df = load()
b = df.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'alpha', 'prior']).head(1)
b = b.assign(cell=b.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f} (val {x['val']:.1f})", axis=1), key=b.apply(lambda x: 'teacher mean' if x.alpha == 0 else f"{x.prior} a{x.alpha:g}", axis=1))
print('##### student test per (ds, ratio) x labels   [dropout/wd on val; repeat 5]')
print(b.pivot_table(index=['ds', 'ratio'], columns='key', values='cell', aggfunc='first').to_string())
print('\n##### label statistics per (ds, ratio) x labels: cells with 0/1/2+ training labels, pool share of unlabelled cells, LOO CE / acc of training labels (teacher cell labels: CE / acc), cell agreement with the true composition, group KL')
for _, x in b.sort_values(['ds', 'ratio', 'prior', 'alpha']).iterrows():
    if x.alpha == 0:
        print(f"  {x.ds:8s} {x.ratio:<6g} teacher mean          agree {x.agree:5.1f}%  KL {x.kl_g:.3f}")
    else:
        print(f"  {x.ds:8s} {x.ratio:<6g} {x.key:14s}  cells {int(x.c0)}/{int(x.c1)}/{int(x.c2)}  unlabelled-cell pool {x.unl_pool:5.1f}%  LOO CE {x.loo_ce:.3f} acc {x.loo_acc:5.1f}% (teacher {x.t_ce:.3f} / {x.t_acc:5.1f}%)  "
              f"argmax changed {x.flipped:4.1f}%  H {x.h0:.3f}->{x.h1:.3f}  agree {x.agree:5.1f}%  KL {x.kl_g:.3f}")
print('\n##### best over alpha per (ds, ratio, prior) vs teacher mean')
for (d_, r_), g in b.groupby(['ds', 'ratio']):
    base = g[g.alpha == 0].iloc[0]
    parts = []
    for pr, gg in g[g.alpha > 0].groupby('prior'):
        top = gg.sort_values('val', ascending=False).iloc[0]
        parts.append(f"{pr}: alpha {top.alpha:g} test {top.test:.2f} ({top.test - base.test:+.2f})")
    print(f"  {d_:8s} {r_:<6g} teacher mean {base.test:.2f}   | " + '   | '.join(parts))
