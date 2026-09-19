# ============ Cell 1: common (SESSION = 'all' | 'C1' / 'C2' / 'C3' = cora 1.3 / 2.6 / 5.2%) =============
# Full Correct & Smooth (Huang et al. 2021) on the KERNEL teacher's posteriors (--refine_teacher kernel so the KL refinement
# and the labels use the same teacher): Correct = out-of-fold training residuals propagated (a1 x 50, autoscale),
# Smooth = labels clamped on the training rows, propagated (a2 x 50). Paired against no C&S at the cora1 val-best (T <= 1)
# condensation per density. Logs the kernel teacher's val/test before and after, cell agreement, student (repeat 5).
SESSION = 'all'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'cs2'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c tcs_smooth_iters /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--refine_teacher kernel --conv_depth 2 --feat_norm 0 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 "
        "--dropout 0.5 --weight_decay 5e-4 --dataset_name cora")
CELLS = {'C1': [(0.013, 'relu1', 0.1, 1.0)], 'C2': [(0.026, 'relu1', 1.0, 5.0)], 'C3': [(0.052, 'relu1', 0.01, 2.0)]}
CELLS = CELLS['C1'] + CELLS['C2'] + CELLS['C3'] if SESSION == 'all' else CELLS[SESSION]
#           name            flags                                                              T
VARIANTS = [('none',        '',                                                               1.0),
            ('C&S full',    '--tcs_smooth_iters 50 --tcs_correct 1 --tcs_resid oof',             1.0),
            ('C&S full T.5','--tcs_smooth_iters 50 --tcs_correct 1 --tcs_resid oof',             0.5),
            ('C&S a.5',     '--tcs_smooth_iters 50 --tcs_correct 1 --tcs_resid oof --tcs_alpha1 0.5 --tcs_alpha2 0.5', 1.0),
            ('smooth only', '--tcs_smooth_iters 50 --tcs_correct 0',                            1.0),
            ('correct only','--tcs_smooth_iters 1 --tcs_correct 1 --tcs_resid oof --tcs_alpha2 0', 1.0),
            ('insample',    '--tcs_smooth_iters 50 --tcs_correct 1 --tcs_resid insample',        1.0)]
DOWN = '0,0.1,0.3,0.5,0.7,0.9;5e-4'
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_P = re.compile(r'teacher_prop\[(before|refine)\]: .*?val ([\d.]+)%  test ([\d.]+)%  H ([\d.]+)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+) .*argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)')
PAT_O = re.compile(r'C&S: \d+-fold OOF teacher acc on train ([\d.]+)%  mean \|resid\|_1 ([\d.]+)')
PAT_M = re.compile(r'C&S correct: argmax changed on ([\d.]+)% of nodes')

def load():
    recs = [json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)]
    return pd.DataFrame(recs)

def done(r, name):
    df = load()
    return len(df) > 0 and ((df.ratio == r) & (df.variant == name)).any()

def run(r, kernel, gamma, mu, name, flags, temp):
    cmd = (f"python main.py {BASE} --ratio {r} --label_kernel {kernel} --gamma {gamma} --bregman {mu} --teacher_temp {temp} "
           f"{flags} --repeat {REPEAT} --down_grid '{DOWN}'")
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC')
    out = p.stdout + p.stderr
    rows = PAT_D.findall(out)
    g = PAT_G.search(out)
    if not rows or not g:
        print('FAIL', r, name, '\n', out[-2500:]); return
    pr = dict((m[1], (float(m[2]), float(m[3]), float(m[4]))) for m in PAT_P.finditer(out))
    ex = re.search(r'expert: train ([\d.]+)%\s+val ([\d.]+)%\s+test ([\d.]+)', out)
    tb = pr.get('before', (float(ex[2]), float(ex[3]), float('nan')) if ex else (None, None, None))
    ta = pr.get('refine', tb)
    oo, mv = PAT_O.search(out), PAT_M.search(out)
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sdv, va in rows:
            f.write(json.dumps(dict(ds='cora', ratio=r, kernel=kernel, gamma=gamma, mu=mu, variant=name, temp=temp,
                                    drop=float(do_), wd=float(wd_), repeat=REPEAT, test=float(te), std=float(sdv), val=float(va),
                                    t_val0=tb[0], t_test0=tb[1], t_val=ta[0], t_test=ta[1], h_t=ta[2],
                                    oof_acc=float(oo[1]) if oo else None, resid=float(oo[2]) if oo else None, moved=float(mv[1]) if mv else None,
                                    kl_g=float(g[1]), agree=float(g[2]), h_true=float(g[3]), h_q=float(g[4]))) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"cora {r:<6g} {name:13s} T={temp:<4g} teacher val/test {tb[0]}/{tb[1]} -> {ta[0]}/{ta[1]}"
          + (f"  OOF {oo[1]}% moved {mv[1] if mv else '?'}%" if oo else '')
          + f"  agree {g[2]}%  H(q) {g[4]}  val {best[5]} (do={best[0]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

print(f'{SESSION}: {CELLS}  variants {[v[0] for v in VARIANTS]}')

# ============ Cell 2: run (done() resumes) =============
for r, kernel, gamma, mu in CELLS:
    for name, flags, temp in VARIANTS:
        if not done(r, name):
            run(r, kernel, gamma, mu, name, flags, temp)

# ============ Cell 3: tables =============
df = load(); assert len(df), 'no logs'
b = df.sort_values('val', ascending=False).groupby(['ratio', 'variant']).head(1)
b = b.assign(cell=b.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1))
order = [v[0] for v in VARIANTS]
print('##### student test per ratio x variant   [val-selected dropout, wd 5e-4, repeat 5]')
print(b.pivot_table(index='ratio', columns='variant', values='cell', aggfunc='first')[order].to_string())
for m, title in [('t_test', 'KERNEL teacher test accuracy after C&S (none = before)'), ('t_val', 'kernel teacher val accuracy after C&S'),
                 ('moved', 'nodes whose argmax the Correct step changed [%]'), ('agree', 'cell argmax agreement [%]'),
                 ('kl_g', 'group KL(true||q)'), ('h_q', 'H(q)')]:
    print(f'\n##### {title}')
    t = b.pivot_table(index='ratio', columns='variant', values=m, aggfunc='first')
    print(t[[c for c in order if c in t.columns]].round(3).to_string())
print('\n##### OOF teacher accuracy on the training nodes / mean L1 residual (C&S rows)')
print(b[b.oof_acc.notna()].groupby('ratio')[['oof_acc', 'resid']].first().to_string())
