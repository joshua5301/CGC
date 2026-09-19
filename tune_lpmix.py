# ============ Cell 1: common (single session; cora) =============
# GIFT-style label mixing with k-hop propagated hard labels: P_t <- (1-g) f_t + g LP_t, LP = A_hat^k Y0 (Y0 = training
# one-hots, zero elsewhere; --lp_alpha 1 removes the restart term), applied to the kernel teacher (H = A^2 X) on both the
# KL refinement and the labels (--refine_teacher kernel). norm: LP rows on the simplex (no-mass rows keep the teacher);
# raw: row mass = confidence. T fixed at 1.
#   stage 1  teacher only: LP alone and the mixed teacher's val/test for k x mode x g (cheap, --teacher_only)
#   stage 2  students at cora 1.3 / 2.6 / 5.2 % for k {1, 2, 3} x mode {norm, raw} x g {0.1, 0.3, 0.5} (+ none), repeat 5
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'lpmix2'
LOG = f'{LOGDIR}/{TAG}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c lp_mode /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--refine_teacher kernel --label_kernel relu1 --conv_depth 2 --feat_norm 0 --no_hyperpara 1 --lr 0.01 --epoch 1000 "
        "--eval_every 10 --dropout 0.5 --weight_decay 5e-4 --dataset_name cora")
DENS = [(0.013, 0.1, 1.0), (0.026, 1.0, 5.0), (0.052, 0.01, 2.0)]     # (ratio, gamma, mu) of the cora1 val-best condensations
GS, MODES, KS = [0.1, 0.3, 0.5], ['norm', 'raw'], [1, 2, 3]
DOWN = '0,0.1,0.3,0.5,0.7,0.9;5e-4'
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_P = re.compile(r'teacher_prop\[(before|refine)\]: .*?val ([\d.]+)%  test ([\d.]+)%  H ([\d.]+)')
PAT_L = re.compile(r'lp_mix: label propagation alone .*?val ([\d.]+)%  test ([\d.]+)%  rows without mass ([\d.]+)%')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+) .*argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)')

def load():
    return pd.DataFrame([json.loads(l) for l in open(LOG)]) if os.path.exists(LOG) else pd.DataFrame()

def done(**key):
    df = load()
    if not len(df):
        return False
    m = pd.Series(True, index=df.index)
    for k, v in key.items():
        m &= (df[k] == v) if k in df.columns else False
    return bool(m.any())

def flags(g, mode, k):
    return f'--lp_mix {g} --lp_mode {mode} --lp_alpha 1.0 --lp_iters {k}' if g > 0 else ''

def teacher(gamma, g, mode, k):
    cmd = f"python main.py {BASE} --ratio 0.026 --gamma {gamma} --bregman 1 {flags(g, mode, k)} --teacher_only 1"   # bregman > 0: the refinement-teacher block (where lp_mix / teacher_only live) must run
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    pr = dict((m[1], (float(m[2]), float(m[3]), float(m[4]))) for m in PAT_P.finditer(out))
    lp = PAT_L.search(out)
    if 'refine' not in pr:
        print('FAIL teacher', gamma, g, mode, k, '\n', out[-1500:]); return
    rec = dict(stage='teacher', gamma=gamma, g=g, mode=mode, k=k, t_val=pr['refine'][0], t_test=pr['refine'][1], h_t=pr['refine'][2],
               lp_val=float(lp[1]) if lp else None, lp_test=float(lp[2]) if lp else None, lp_nomass=float(lp[3]) if lp else None)
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(f"  [teacher] gamma={gamma:<5g} k={k} g={g:<4g} {mode:4s}  val {rec['t_val']:.2f} test {rec['t_test']:.2f} H {rec['h_t']:.3f}"
          + (f"   (LP alone val {lp[1]} test {lp[2]}, no-mass rows {lp[3]}%)" if lp else ''))

def student(ratio, gamma, mu, g, mode, k):
    cmd = (f"python main.py {BASE} --ratio {ratio} --gamma {gamma} --bregman {mu} --teacher_temp 1 {flags(g, mode, k)} "
           f"--repeat {REPEAT} --down_grid '{DOWN}'")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    rows = PAT_D.findall(out); gd = PAT_G.search(out)
    if not rows or not gd:
        print('FAIL', ratio, g, mode, k, '\n', out[-2000:]); return
    pr = dict((m[1], (float(m[2]), float(m[3]), float(m[4]))) for m in PAT_P.finditer(out))
    ex = re.search(r'expert: train ([\d.]+)%\s+val ([\d.]+)%\s+test ([\d.]+)', out)
    ta = pr.get('refine', (float(ex[2]), float(ex[3]), float('nan')) if ex else (None, None, None))
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sdv, va in rows:
            f.write(json.dumps(dict(stage='student', ratio=ratio, gamma=gamma, mu=mu, g=g, mode=mode, k=k,
                                    drop=float(do_), wd=float(wd_), repeat=REPEAT, test=float(te), std=float(sdv), val=float(va),
                                    t_val=ta[0], t_test=ta[1], h_t=ta[2], kl_g=float(gd[1]), agree=float(gd[2]), h_q=float(gd[4]))) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"cora {ratio:<6g} k={k} g={g:<4g} {mode:4s} teacher val/test {ta[0]}/{ta[1]}  agree {gd[2]}%  H(q) {gd[4]}  "
          f"val {best[5]} (do={best[0]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

print('lpmix2: GIFT-style mixing with k-hop propagated hard labels, cora (T fixed 1)')

# ============ Cell 2: teacher stage (LP alone + mixed teacher accuracy) =============
for (ratio, gamma, mu) in DENS:
    for g, mode, k in [(0.0, 'norm', 0)] + list(itertools.product(GS + [1.0], MODES, KS)):
        if not done(stage='teacher', gamma=gamma, g=g, mode=mode, k=k):
            teacher(gamma, g, mode, k)
t = load(); assert len(t) and 'stage' in t.columns, 'no teacher rows logged (see FAIL lines)'; t = t[t.stage == 'teacher']
print('##### mixed teacher TEST accuracy per (gamma, k, mode) x g   (g 0 = teacher alone, g 1 = k-hop LP alone)')
print(t.pivot_table(index=['gamma', 'k', 'mode'], columns='g', values='t_test').round(2).to_string())
print('##### mixed teacher VAL accuracy')
print(t.pivot_table(index=['gamma', 'k', 'mode'], columns='g', values='t_val').round(2).to_string())
print('##### rows without LP mass [%] per k'); print(t[t.g > 0].groupby('k').lp_nomass.first().to_string())

# ============ Cell 3: students =============
for (ratio, gamma, mu) in DENS:
    if not done(stage='student', ratio=ratio, g=0.0):
        student(ratio, gamma, mu, 0.0, 'norm', 0)
    for k, mode, g in itertools.product(KS, MODES, GS):
        if not done(stage='student', ratio=ratio, g=g, mode=mode, k=k):
            student(ratio, gamma, mu, g, mode, k)

# ============ Cell 4: tables =============
df = load(); st = df[df.stage == 'student'].copy()
st['variant'] = st.apply(lambda x: 'none' if x.g == 0 else f"k{int(x.k)} {x['mode']} g{x.g:g}", axis=1)
order = ['none'] + [f"k{k} {m} g{g:g}" for k in KS for m in MODES for g in GS]
b = st.sort_values('val', ascending=False).groupby(['ratio', 'variant']).head(1)
b = b.assign(cell=b.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1))
print('##### student test per variant x ratio   [T 1, val-selected dropout, wd 5e-4, repeat 5]')
print(b.pivot_table(index='ratio', columns='variant', values='cell', aggfunc='first').reindex(columns=order).T.to_string())
print('##### val-selected over (k, mode, g) per ratio  vs  none')
for r, gr in b.groupby('ratio'):
    top = gr.sort_values('val', ascending=False).iloc[0]; base = gr[gr.variant == 'none']
    print(f"  {r:g}: best-val {top.variant}  val {top.val:.2f}  test {top.test:.2f}+-{top['std']:.2f}   | none test {base.test.iloc[0]:.2f}   | oracle {gr.test.max():.2f} ({gr.loc[gr.test.idxmax(), 'variant']})")
for m, title in [('t_test', 'mixed teacher test accuracy'), ('agree', 'cell argmax agreement [%]'), ('h_q', 'H(q)'), ('kl_g', 'group KL(true||q)')]:
    print('##### ' + title)
    print(b.pivot_table(index='ratio', columns='variant', values=m, aggfunc='first').reindex(columns=order).round(3).T.to_string())
