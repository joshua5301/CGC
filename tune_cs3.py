# ============ Cell 1: common (single session; cora) =============
# Naive Correct & Smooth (Huang et al. 2021) done properly: the base predictor is graph-blind (kernel teacher on the raw
# features X, --label_feat first) and every choice is made on the TEACHER'S validation accuracy, without students:
#   stage 1  X teacher: kernel {erf, relu1} x gamma {0.01, 0.1, 1, 10, 30} x fn {0, 1}                     (20 teacher fits, --teacher_only)
#   stage 2  on the val-best X teacher: Smooth a2 {0.5, 0.8, 0.9}; then at the best a2, Correct a1 {0.5, 0.8, 1.0}
#            x scale {auto, 0.3, 1.0} (OOF residuals) + in-sample residuals                                  (13 teacher fits)
#   stage 3  students at cora 1.3 / 2.6 / 5.2 % with {X teacher, + best Smooth, + best C&S}, clustering on H = A^2 X, repeat 5
# Reference (cs2, H teacher, same condensation): teacher 82.4 / 82.1 / 82.1, student 83.9 / 84.6 / 83.5.
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'cs3'
LOG = f'{LOGDIR}/{TAG}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c teacher_only /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--refine_teacher kernel --label_feat first --conv_depth 2 --bregman 1 --teacher_temp 1 --no_hyperpara 1 "
        "--lr 0.01 --epoch 1000 --eval_every 10 --dropout 0.5 --weight_decay 5e-4 --dataset_name cora --ratio 0.026")
DENS = [(0.013, 1.0), (0.026, 5.0), (0.052, 2.0)]        # (ratio, mu) of the cora1 val-best condensations
KERNELS, GAMMAS, FNS = ['erf', 'relu1'], [0.01, 0.1, 1.0, 10.0, 30.0], [0, 1]
A2S, A1S, SCALES = [0.5, 0.8, 0.9], [0.5, 0.8, 1.0], ['auto', '0.3', '1.0']
DOWN = '0,0.1,0.3,0.5,0.7,0.9;5e-4'
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_P = re.compile(r'teacher_prop\[(before|refine)\]: .*?val ([\d.]+)%  test ([\d.]+)%  H ([\d.]+)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+) .*argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)')
PAT_O = re.compile(r'C&S: \d+-fold OOF teacher acc on train ([\d.]+)%  mean \|resid\|_1 ([\d.]+)')
PAT_M = re.compile(r'C&S correct: argmax changed on ([\d.]+)% of nodes')

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

def teacher(kernel, gamma, fn, cs='', tag='stage1'):
    """--teacher_only run: returns (val, test, H) of the (propagated) teacher; logs it"""
    cmd = f"python main.py {BASE} --label_kernel {kernel} --gamma {gamma} --feat_norm {fn} {cs} --teacher_only 1"
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC')
    out = p.stdout + p.stderr
    pr = dict((m[1], (float(m[2]), float(m[3]), float(m[4]))) for m in PAT_P.finditer(out))
    if 'refine' not in pr:
        print('FAIL', kernel, gamma, fn, cs, '\n', out[-1500:]); return None
    oo, mv = PAT_O.search(out), PAT_M.search(out)
    rec = dict(stage=tag, kernel=kernel, gamma=gamma, fn=fn, cs=cs, t_val=pr['refine'][0], t_test=pr['refine'][1], h_t=pr['refine'][2],
               t_val0=pr.get('before', pr['refine'])[0], t_test0=pr.get('before', pr['refine'])[1],
               oof_acc=float(oo[1]) if oo else None, moved=float(mv[1]) if mv else None)
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(f"  [{tag}] {kernel:5s} g={gamma:<5g} fn={fn} {cs:60s} teacher val {rec['t_val']:.2f} test {rec['t_test']:.2f}"
          + (f"  (before {rec['t_val0']:.2f}/{rec['t_test0']:.2f}, moved {rec['moved']}%)" if cs else ''))
    return rec

def student(ratio, mu, kernel, gamma, fn, cs, name):
    cmd = (f"python main.py {BASE} --ratio {ratio} --bregman {mu} --label_kernel {kernel} --gamma {gamma} --feat_norm {fn} {cs} "
           f"--repeat {REPEAT} --down_grid '{DOWN}'")
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC')
    out = p.stdout + p.stderr
    rows = PAT_D.findall(out); g = PAT_G.search(out)
    if not rows or not g:
        print('FAIL', ratio, name, '\n', out[-2000:]); return
    pr = dict((m[1], (float(m[2]), float(m[3]), float(m[4]))) for m in PAT_P.finditer(out))
    ta = pr.get('refine', (None, None, None))
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sdv, va in rows:
            f.write(json.dumps(dict(stage='student', ratio=ratio, mu=mu, kernel=kernel, gamma=gamma, fn=fn, cs=cs, variant=name,
                                    drop=float(do_), wd=float(wd_), repeat=REPEAT, test=float(te), std=float(sdv), val=float(va),
                                    t_val=ta[0], t_test=ta[1], h_t=ta[2], kl_g=float(g[1]), agree=float(g[2]), h_q=float(g[4]))) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"cora {ratio:<6g} {name:12s} teacher val/test {ta[0]}/{ta[1]}  agree {g[2]}%  H(q) {g[4]}  "
          f"val {best[5]} (do={best[0]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

print('cs3: naive C&S on cora, teacher-val selection')

# ============ Cell 2: stage 1 - X teacher selection (kernel x gamma x fn) =============
for kernel, gamma, fn in itertools.product(KERNELS, GAMMAS, FNS):
    if not done(stage='stage1', kernel=kernel, gamma=gamma, fn=fn):
        teacher(kernel, gamma, fn)
s1 = load(); s1 = s1[s1.stage == 'stage1']
print(s1.pivot_table(index=['kernel', 'fn'], columns='gamma', values='t_val').round(2).to_string())
bt = s1.sort_values('t_val', ascending=False).iloc[0]
K, G, FN = bt.kernel, float(bt.gamma), int(bt.fn)
print(f"--> X teacher: {K} gamma {G:g} fn {FN}   val {bt.t_val:.2f} test {bt.t_test:.2f}")

# ============ Cell 3: stage 2 - Smooth a2, then Correct a1 x scale, on the teacher's val =============
for a2 in A2S:
    cs = f'--tcs_smooth_iters 50 --tcs_correct 0 --tcs_alpha2 {a2}'
    if not done(stage='smooth', cs=cs):
        teacher(K, G, FN, cs, 'smooth')
sm = load(); sm = sm[sm.stage == 'smooth'].sort_values('t_val', ascending=False).iloc[0]
A2 = float(re.search(r'--tcs_alpha2 ([\d.]+)', sm.cs)[1])
print(f"--> Smooth: a2 {A2:g}  teacher val {sm.t_val:.2f} test {sm.t_test:.2f}")
for a1, sc in itertools.product(A1S, SCALES):
    cs = f'--tcs_smooth_iters 50 --tcs_correct 1 --tcs_resid oof --tcs_alpha1 {a1} --tcs_scale {sc} --tcs_alpha2 {A2}'
    if not done(stage='correct', cs=cs):
        teacher(K, G, FN, cs, 'correct')
cs = f'--tcs_smooth_iters 50 --tcs_correct 1 --tcs_resid insample --tcs_alpha1 0.8 --tcs_scale auto --tcs_alpha2 {A2}'
if not done(stage='correct', cs=cs):
    teacher(K, G, FN, cs, 'correct')
co = load(); co = co[co.stage == 'correct']
print(co[['cs', 't_val', 't_test', 'oof_acc', 'moved']].sort_values('t_val', ascending=False).to_string(index=False))
cb = co.sort_values('t_val', ascending=False).iloc[0]
print(f"--> C&S: {cb.cs}  teacher val {cb.t_val:.2f} test {cb.t_test:.2f}")

# ============ Cell 4: stage 3 - students with {X teacher, + Smooth, + C&S} at three densities =============
VAR = [('X teacher', ''), ('X + smooth', sm.cs), ('X + C&S', cb.cs)]
for (ratio, mu), (name, cs) in itertools.product(DENS, VAR):
    if not done(stage='student', ratio=ratio, variant=name):
        student(ratio, mu, K, G, FN, cs, name)

# ============ Cell 5: tables =============
df = load()
b = df[df.stage == 'student'].sort_values('val', ascending=False).groupby(['ratio', 'variant']).head(1)
b = b.assign(cell=b.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1))
order = [v[0] for v in VAR]
print(f'##### X teacher = {K} gamma {G:g} fn {FN};  Smooth = {sm.cs};  C&S = {cb.cs}')
print('\n##### student test per ratio x variant   [val-selected dropout, wd 5e-4, repeat 5]   (H-teacher reference: 83.9 / 84.6 / 83.5)')
print(b.pivot_table(index='ratio', columns='variant', values='cell', aggfunc='first')[order].to_string())
for m, title in [('t_test', 'teacher test accuracy (H-teacher reference 82.4 / 82.1 / 82.1)'), ('t_val', 'teacher val accuracy'),
                 ('agree', 'cell argmax agreement [%]'), ('h_q', 'H(q)')]:
    print(f'\n##### {title}')
    print(b.pivot_table(index='ratio', columns='variant', values=m, aggfunc='first')[order].round(3).to_string())
