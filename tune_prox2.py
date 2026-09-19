# ============ Cell 1: common (citeseer 3.6%; SESSION = 'A' mu 0.2 | 'B' mu 0.5 | 'C' mu 1) =============
# Full sweep of the proximity weights (--prox_tau, no floor) jointly with the axes they could interact with:
#   gamma (teacher accuracy / direction) x mu (KL term) x tau (proximity weight) x T (label scale)
#   = {1, 3, 10, 30} x {0.2, 0.5, 1} x {0, 2, 1, 0.5} x {2, 1, 0.5, 0.25}   = 192 condensations, 64 per session (one mu each)
# fixed: erf, fn 1, raw, basis 3000, depth 2, lr 0.01; dropout {0,...,.9} x wd {5e-4, 5e-3} on val, repeat 3.
# tau 0 rows = the plain cell mean, so "does opening tau raise the val-best" is read inside one grid.
SESSION = 'A'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'prox2'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run("grep -c \"default=0.0, help='floor of the proximity weight\" /content/CGC/scr/para.py", shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale (prox floor)'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--label_kernel erf --feat_norm 1 --conv_depth 2 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 "
        "--dropout 0.5 --weight_decay 5e-4 --dataset_name citeseer --ratio 0.036")
GAMMAS = [1.0, 3.0, 10.0, 30.0]
MUS = {'A': [0.2], 'B': [0.5], 'C': [1.0]}[SESSION]
TAUS, TEMPS = [0.0, 2.0, 1.0, 0.5], [2.0, 1.0, 0.5, 0.25]
DOWN = '0,0.1,0.3,0.5,0.7,0.9;5e-4,5e-3'
REPEAT = 3
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+) .*argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)')
PAT_W = re.compile(r'prox weights: .*?w mean ([\d.]+) .*?effective sample ([\d.]+)%')

def load():
    return pd.DataFrame([json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)])

def done(gamma, mu, tau, temp):
    df = load()
    return len(df) > 0 and ((df.gamma == gamma) & (df.mu == mu) & (df.tau == tau) & (df.temp == temp)).any()

def run(gamma, mu, tau, temp):
    cmd = (f"python main.py {BASE} --gamma {gamma} --bregman {mu} --teacher_temp {temp} --prox_tau {tau} "
           f"--repeat {REPEAT} --down_grid '{DOWN}'")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    rows = PAT_D.findall(out); g = PAT_G.search(out); w = PAT_W.search(out)
    if not rows or not g:
        print('FAIL', gamma, mu, tau, temp, '\n', out[-2000:]); return
    ex = re.search(r'expert: train ([\d.]+)%\s+val ([\d.]+)%\s+test ([\d.]+)', out)
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds='citeseer', ratio=0.036, gamma=gamma, mu=mu, tau=tau, temp=temp,
                                    drop=float(do_), wd=float(wd_), repeat=REPEAT, test=float(te), std=float(sd), val=float(va),
                                    t_test=float(ex[3]) if ex else None, kl_g=float(g[1]), agree=float(g[2]), h_true=float(g[3]), h_q=float(g[4]),
                                    eff=float(w[2]) if w else 100.0)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"citeseer 3.6% g={gamma:<3g} mu={mu:<3g} tau={tau:<3g} T={temp:<4g}" + (f"  eff {w[2]}%" if w else '') +
          f"  agree {g[2]}%  KL {g[1]}  H(q) {g[4]}  val {best[5]} (do={best[0]} wd={best[1]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

print(f'prox2 {SESSION}: citeseer 3.6%  gamma {GAMMAS} x mu {MUS} x tau {TAUS} x T {TEMPS}')

# ============ Cell 2: run (done() resumes) =============
for gamma, mu, tau, temp in itertools.product(GAMMAS, MUS, TAUS, TEMPS):
    if not done(gamma, mu, tau, temp):
        run(gamma, mu, tau, temp)

# ============ Cell 3: tables (run where all prox2_*.jsonl are present) =============
df = load()
one = df.sort_values('val', ascending=False).drop_duplicates(['gamma', 'mu', 'tau', 'temp'])     # val-best recipe per config
print('##### val-selected test per (gamma, mu) x tau, T selected on val   [repeat 3]')
bt = one.sort_values('val', ascending=False).groupby(['gamma', 'mu', 'tau']).head(1)
print(bt.assign(cell=bt.apply(lambda x: f"{x['test']:.1f} (T{x['temp']:g})", axis=1)).pivot_table(index=['gamma', 'mu'], columns='tau', values='cell', aggfunc='first').to_string())
print('\n##### val-selected test per (gamma, T) x tau, mu selected on val')
bt = one.sort_values('val', ascending=False).groupby(['gamma', 'temp', 'tau']).head(1)
print(bt.pivot_table(index=['gamma', 'temp'], columns='tau', values='test').round(2).to_string())
print('\n##### marginal: val-best over everything else, per tau')
for tau, g in one.groupby('tau'):
    top = g.sort_values('val', ascending=False).iloc[0]
    print(f"  tau {tau:<3g}: val {top.val:.2f}  test {top.test:.2f}+-{top['std']:.2f}  [g{top.gamma:g} mu{top.mu:g} T{top.temp:g} do{top['drop']:g} wd{top.wd:g}]   oracle over grid {g.test.max():.2f}")
top = one.sort_values('val', ascending=False).iloc[0]; base = one[one.tau == 0].sort_values('val', ascending=False).iloc[0]
print(f"\n##### overall val-best: tau {top.tau:g} g{top.gamma:g} mu{top.mu:g} T{top.temp:g}  test {top.test:.2f}   | tau 0 val-best: test {base.test:.2f}   | Spearman(val, test) {one.val.corr(one.test, method='spearman'):+.2f}")
for m, title in [('agree', 'cell argmax agreement [%]'), ('kl_g', 'group KL(true||q)'), ('h_q', 'H(q)')]:
    print(f'\n##### {title}   [T 1, mu 0.2]')
    print(one[(one.temp == 1.0) & (one.mu == 0.2)].pivot_table(index='gamma', columns='tau', values=m).round(3).to_string())
