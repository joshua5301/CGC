# ============ Cell 1 (single session) =============
# T = 0.25 was the lower edge of every grid and was selected at citeseer 0.9 / 1.8 / 3.6 % and arxiv 0.05 %. Extend the
# temperature downwards at the protocol-A val-best condensation of those cells: T {0.25, 0.15, 0.1, 0.05, 0.02} (0.02 ~ argmax).
# Recipe on val, repeat 5. If a T < 0.25 wins on val, the grid becomes {1, 0.5, 0.25, 0.1} in the final reselection.
import subprocess, re, json, os, time, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'lowT1'
LOG = f'{LOGDIR}/{TAG}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--conv_depth 2 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 --dropout 0.5 --weight_decay 5e-4")
#         ds          ratio   kernel   gamma  fn  mu    dropouts                wds
CELLS = [('citeseer', 0.009,  'erf',   10.0, 1, 0.2, '0,0.1,0.3,0.5,0.7,0.9', '5e-4,5e-3'),
         ('citeseer', 0.018,  'erf',   10.0, 1, 0.2, '0,0.1,0.3,0.5,0.7,0.9', '5e-4,5e-3'),
         ('citeseer', 0.036,  'erf',   3.0,  1, 0.2, '0,0.1,0.3,0.5,0.7,0.9', '5e-4,5e-3'),
         ('arxiv',    0.0005, 'erf',   1e-3, 0, 0.5, '0,0.1,0.3,0.5,0.7',     '5e-4')]
TEMPS = [0.25, 0.15, 0.1, 0.05, 0.02]
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+) .*argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)')

def load():
    return pd.DataFrame([json.loads(l) for l in open(LOG)]) if os.path.exists(LOG) else pd.DataFrame()

def done(ds, r, temp):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.temp == temp)).any()

def run(ds, r, kernel, gamma, fn, mu, dos, wds, temp):
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --gamma {gamma} --feat_norm {fn} "
           f"--bregman {mu} --teacher_temp {temp} --repeat {REPEAT} --down_grid '{dos};{wds}'")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    rows = PAT_D.findall(out); g = PAT_G.search(out)
    if not rows:
        print('FAIL', ds, r, temp, '\n', out[-2000:]); return
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, kernel=kernel, gamma=gamma, fn=fn, mu=mu, temp=temp,
                                    drop=float(do_), wd=float(wd_), repeat=REPEAT, test=float(te), std=float(sd), val=float(va),
                                    kl_g=float(g[1]) if g else None, agree=float(g[2]) if g else None, h_q=float(g[4]) if g else None)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"{ds:8s} {r:<6g} T={temp:<5g}" + (f"  agree {g[2]}%  KL {g[1]}  H(q) {g[4]}" if g else '') +
          f"  val {best[5]} (do={best[0]} wd={best[1]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

# ============ Cell 2: run =============
for ds, r, kernel, gamma, fn, mu, dos, wds in CELLS:
    for temp in TEMPS:
        if not done(ds, r, temp):
            run(ds, r, kernel, gamma, fn, mu, dos, wds, temp)

# ============ Cell 3: table =============
df = load()
b = df.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'temp']).head(1)
b = b.assign(cell=b.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f} (val {x['val']:.1f})", axis=1))
print('##### student test per (ds, ratio) x T   [val-selected dropout/wd, repeat 5]   (T 0.25 = current grid edge)')
print(b.pivot_table(index=['ds', 'ratio'], columns='temp', values='cell', aggfunc='first').to_string())
print('\n##### T selected on val per cell')
for (ds, r), g in b.groupby(['ds', 'ratio']):
    top = g.sort_values('val', ascending=False).iloc[0]; e = g[g.temp == 0.25].iloc[0]
    print(f"  {ds} {r:g}: best-val T {top.temp:g}  test {top.test:.2f}   | T 0.25: {e.test:.2f}   | oracle {g.test.max():.2f} (T {g.loc[g.test.idxmax(), 'temp']:g})")
print('\n##### H(q) / agreement per T')
print(b.pivot_table(index=['ds', 'ratio'], columns='temp', values=['h_q', 'agree'], aggfunc='first').round(3).to_string())
