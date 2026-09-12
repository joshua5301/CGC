# ============ Cell 1: common (identical in all three sessions except SESSION) =============
SESSION = 'C'          # 'A' / 'B' / 'C'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'stage'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 220)

# condensation: kernel teacher (rkhs prior), basis fixed at min(N, 3000), gamma swept
# downstream : 2-layer GCN-256, Adam lr 0.01, 1000 epochs, best-val epoch; dropout / wd swept via --down_grid
BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --expert_basis 3000 "
        "--no_hyperpara 1 --lr 0.01 --epoch 1000 --dropout 0.5 --weight_decay 5e-4")
RATIOS = {'cora': [0.013, 0.026, 0.052], 'citeseer': [0.009, 0.018, 0.036],
          'arxiv': [0.0005, 0.0025, 0.005], 'flickr': [0.001, 0.005, 0.01],
          'reddit': [0.0005, 0.001, 0.002]}
MINE = {'A': ['arxiv'], 'B': ['reddit'], 'C': ['cora', 'citeseer', 'flickr']}[SESSION]
KERNELS = ['erf', 'relu1', 'relu2']
GAMMAS  = [1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]
DOWN1   = '0,0.5;0,5e-4'                          # stage 1: interaction check
DOWN2   = '0,0.3,0.5,0.7;0,1e-4,5e-4,2e-3'        # stage 2: downstream recipe
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+): ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')

def load():
    recs = [json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)]
    return pd.DataFrame(recs)

def done(ds, r, kernel, gamma, tag):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.kernel == kernel)
                            & (df.gamma == gamma) & (df.tag == tag)).any()

def run(ds, r, kernel, gamma, down, repeat, tag):
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} "
           f"--gamma {gamma} --repeat {repeat} --down_grid '{down}'")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC').stdout
    rows = PAT_D.findall(out)
    if not rows:
        print('FAIL', ds, r, kernel, gamma, '\n', out[-1200:]); return
    ex = re.search(r'expert:.*', out); ex = ex[0] if ex else ''
    with open(LOG, 'a') as f:
        for do_, wd_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, kernel=kernel, gamma=gamma, drop=float(do_), wd=float(wd_),
                                    repeat=repeat, tag=tag, test=float(te), std=float(sd), val=float(va),
                                    expert=ex)) + '\n')
    best = max(rows, key=lambda x: float(x[4]))
    print(f"{ds:8s} r={r:<7g} {kernel:5s} g={gamma:<6g} [{tag}]  best val {best[4]} "
          f"(do={best[0]} wd={best[1]}) test {best[2]}±{best[3]}  ({round(time.time() - t)}s)  {ex[:50]}")

def grid(ds, repeat=2):
    for r in RATIOS[ds]:
        for kernel, gamma in itertools.product(KERNELS, GAMMAS):
            if not done(ds, r, kernel, gamma, 'stage1'):
                run(ds, r, kernel, gamma, DOWN1, repeat, 'stage1')

def best_cond(ds):
    g = load()
    if not len(g): return g
    g = g[(g.ds == ds) & (g.tag == 'stage1')]
    return g.sort_values('val', ascending=False).groupby('ratio').head(1)

# ============ Cell 2: stage 1 — condensation axes under 4 downstream recipes =============
# 18 condensations per density; small graphs ~45 s each, arxiv/reddit ~3 min each
for ds in MINE:
    grid(ds)

# ============ Cell 3: stage 2 — 16 downstream recipes on the val-best condensation per density =============
for ds in MINE:
    for _, row in best_cond(ds).iterrows():
        if not done(ds, row.ratio, row.kernel, row.gamma, 'stage2'):
            run(ds, row.ratio, row.kernel, row.gamma, DOWN2, 3, 'stage2')

# ============ Cell 4: tables (any session) =============
df = load(); assert len(df), 'no logs found in Drive'
s1 = df[df.tag == 'stage1']
for ds in RATIOS:
    sub = s1[s1.ds == ds]
    if not len(sub): continue
    for ax in ['kernel', 'gamma']:
        print(f'\n--- {ds}: stage 1 best val, rows (ratio, drop, wd) x {ax}')
        print(sub.groupby(['ratio', 'drop', 'wd', ax]).val.max().unstack().round(2).to_string())
s2 = df[df.tag == 'stage2']
for ds in RATIOS:
    sub = s2[s2.ds == ds]
    if not len(sub): continue
    print(f'\n##### {ds}: stage 2 val, rows (ratio, drop) x wd')
    print(sub.pivot_table(index=['ratio', 'drop'], columns='wd', values='val').round(2).to_string())
    print(f'##### {ds}: stage 2 test, rows (ratio, drop) x wd')
    print(sub.pivot_table(index=['ratio', 'drop'], columns='wd', values='test').round(2).to_string())
if len(s2):
    fin = s2.sort_values('val', ascending=False).groupby(['ds', 'ratio']).head(1).copy()
    fin['cell'] = fin.apply(lambda x: f"{x['test']:.1f}±{x['std']:.1f}", axis=1)
    fin['cfg'] = fin.apply(lambda x: f"{x['kernel']} γ={x['gamma']:g} do={x['drop']:g} wd={x['wd']:g}", axis=1)
    print('\n##### MAIN TABLE (val-selected, repeat 3)')
    print(fin.pivot(index='ds', columns='ratio', values='cell').to_string())
    print(fin.pivot(index='ds', columns='ratio', values='cfg').to_string())
    gc = s2[(s2['drop'] == 0.5) & (s2.wd == 5e-4)].copy()
    gc['cell'] = gc.apply(lambda x: f"{x['test']:.1f}±{x['std']:.1f}", axis=1)
    print('\n##### GCond fixed recipe (dropout 0.5, wd 5e-4) on the same condensed graphs')
    print(gc.pivot(index='ds', columns='ratio', values='cell').to_string())
