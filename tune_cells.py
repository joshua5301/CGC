# ============ Cell 1: common (identical in all three sessions except SESSION) =============
SESSION = 'A'          # 'A' / 'B' / 'C'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'full'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --epoch 1000 --weight_decay 5e-4")
RATIOS = {'cora': [0.013, 0.026, 0.052], 'citeseer': [0.009, 0.018, 0.036],
          'arxiv': [0.0005, 0.0025, 0.005], 'flickr': [0.001, 0.005, 0.01],
          'reddit': [0.0005, 0.001, 0.002]}
KERNELS = ['erf', 'relu1', 'relu2']
BASES   = [0, 1000, 3000]            # 0 = landmarks
GAMMAS  = [1e-4, 1e-3, 1e-2]
DROPS   = [0.0, 0.5]
REPEAT  = 2
PAT = re.compile(r'== gcn: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')

def run(ds, r, kernel, basis, gamma, drop, repeat, tag=''):
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} "
           f"--expert_basis {basis} --gamma {gamma} --dropout {drop} --repeat {repeat}")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC').stdout
    m = PAT.search(out)
    if not m:
        print('FAIL', ds, r, kernel, basis, gamma, drop, '\n', out[-1200:]); return None
    rec = dict(ds=ds, ratio=r, kernel=kernel, basis=basis, gamma=gamma, drop=drop, repeat=repeat, tag=tag,
               test=float(m[1]), std=float(m[2]), val=float(m[3]), sec=round(time.time() - t))
    ex = re.search(r'expert:.*', out); rec['expert'] = ex[0] if ex else ''
    with open(LOG, 'a') as f: f.write(json.dumps(rec) + '\n')
    print(f"{ds:8s} r={r:<7g} {kernel:5s} b={basis:<5} g={gamma:<6g} do={drop}  val {rec['val']:.2f}  "
          f"test {rec['test']:.2f}±{rec['std']:.2f}  ({rec['sec']}s)  {rec['expert'][:60]}")
    return rec

def load():
    recs = [json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)]
    return pd.DataFrame(recs)

def done(ds, r, kernel, basis, gamma, drop, tag):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.kernel == kernel) & (df.basis == basis)
                            & (df.gamma == gamma) & (df['drop'] == drop) & (df.tag == tag)).any()

def grid(ds, repeat=REPEAT):
    for r in RATIOS[ds]:
        for kernel, basis, gamma, drop in itertools.product(KERNELS, BASES, GAMMAS, DROPS):
            if not done(ds, r, kernel, basis, gamma, drop, 'grid'):
                run(ds, r, kernel, basis, gamma, drop, repeat, 'grid')

# ============ Cell 2: grid — keep only your session's line (54 runs per density) =============
# session A  (~3 h)
grid('arxiv')
# session B  (~3 h)
# grid('reddit')
# session C  (~2.5 h)
# grid('flickr'); grid('cora'); grid('citeseer')

# ============ Cell 3: best by val -> repeat 10 (after all grids finish) =============
def best_by_val(ds):
    df = load()
    if not len(df): return df
    df = df[(df.ds == ds) & (df.tag == 'grid')]
    return df.sort_values('val', ascending=False).groupby('ratio').head(1)

MINE = {'A': ['arxiv'], 'B': ['reddit'], 'C': ['flickr', 'cora', 'citeseer']}[SESSION]
for ds in MINE:
    for _, row in best_by_val(ds).iterrows():
        if not done(ds, row.ratio, row.kernel, int(row.basis), row.gamma, row['drop'], 'final'):
            run(ds, row.ratio, row.kernel, int(row.basis), row.gamma, row['drop'], 10, 'final')

# ============ Cell 4: tables (any session) =============
df = load(); assert len(df), 'no logs found in Drive'
g = df[df.tag == 'grid']
pd.set_option('display.width', 200)
for ds in RATIOS:
    sub = g[g.ds == ds]
    if not len(sub): continue
    print(f'\n##### {ds}: best by val per density')
    print(best_by_val(ds)[['ratio', 'kernel', 'basis', 'gamma', 'drop', 'val', 'test', 'std']].to_string(index=False))
    for ax in ['kernel', 'basis', 'gamma', 'drop']:
        print(f'--- {ds}: best val over the other axes, by ratio x {ax}')
        print(sub.groupby(['ratio', ax]).val.max().unstack().round(2).to_string())
fin = df[df.tag == 'final'].copy()
if len(fin):
    fin['cell'] = fin.apply(lambda x: f"{x['test']:.1f}±{x['std']:.1f}", axis=1)
    fin['cfg'] = fin.apply(lambda x: f"{x['kernel']} b={x['basis']:g} γ={x['gamma']:g} do={x['drop']:g}", axis=1)
    print('\n##### FINAL (repeat 10)')
    print(fin.pivot(index='ds', columns='ratio', values='cell').to_string())
    print(fin.pivot(index='ds', columns='ratio', values='cfg').to_string())
