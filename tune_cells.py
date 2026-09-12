# ============ Cell 1: common (identical in all three sessions except SESSION) =============
SESSION = 'A'          # 'A' / 'B' / 'C'
import subprocess, re, json, os, time, glob
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
LOG = f'{LOGDIR}/kernel_{SESSION}.jsonl'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --head ce "
        "--label_mode kernel_mean --label_kernel erf")
RATIOS = {'cora': [0.013, 0.026, 0.052], 'citeseer': [0.009, 0.018, 0.036],
          'arxiv': [0.0005, 0.0025, 0.005], 'flickr': [0.001, 0.005, 0.01],
          'reddit': [0.0005, 0.001, 0.002]}
EXTRA = {'cora': '--weight_decay 5e-4 --expert_basis 0', 'citeseer': '--weight_decay 5e-4 --expert_basis 0',
         'arxiv': '--expert_basis 1000', 'flickr': '--expert_basis 1000', 'reddit': '--expert_basis 1000'}
GAMMAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]           # the single hyperparameter
PAT = re.compile(r'== gcn: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')

def run(ds, r, gamma, repeat, tag=''):
    cmd = (f"python main.py {BASE} {EXTRA[ds]} --dataset_name {ds} --ratio {r} "
           f"--gamma {gamma} --repeat {repeat}")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC').stdout
    m = PAT.search(out)
    if not m:
        print('FAIL', ds, r, gamma, '\n', out[-1500:]); return None
    rec = dict(ds=ds, ratio=r, gamma=gamma, repeat=repeat, tag=tag,
               test=float(m[1]), std=float(m[2]), val=float(m[3]), sec=round(time.time() - t))
    ex = re.search(r'expert:.*', out); rec['expert'] = ex[0] if ex else ''
    with open(LOG, 'a') as f: f.write(json.dumps(rec) + '\n')
    print(f"{ds:9s} r={r:<7g} gamma={gamma:<6g}  val {rec['val']:.2f}  "
          f"test {rec['test']:.2f}±{rec['std']:.2f}  ({rec['sec']}s)  {rec['expert']}")
    return rec

def load():
    recs = [json.loads(l) for f in glob.glob(f'{LOGDIR}/kernel_*.jsonl') for l in open(f)]
    return pd.DataFrame(recs)

def done(ds, r, gamma, tag):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.gamma == gamma) & (df.tag == tag)).any()

def grid(ds, repeat=3):
    for r in RATIOS[ds]:
        for gamma in GAMMAS:
            if not done(ds, r, gamma, 'grid'):
                run(ds, r, gamma, repeat, 'grid')

# ============ Cell 2: grid — keep only your session's line =============
# session A
grid('arxiv')
# session B
# grid('reddit')
# session C
# grid('flickr'); grid('cora'); grid('citeseer')

# ============ Cell 3: pick by val, rerun with repeat 10 (after all grids finish) =============
def best_by_val(ds):
    df = load(); df = df[(df.ds == ds) & (df.tag == 'grid')]
    return df.sort_values('val', ascending=False).groupby('ratio').head(1)

MINE = {'A': ['arxiv'], 'B': ['reddit'], 'C': ['flickr', 'cora', 'citeseer']}[SESSION]
for ds in MINE:
    for _, row in best_by_val(ds).iterrows():
        if not done(ds, row.ratio, row.gamma, 'final'):
            run(ds, row.ratio, row.gamma, 10, 'final')

# ============ Cell 4: main table (any session) =============
df = load(); fin = df[df.tag == 'final'].copy()
fin['cell'] = fin.apply(lambda x: f"{x['test']:.1f}±{x['std']:.1f}", axis=1)
print(fin.pivot(index='ds', columns='ratio', values='cell').to_string())
print(fin.pivot(index='ds', columns='ratio', values='gamma').to_string())
print(fin[['ds', 'ratio', 'gamma', 'val', 'test', 'std']].to_string(index=False))
# full grid view: val per (ds, ratio, gamma)
g = df[df.tag == 'grid']
print(g.pivot_table(index=['ds', 'ratio'], columns='gamma', values='val').round(2).to_string())
print(g.pivot_table(index=['ds', 'ratio'], columns='gamma', values='test').round(2).to_string())
