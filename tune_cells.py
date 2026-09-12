# ============ Cell 1: common (identical in all three sessions except SESSION) =============
SESSION = 'A'          # 'A' / 'B' / 'C'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'logistic'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --head ce "
        "--label_mode logistic_mean --label_kernel erf")
RATIOS = {'cora': [0.013, 0.026, 0.052], 'citeseer': [0.009, 0.018, 0.036],
          'arxiv': [0.0005, 0.0025, 0.005], 'flickr': [0.001, 0.005, 0.01],
          'reddit': [0.0005, 0.001, 0.002]}
EXTRA = {'cora': '--weight_decay 5e-4 --expert_basis 0', 'citeseer': '--weight_decay 5e-4 --expert_basis 0',
         'arxiv': '--expert_basis 1000', 'flickr': '--expert_basis 1000', 'reddit': '--expert_basis 1000'}
BETAS  = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]      # kernel ridge
GAMMAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]            # dual-coefficient penalty
REPEAT = 2
PAT = re.compile(r'== gcn: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')

def run(ds, r, beta, gamma, repeat, tag=''):
    cmd = (f"python main.py {BASE} {EXTRA[ds]} --dataset_name {ds} --ratio {r} "
           f"--beta {beta} --gamma {gamma} --repeat {repeat}")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC').stdout
    m = PAT.search(out)
    if not m:
        print('FAIL', ds, r, beta, gamma, '\n', out[-1500:]); return None
    rec = dict(ds=ds, ratio=r, beta=beta, gamma=gamma, repeat=repeat, tag=tag,
               test=float(m[1]), std=float(m[2]), val=float(m[3]), sec=round(time.time() - t))
    ex = re.search(r'expert:.*', out); rec['expert'] = ex[0] if ex else ''
    with open(LOG, 'a') as f: f.write(json.dumps(rec) + '\n')
    print(f"{ds:9s} r={r:<7g} beta={beta:<6g} gamma={gamma:<6g}  val {rec['val']:.2f}  "
          f"test {rec['test']:.2f}±{rec['std']:.2f}  ({rec['sec']}s)  {rec['expert']}")
    return rec

def load():
    recs = [json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)]
    return pd.DataFrame(recs)

def done(ds, r, beta, gamma, tag):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.beta == beta)
                            & (df.gamma == gamma) & (df.tag == tag)).any()

def grid(ds, repeat=REPEAT):
    for r in RATIOS[ds]:
        for beta, gamma in itertools.product(BETAS, GAMMAS):
            if not done(ds, r, beta, gamma, 'grid'):
                run(ds, r, beta, gamma, repeat, 'grid')

# ============ Cell 2: grid — keep only your session's line =============
# session A  (90 runs, ~2 h)
grid('arxiv')
# session B  (90 runs, ~1.6 h)
# grid('reddit')
# session C  (~1.2 h)
# grid('flickr'); grid('cora'); grid('citeseer')

# ============ Cell 3: pick by val, rerun with repeat 10 (after all grids finish) =============
def best_by_val(ds):
    df = load()
    if not len(df): return df
    df = df[(df.ds == ds) & (df.tag == 'grid')]
    return df.sort_values('val', ascending=False).groupby('ratio').head(1)

MINE = {'A': ['arxiv'], 'B': ['reddit'], 'C': ['flickr', 'cora', 'citeseer']}[SESSION]
for ds in MINE:
    for _, row in best_by_val(ds).iterrows():
        if not done(ds, row.ratio, row.beta, row.gamma, 'final'):
            run(ds, row.ratio, row.beta, row.gamma, 10, 'final')

# ============ Cell 4: tables (any session) =============
df = load(); assert len(df), 'no logs found in Drive'
g = df[df.tag == 'grid']
for ds in RATIOS:
    for r in RATIOS[ds]:
        sub = g[(g.ds == ds) & (g.ratio == r)]
        if not len(sub): continue
        print(f'\n### {ds} r={r}  (val)')
        print(sub.pivot_table(index='beta', columns='gamma', values='val').round(2).to_string())
        print(f'### {ds} r={r}  (test)')
        print(sub.pivot_table(index='beta', columns='gamma', values='test').round(2).to_string())
fin = df[df.tag == 'final'].copy()
if len(fin):
    fin['cell'] = fin.apply(lambda x: f"{x['test']:.1f}±{x['std']:.1f}", axis=1)
    fin['cfg'] = fin.apply(lambda x: f"β={x['beta']:g} γ={x['gamma']:g}", axis=1)
    print(fin.pivot(index='ds', columns='ratio', values='cell').to_string())
    print(fin.pivot(index='ds', columns='ratio', values='cfg').to_string())
    print(fin[['ds', 'ratio', 'beta', 'gamma', 'val', 'test', 'std']].to_string(index=False))
