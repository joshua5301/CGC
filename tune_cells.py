# ============ Cell 1: 공통 (세 세션 동일, SESSION 만 다르게) =============
SESSION = 'A'          # 세션마다 'A' / 'B' / 'C'
import subprocess, re, itertools, json, os, time, glob
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
LOG = f'{LOGDIR}/log_{SESSION}.jsonl'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --head ce "
        "--label_mode logistic_mean --label_kernel erf")
RATIOS = {'cora': [0.013, 0.026, 0.052], 'citeseer': [0.009, 0.018, 0.036],
          'arxiv': [0.0005, 0.0025, 0.005], 'flickr': [0.001, 0.005, 0.01],
          'reddit': [0.0005, 0.001, 0.002]}
EXTRA = {'cora': '--weight_decay 5e-4', 'citeseer': '--weight_decay 5e-4',
         'arxiv': '', 'flickr': '', 'reddit': ''}
BASIS = {'cora': [0, 300], 'citeseer': [0, 300], 'arxiv': [1000], 'flickr': [1000], 'reddit': [1000]}
BETAS, GAMMAS = [1e-3, 2.8e-3, 1e-2, 3e-2], [1e-5, 1e-4, 1e-3]
PAT = re.compile(r'== gcn: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')

def run(ds, r, beta, gamma, repeat, tag='', basis=1000):
    cmd = (f"python main.py {BASE} {EXTRA[ds]} --expert_basis {basis} --dataset_name {ds} "
           f"--ratio {r} --beta {beta} --gamma {gamma} --repeat {repeat}")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC').stdout
    m = PAT.search(out)
    if not m:
        print('FAIL', ds, r, beta, gamma, basis, '\n', out[-1500:]); return None
    rec = dict(ds=ds, ratio=r, beta=beta, gamma=gamma, basis=basis, repeat=repeat, tag=tag,
               test=float(m[1]), std=float(m[2]), val=float(m[3]), sec=round(time.time() - t))
    ex = re.search(r'expert:.*', out); rec['expert'] = ex[0] if ex else ''
    with open(LOG, 'a') as f: f.write(json.dumps(rec) + '\n')
    print(f"{ds:9s} r={r:<7g} basis={basis:<5} beta={beta:<7g} gamma={gamma:<6g}  "
          f"val {rec['val']:.2f}  test {rec['test']:.2f}±{rec['std']:.2f}  ({rec['sec']}s)")
    return rec

def load():                       # 세 세션 로그 병합
    recs = [json.loads(l) for f in glob.glob(f'{LOGDIR}/log_*.jsonl') for l in open(f)]
    return pd.DataFrame(recs)

def done(ds, r, beta, gamma, tag, basis):
    df = load()
    return len(df) and ((df.ds == ds) & (df.ratio == r) & (df.beta == beta) & (df.gamma == gamma)
                        & (df.basis == basis) & (df.tag == tag)).any()

def grid(ds, repeat=3):
    for r in RATIOS[ds]:
        for basis, beta, gamma in itertools.product(BASIS[ds], BETAS, GAMMAS):
            if not done(ds, r, beta, gamma, 'grid', basis):
                run(ds, r, beta, gamma, repeat, 'grid', basis)

# ============ Cell 2: 격자 — 세션별로 한 줄만 남기고 지우기 =============
# 세션 A
grid('arxiv')
# 세션 B
# grid('reddit')
# 세션 C
# grid('flickr'); grid('cora'); grid('citeseer')

# ============ Cell 3: val 기준 최적 → repeat 10 (격자가 다 끝난 뒤, 아무 세션에서나) =============
def best_by_val(ds):
    df = load(); df = df[(df.ds == ds) & (df.tag == 'grid')]
    return df.sort_values('val', ascending=False).groupby('ratio').head(1)

MINE = {'A': ['arxiv'], 'B': ['reddit'], 'C': ['flickr', 'cora', 'citeseer']}[SESSION]
for ds in MINE:
    for _, row in best_by_val(ds).iterrows():
        if not done(ds, row.ratio, row.beta, row.gamma, 'final', row.basis):
            run(ds, row.ratio, row.beta, row.gamma, 10, 'final', int(row.basis))

# ============ Cell 4: 메인 표 (어느 세션에서나) =============
df = load(); fin = df[df.tag == 'final'].copy()
fin['cell'] = fin.apply(lambda x: f"{x.test:.1f}±{x.std:.1f}", axis=1)
fin['cfg'] = fin.apply(lambda x: f"b={x.basis:g} β={x.beta:g} γ={x.gamma:g}", axis=1)
print(fin.pivot(index='ds', columns='ratio', values='cell').to_string())
print(fin.pivot(index='ds', columns='ratio', values='cfg').to_string())
print(fin[['ds', 'ratio', 'basis', 'beta', 'gamma', 'val', 'test', 'std']].to_string(index=False))
