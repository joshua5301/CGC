# ============ Cell 1: 공통 =============
import subprocess, re, itertools, json, os, time
import pandas as pd

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --head ce "
        "--label_mode logistic_mean --label_kernel erf")

RATIOS = {                      # CGC/GCond 표준 밀도
    'cora':     [0.013, 0.026, 0.052],
    'citeseer': [0.009, 0.018, 0.036],
    'arxiv':    [0.0005, 0.0025, 0.005],
    'flickr':   [0.001, 0.005, 0.01],
    'reddit':   [0.0005, 0.001, 0.002],
}
EXTRA = {                       # 데이터셋별 고정 플래그
    'cora':     '--expert_basis 0 --weight_decay 5e-4',
    'citeseer': '--expert_basis 0 --weight_decay 5e-4',
    'arxiv':    '--expert_basis 1000',
    'flickr':   '--expert_basis 1000',
    'reddit':   '--expert_basis 1000',
}
BETAS  = [1e-3, 2.8e-3, 1e-2, 3e-2]
GAMMAS = [1e-5, 1e-4, 1e-3]

LOG = '/content/tune_log.jsonl'
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
               test=float(m[1]), std=float(m[2]), val=float(m[3]), sec=round(time.time()-t))
    ex = re.search(r'expert:.*', out); rec['expert'] = ex[0] if ex else ''
    with open(LOG, 'a') as f: f.write(json.dumps(rec) + '\n')
    print(f"{ds:9s} r={r:<7g} beta={beta:<7g} gamma={gamma:<6g}  val {rec['val']:.2f}  test {rec['test']:.2f}±{rec['std']:.2f}  ({rec['sec']}s)")
    return rec

def load():
    if not os.path.exists(LOG): return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in open(LOG)])

def done(ds, r, beta, gamma, tag):
    df = load()
    return len(df) and ((df.ds==ds)&(df.ratio==r)&(df.beta==beta)&(df.gamma==gamma)&(df.tag==tag)).any()

# ============ Cell 2: 격자 (데이터셋 하나씩, 중단 후 재실행해도 이어감) =============
def grid(ds, repeat=3):
    for r in RATIOS[ds]:
        for beta, gamma in itertools.product(BETAS, GAMMAS):
            if not done(ds, r, beta, gamma, 'grid'):
                run(ds, r, beta, gamma, repeat, 'grid')

grid('arxiv')      # ~36 runs × ~1.5 min
grid('reddit')
grid('flickr')
grid('cora')
grid('citeseer')

# ============ Cell 3: val 기준 최적 선택 + 최종 repeat 10 =============
def best_by_val(ds):
    df = load(); df = df[(df.ds==ds)&(df.tag=='grid')]
    return df.sort_values('val', ascending=False).groupby('ratio').head(1)

for ds in RATIOS:
    for _, row in best_by_val(ds).iterrows():
        if not done(ds, row.ratio, row.beta, row.gamma, 'final'):
            run(ds, row.ratio, row.beta, row.gamma, 10, 'final')

# ============ Cell 4: 메인 표 =============
df = load(); fin = df[df.tag=='final'].copy()
fin['cell'] = fin.apply(lambda x: f"{x.test:.1f}±{x.std:.1f}", axis=1)
fin['cfg']  = fin.apply(lambda x: f"β={x.beta:g} γ={x.gamma:g}", axis=1)
print(fin.pivot(index='ds', columns='ratio', values='cell').to_string())
print(fin.pivot(index='ds', columns='ratio', values='cfg').to_string())
print(fin[['ds','ratio','val','test','std','beta','gamma']].to_string(index=False))
