# ============ Cell 1: common (SESSION = 'all' | 'A' arxiv | 'C' cora + citeseer) =============
# Equal-mass cells by entropic optimal transport (--sinkhorn eps): the l1 cost (distance / s + mu * KL) with uniform
# marginals (node mass 1, cell mass N/k), eps annealed from sk_anneal*eps to eps over sk_iters rounds, hardened by argmax.
# Sweep eps x rounds at the main-table condensation (one density per dataset), paired with base (eps 0), repeat 5.
SESSION = 'all'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'sk1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c sk_anneal /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--conv_depth 2 --feat_norm 0 --bregman 0.3 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 "
        "--dropout 0.5 --weight_decay 5e-4")
#          ds          ratio   kernel   gamma  T
CELLS = {'A': [('arxiv',    0.0025, 'erf',   1e-4, 0.25)],
         'C': [('cora',     0.026,  'relu1', 1.0,  1.0),
               ('citeseer', 0.018,  'erf',   10.0, 1.0)]}
CELLS = CELLS['A'] + CELLS['C'] if SESSION == 'all' else CELLS[SESSION]
#          (eps, rounds, anneal)      eps 0 = base (plain l1)
GRID = [(0.0, 0, 1.0), (0.05, 10, 10.0), (0.02, 10, 10.0), (0.01, 10, 10.0), (0.02, 20, 10.0), (0.02, 10, 1.0)]
DOWN = '0,0.1,0.3,0.5,0.7,0.9;5e-4'
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+)  L1 ([\d.]+)  argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)')
PAT_S = re.compile(r'sinkhorn \(after l1\).*last round moved (\d+) nodes  cell size min/max (\d+)/(\d+) \(target (\d+)\)')
PAT_C = re.compile(r'cell diag: within-var ([\d.]+) \(([\d.]+)% of total\).*size min/p10/med/p90/max (\d+)/(\d+)/(\d+)/(\d+)/(\d+)')

def load():
    recs = [json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)]
    return pd.DataFrame(recs)

def done(ds, r, cfg):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.eps == cfg[0]) & (df.rounds == cfg[1]) & (df.anneal == cfg[2])).any()

def run(ds, r, kernel, gamma, temp, cfg):
    eps, rounds, anneal = cfg
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --gamma {gamma} "
           f"--teacher_temp {temp} --sinkhorn {eps} --sk_iters {max(rounds, 1)} --sk_anneal {anneal} "
           f"--repeat {REPEAT} --down_grid '{DOWN}'")
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC')
    out = p.stdout + p.stderr
    rows = PAT_D.findall(out)
    g = PAT_G.search(out)
    if not rows or not g:
        print('FAIL', ds, r, cfg, '\n', out[-2000:]); return
    sk, cd = PAT_S.search(out), PAT_C.search(out)
    ct = re.search(r'Condensation time: ([0-9.]+)', out); ct = float(ct[1]) if ct else float('nan')
    ex = dict(kl_g=float(g[1]), agree=float(g[3]), h_true=float(g[4]), h_q=float(g[5]),
              moved=int(sk[1]) if sk else None, size_min=int(sk[2]) if sk else (int(cd[3]) if cd else None),
              size_max=int(sk[3]) if sk else (int(cd[7]) if cd else None), target=int(sk[4]) if sk else None,
              within=float(cd[2]) if cd else None, cond_s=ct)
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, eps=eps, rounds=rounds, anneal=anneal, gamma=gamma, temp=temp,
                                    drop=float(do_), wd=float(wd_), repeat=REPEAT,
                                    test=float(te), std=float(sd), val=float(va), **ex)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"{ds:8s} r={r:<7g} eps={eps:<5g} rounds={rounds:<3d} anneal={anneal:<3g}  size {ex['size_min']}/{ex['size_max']}"
          + (f" (target {ex['target']}, moved {ex['moved']})" if sk else '')
          + f"  within-var {ex['within']}%  group KL {ex['kl_g']:.4f} agree {ex['agree']:.1f}%  "
          f"val {best[5]} test {best[3]}±{best[4]}  ({round(time.time() - t)}s, cond {ct:.0f}s)")

print(f'{SESSION}: {[(c[0], c[1]) for c in CELLS]}  grid {GRID}')

# ============ Cell 2: run (done() resumes) =============
for ds, r, kernel, gamma, temp in CELLS:
    for cfg in GRID:
        if not done(ds, r, cfg):
            run(ds, r, kernel, gamma, temp, cfg)

# ============ Cell 3: table (run where all sk1_*.jsonl are present) =============
df = load(); assert len(df), 'no logs'
b = df.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'eps', 'rounds', 'anneal']).head(1)
b = b.assign(cfg=b.apply(lambda x: 'base' if x.eps == 0 else f"eps{x.eps:g} r{int(x.rounds)} a{x.anneal:g}", axis=1),
             cell=b.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1))
order = ['base'] + [f"eps{e:g} r{r} a{a:g}" for e, r, a in GRID if e > 0]
print('##### student test per ds x sinkhorn config   [val-selected dropout, wd 5e-4, repeat 5]')
print(b.pivot_table(index='ds', columns='cfg', values='cell', aggfunc='first')[order].to_string())
for m, title in [('size_max', 'largest cell'), ('size_min', 'smallest cell'), ('within', 'within-cell posterior variance [% of total]'),
                 ('kl_g', 'group KL(true||q)'), ('agree', 'argmax agreement [%]'), ('cond_s', 'condensation time [s]')]:
    print(f'\n##### {title}')
    t = b.pivot_table(index='ds', columns='cfg', values=m, aggfunc='first')
    print(t[[c for c in order if c in t.columns]].round(3).to_string())
