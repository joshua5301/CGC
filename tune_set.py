# ============ Cell 1: common (SESSION = 'all' | 'A' arxiv | 'C' cora + citeseer) =============
# Group teacher (--set_head 1): a Deep-Sets head on the fixed kernel teacher predicts each cell's class composition.
# phi = [f(h), log f(h), (rank-r projection of h)], mean pooling, residual MLP initialised at the identity, trained on
# cell-like sets (k-means at 0.5/1/2/4/8 x the cell count + the actual cells) with the mean one-hot of their labelled
# members as target, early-stopped on the val-labelled members. Partition, node teacher and KL term unchanged.
# Paired against --set_head 0 at the group1 val-best (gamma, T) per dataset; group diag + student, repeat 5.
SESSION = 'all'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'set1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c set_mults /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--conv_depth 2 --feat_norm 0 --bregman 0.3 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 "
        "--dropout 0.5 --weight_decay 5e-4")
#          ds          ratio   kernel   (gamma, T) grid: group1 val-best and its neighbours
CELLS = {'A': [('arxiv',    0.0025, 'erf',   [(1e-4, 1.0), (1e-4, 0.25)])],
         'C': [('cora',     0.026,  'relu1', [(1.0, 1.0), (1.0, 0.5), (0.1, 1.0)]),
               ('citeseer', 0.018,  'erf',   [(10.0, 1.0), (30.0, 0.5), (1.0, 0.5)])]}
CELLS = CELLS['A'] + CELLS['C'] if SESSION == 'all' else CELLS[SESSION]
#        (set_head, rank, hidden, wd)
HEADS = [(0, 0, 0, 0.0), (1, 0, 32, 1e-3), (1, 8, 32, 1e-3), (1, 0, 32, 1e-2)]
DOWN = '0,0.1,0.3,0.5,0.7,0.9;5e-4'
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+)  L1 ([\d.]+)  argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)'
                   r'  \(n-weighted, (\d+) cells\)(?:  \| node teacher on the pool: NLL ([\d.]+)  acc ([\d.]+)%  H ([\d.]+))?')
PAT_S = re.compile(r'set head: (\d+) sets \((\d+) with train labels(?:, (\d+) with val labels)?\).*set CE train ([\d.]+) -> ([\d.]+)'
                   r'(?:  val ([\d.]+) -> ([\d.]+) \(best at step (\d+)\))?')
PAT_M = re.compile(r'set head: labels moved  mean L1 ([\d.]+)  argmax changed ([\d.]+)% of cells  H ([\d.]+) -> ([\d.]+)')

def load():
    recs = [json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)]
    return pd.DataFrame(recs)

def done(ds, r, gamma, temp, head):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.gamma == gamma) & (df.temp == temp)
                            & (df.set_head == head[0]) & (df.rank == head[1]) & (df.set_wd == head[3])).any()

def run(ds, r, kernel, gamma, temp, head):
    sh, rank, hidden, wd = head
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --gamma {gamma} "
           f"--teacher_temp {temp} --set_head {sh} --set_rank {rank} --set_hidden {max(hidden, 1)} --set_wd {wd} "
           f"--repeat {REPEAT} --down_grid '{DOWN}'")
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC')
    out = p.stdout + p.stderr
    rows = PAT_D.findall(out)
    g = PAT_G.search(out)
    if not rows or not g:
        print('FAIL', ds, r, gamma, temp, head, '\n', out[-2000:]); return
    sf, sm = PAT_S.search(out), PAT_M.search(out)
    gd = dict(kl_g=float(g[1]), l1_g=float(g[2]), agree=float(g[3]), h_true=float(g[4]), h_q=float(g[5]),
              nll_n=float(g[7]) if g[7] else None, acc_n=float(g[8]) if g[8] else None)
    sd = dict(n_sets=int(sf[1]) if sf else None, ce0=float(sf[4]) if sf else None, ce1=float(sf[5]) if sf else None,
              va0=float(sf[6]) if sf and sf[6] else None, va1=float(sf[7]) if sf and sf[7] else None,
              best_it=int(sf[8]) if sf and sf[8] else None,
              moved=float(sm[1]) if sm else None, flipped=float(sm[2]) if sm else None)
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd_, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, kernel=kernel, gamma=gamma, temp=temp, set_head=sh, rank=rank,
                                    hidden=hidden, set_wd=wd, drop=float(do_), wd=float(wd_), repeat=REPEAT,
                                    test=float(te), std=float(sd_), val=float(va), **gd, **sd)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"{ds:8s} r={r:<7g} g={gamma:<5g} T={temp:<4g} head={sh} rank={rank} wd={wd:g}  "
          + (f"setCE {sd['ce0']:.3f}->{sd['ce1']:.3f} val {sd['va0']}->{sd['va1']} flipped {sd['flipped']}%  " if sf else '')
          + f"group KL {gd['kl_g']:.4f} agree {gd['agree']:.1f}% H(q) {gd['h_q']:.3f}  val {best[5]} test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

print(f'{SESSION}: {[(c[0], c[1]) for c in CELLS]}  heads {HEADS}')

# ============ Cell 2: run (done() resumes) =============
for ds, r, kernel, grid in CELLS:
    for (gamma, temp), head in itertools.product(grid, HEADS):
        if not done(ds, r, gamma, temp, head):
            run(ds, r, kernel, gamma, temp, head)

# ============ Cell 3: tables (run where all set1_*.jsonl are present) =============
df = load(); assert len(df), 'no logs'
b = df.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'gamma', 'temp', 'set_head', 'rank', 'set_wd']).head(1)
b = b.assign(head=b.apply(lambda x: 'node-mean' if x.set_head == 0 else f"set r{int(x['rank'])} wd{x.set_wd:g}", axis=1),
             cell=b.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1))
print('##### student test per (ds, gamma, T) x head   [val-selected dropout, repeat 5]')
print(b.pivot_table(index=['ds', 'gamma', 'temp'], columns='head', values='cell', aggfunc='first').to_string())
for m, title in [('kl_g', 'group KL(true||q)'), ('agree', 'argmax agreement with the true composition [%]'),
                 ('h_q', 'H(q) [H(true) in the last column]'), ('flipped', 'cells whose argmax the set head changed [%]'),
                 ('va1', 'set CE on val-labelled members after training (va0 = before)')]:
    print(f'\n##### {title}')
    t = b.pivot_table(index=['ds', 'gamma', 'temp'], columns='head', values=m, aggfunc='first').round(4)
    if m == 'h_q':
        t['H(true)'] = b.groupby(['ds', 'gamma', 'temp']).h_true.first().round(3)
    if m == 'va1':
        t['before'] = b[b.set_head == 1].groupby(['ds', 'gamma', 'temp']).va0.first().round(4)
    print(t.to_string())
