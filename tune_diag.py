# ============ Cell 1: common (SESSION = 'all' | 'A' arxiv | 'C' cora + citeseer) =============
# Two short diagnostics at the main-table condensation (raw, l1, mu 0.3; one density per dataset), repeat 5:
#   student/bound mismatch : the student weights condensed nodes uniformly (1/n_j per node) while the bound is n_j-weighted.
#       cell_weight 1  -> n_j-weighted student loss (DIAGNOSTIC ONLY, changes the downstream loss)
#       balanced s     -> equal-mass cells (cap = s * N/k after the l1 sweeps), the protocol-clean way to get the same effect
#   residual correction    : q_j += n_j^L/(n_j^L + kappa) * mean OOF residual of the labelled members (5-fold teacher)
SESSION = 'all'
import subprocess, re, json, os, time, glob
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'diag1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c resid_folds /content/CGC/scr/para.py', shell=True,
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
VARIANTS = [('base', ''), ('cell_weight 1', '--cell_weight 1'), ('balanced 1.5', '--balanced 1.5'),
            ('balanced 1.2', '--balanced 1.2'), ('resid k1', '--resid_corr 1'), ('resid k3', '--resid_corr 3'),
            ('resid k10', '--resid_corr 10')]
DOWN = '0,0.1,0.3,0.5,0.7,0.9;5e-4'
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+)  L1 ([\d.]+)  argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)')
PAT_R = re.compile(r'resid_corr: kappa [\d.]+  \d+-fold OOF teacher acc ([\d.]+)%  cells with labels (\d+)/(\d+) \(labels/cell ([\d.]+)\)  '
                   r'mean L1 move ([\d.]+)  argmax changed ([\d.]+)%')
PAT_B = re.compile(r'balanced \(after l1\): cap (\d+) .*cell size min/max (\d+)/(\d+)')

def load():
    recs = [json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)]
    return pd.DataFrame(recs)

def done(ds, r, name):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.variant == name)).any()

def run(ds, r, kernel, gamma, temp, name, flags):
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --gamma {gamma} "
           f"--teacher_temp {temp} {flags} --repeat {REPEAT} --down_grid '{DOWN}'")
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC')
    out = p.stdout + p.stderr
    rows = PAT_D.findall(out)
    g = PAT_G.search(out)
    if not rows or not g:
        print('FAIL', ds, r, name, '\n', out[-2000:]); return
    rr, bb = PAT_R.search(out), PAT_B.search(out)
    cd = re.search(r'cell diag.*', out); cd = cd[0] if cd else ''
    ex = dict(kl_g=float(g[1]), agree=float(g[3]), h_true=float(g[4]), h_q=float(g[5]),
              oof_acc=float(rr[1]) if rr else None, lab_cells=f'{rr[2]}/{rr[3]}' if rr else None,
              lab_per_cell=float(rr[4]) if rr else None, moved=float(rr[5]) if rr else None, flipped=float(rr[6]) if rr else None,
              cap=int(bb[1]) if bb else None, size_min=int(bb[2]) if bb else None, size_max=int(bb[3]) if bb else None, diag=cd)
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, variant=name, gamma=gamma, temp=temp, drop=float(do_), wd=float(wd_),
                                    repeat=REPEAT, test=float(te), std=float(sd), val=float(va), **ex)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    extra = (f"OOF acc {ex['oof_acc']} labelled cells {ex['lab_cells']} flipped {ex['flipped']}%  " if rr else
             f"cap {ex['cap']} size {ex['size_min']}/{ex['size_max']}  " if bb else '')
    print(f"{ds:8s} r={r:<7g} {name:14s} {extra}group KL {ex['kl_g']:.4f} agree {ex['agree']:.1f}%  "
          f"val {best[5]} test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

print(f'{SESSION}: {[(c[0], c[1]) for c in CELLS]}  variants {[v[0] for v in VARIANTS]}')

# ============ Cell 2: run (done() resumes) =============
for ds, r, kernel, gamma, temp in CELLS:
    for name, flags in VARIANTS:
        if not done(ds, r, name):
            run(ds, r, kernel, gamma, temp, name, flags)

# ============ Cell 3: table (run where all diag1_*.jsonl are present) =============
df = load(); assert len(df), 'no logs'
b = df.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'variant']).head(1)
b = b.assign(cell=b.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1))
order = [v[0] for v in VARIANTS]
print('##### student test per ds x variant   [val-selected dropout, wd 5e-4, repeat 5]')
print(b.pivot_table(index='ds', columns='variant', values='cell', aggfunc='first')[order].to_string())
for m, title in [('kl_g', 'group KL(true||q)'), ('agree', 'argmax agreement with the true composition [%]'),
                 ('size_max', 'largest cell (balanced rows)'), ('flipped', 'cells whose argmax the residual correction changed [%]')]:
    print(f'\n##### {title}')
    t = b.pivot_table(index='ds', columns='variant', values=m, aggfunc='first')
    print(t[[c for c in order if c in t.columns]].round(4).to_string())
print('\n##### OOF teacher accuracy (resid rows) and labels per labelled cell')
print(b[b.variant.str.startswith('resid')].groupby('ds')[['oof_acc', 'lab_cells', 'lab_per_cell']].first().to_string())
