# ============ Cell 1: common (SESSION = 'all' | 'A' arxiv | 'C' cora + citeseer) =============
# Student-in-the-loop ablation (--cycle k): after the student-agnostic condensation, k rounds of
#   linear student W fitted on the condensed set -> pool re-clustered in the student's logit metric ||(h - c) W||
#   -> cell-mean teacher labels.
# k = 0 is the main method (Euclidean = worst case over all L-Lipschitz students). The loop is the best case for one
# linear student; the downstream GCN then measures how much of that transfers to a different architecture.
# One density per dataset at the main-table config (raw space, l1 k-medians, erf/relu1 as selected), wd 5e-4,
# dropout on val, repeat 5.
SESSION = 'all'
import subprocess, re, json, os, time, glob
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'cycle1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c cycle_gamma /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--conv_depth 2 --feat_norm 0 --teacher_temp 1 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 "
        "--dropout 0.5 --weight_decay 5e-4")
#          ds          ratio   kernel  gamma  mu      (final5 val-best condensation, raw space)
CELLS = {'A': [('arxiv',    0.0025, 'erf',   1e-4, 0.3)],
         'C': [('cora',     0.026,  'relu1', 1.0,  0.3),
               ('citeseer', 0.018,  'erf',   10.0, 0.3)]}
CELLS = CELLS['A'] + CELLS['C'] if SESSION == 'all' else CELLS[SESSION]
CYCLES = [0, 1, 3]
CYCLE_GAMMA = 1e-2
DOWN = '0,0.1,0.3,0.5,0.7,0.9;5e-4'
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_C = re.compile(r'cycle (\d+): linear student on the condensed set  train ([\d.]+)%(?:  test ([\d.]+)%)?.*label change ([\d.]+)')

def load():
    recs = [json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)]
    return pd.DataFrame(recs)

def done(ds, r, cyc):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.cycle == cyc)).any()

def run(ds, r, kernel, gamma, mu, cyc):
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --gamma {gamma} "
           f"--bregman {mu} --cycle {cyc} --cycle_gamma {CYCLE_GAMMA} --repeat {REPEAT} --down_grid '{DOWN}'")
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC')
    out = p.stdout + p.stderr
    rows = PAT_D.findall(out)
    if not rows:
        print('FAIL', ds, r, cyc, '\n', out[-1500:]); return
    ex = re.search(r'expert: train ([\d.]+)%(?:\s+val ([\d.]+)%)?(?:\s+test ([\d.]+))?', out)
    cyc_rows = PAT_C.findall(out)
    cd = re.search(r'cell diag.*', out); cd = cd[0] if cd else ''
    ct = re.search(r'Condensation time: ([0-9.]+)', out); ct = float(ct[1]) if ct else float('nan')
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, kernel=kernel, gamma=gamma, mu=mu, cycle=cyc,
                                    drop=float(do_), wd=float(wd_), repeat=REPEAT,
                                    test=float(te), std=float(sd), val=float(va),
                                    t_test=float(ex[3]) if ex and ex[3] else None,
                                    s_train=float(cyc_rows[-1][1]) if cyc_rows else None,
                                    s_test=float(cyc_rows[-1][2]) if cyc_rows and cyc_rows[-1][2] else None,
                                    dY=float(cyc_rows[-1][3]) if cyc_rows else None,
                                    diag=cd, cond_s=ct)) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    stud = (f"  student test {cyc_rows[-1][2]} dY {cyc_rows[-1][3]}" if cyc_rows and cyc_rows[-1][2]
            else f"  student train {cyc_rows[-1][1]}" if cyc_rows else '')
    print(f"{ds:8s} r={r:<7g} {kernel:5s} g={gamma:<5g} mu={mu:g} cycle={cyc}  teacher {(ex[3] or ex[1]) if ex else '?'}{stud}  "
          f"val {best[5]} (do={best[0]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s, cond {ct:.0f}s)")

print(f'{SESSION}: {CELLS}  cycles {CYCLES}')

# ============ Cell 2: run (done() resumes) =============
for ds, r, kernel, gamma, mu in CELLS:
    for cyc in CYCLES:
        if not done(ds, r, cyc):
            run(ds, r, kernel, gamma, mu, cyc)

# ============ Cell 3: table (run where all cycle1_*.jsonl are present) =============
df = load(); assert len(df), 'no logs'
b = df.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'cycle']).head(1)
b = b.assign(cell=b.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f} (do={x['drop']:g})", axis=1))
print('##### downstream GCN test (val-selected dropout, wd 5e-4, repeat 5) per (ds, ratio) x cycle')
print(b.pivot_table(index=['ds', 'ratio'], columns='cycle', values='cell', aggfunc='first').to_string())
print('\n##### in-loop linear student test accuracy (last cycle) / label change per cycle')
print(b.pivot_table(index=['ds', 'ratio'], columns='cycle', values=['s_test', 'dY'], aggfunc='first').round(3).to_string())
print('\n##### condensation time [s]')
print(b.pivot_table(index=['ds', 'ratio'], columns='cycle', values='cond_s', aggfunc='first').round(1).to_string())
