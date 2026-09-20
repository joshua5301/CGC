# ============ Cell 1: common (SESSION = 'flickr' | 'reddit') =============
# First sweep of the inductive datasets on the rewritten code, narrowed protocol: ONE condensation config per dataset is
# chosen by the pooled validation accuracy over the three densities (sum of the per-density val), the student dropout per
# density on val. Grid: kernel {erf, relu} x gamma {1e-4, 1e-3, 1e-2} x T {1, 0.25} x kl_weight {0.2, 1} = 24 configs x 3
# densities; dropout {0.1, 0.5, 0.9} inside main.py; wd 5e-4, basis 3000, repeat 3 (the table is re-run at repeat 10).
SESSION = 'flickr'
import subprocess, re, json, os, time, glob, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'ind1'
LOG = f'{LOGDIR}/{TAG}_{SESSION}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC fetch origin && git -C /content/CGC reset --hard origin/main', shell=True)
subprocess.run('pip install -q faiss-cpu', shell=True)

RATIOS = {'flickr': [0.001, 0.005, 0.01], 'reddit': [0.0005, 0.001, 0.002]}[SESSION]
KERNELS, GAMMAS, TEMPS, KLS = ['erf', 'relu'], [1e-4, 1e-3, 1e-2], [1.0, 0.25], [0.2, 1.0]
DROPOUTS, REPEAT = '0.1,0.5,0.9', 3
PAT_D = re.compile(r'-- dropout ([\d.]+): val ([\d.]+)  test ([\d.]+) \+- ([\d.]+)')
PAT_C = re.compile(r'condensed: (\d+) nodes \(budget (\d+)\)  time ([\d.]+) s')

def load():
    return pd.DataFrame([json.loads(l) for f in glob.glob(f'{LOGDIR}/{TAG}_*.jsonl') for l in open(f)])

def done(r, kernel, gamma, temp, kl):
    df = load()
    return len(df) > 0 and ((df.ratio == r) & (df.kernel == kernel) & (df.gamma == gamma) & (df.temp == temp) & (df.kl == kl)).any()

def run(r, kernel, gamma, temp, kl):
    cmd = (f"python main.py --gpu 0 --raw_data_dir /content/data/ --dataset_name {SESSION} --ratio {r} --teacher_kernel {kernel} "
           f"--gamma {gamma} --T {temp} --kl_weight {kl} --dropout {DROPOUTS} --repeat {REPEAT}")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    rows = PAT_D.findall(out); c = PAT_C.search(out)
    if not rows:
        print('FAIL', r, kernel, gamma, temp, kl, '\n', out[-2000:]); return
    with open(LOG, 'a') as f:
        for do_, va, te, sd in rows:
            f.write(json.dumps(dict(ds=SESSION, ratio=r, kernel=kernel, gamma=gamma, temp=temp, kl=kl, dropout=float(do_),
                                    val=float(va), test=float(te), std=float(sd), repeat=REPEAT,
                                    cells=int(c[1]) if c else None, cond_s=float(c[3]) if c else None)) + '\n')
    best = max(rows, key=lambda x: float(x[1]))
    print(f"{SESSION:7s} {r:<6g} {kernel:4s} g={gamma:<6g} T={temp:<4g} kl={kl:<3g}  val {best[1]} (do={best[0]}) test {best[2]}±{best[3]}"
          f"  cond {c[3] if c else '?'}s  ({round(time.time() - t)}s)")

print(f'ind1 {SESSION}: ratios {RATIOS}  {len(KERNELS) * len(GAMMAS) * len(TEMPS) * len(KLS)} configs x 3 densities')

# ============ Cell 2: run (done() resumes) =============
for r, kernel, gamma, temp, kl in itertools.product(RATIOS, KERNELS, GAMMAS, TEMPS, KLS):
    if not done(r, kernel, gamma, temp, kl):
        run(r, kernel, gamma, temp, kl)

# ============ Cell 3: tables =============
df = load(); df = df[df.ds == SESSION]
KEY = ['kernel', 'gamma', 'temp', 'kl']
b = df.sort_values('val', ascending=False).groupby(KEY + ['ratio']).head(1)      # dropout on val per (config, density)
print('##### per-density val-best (dropout on val) - the "full" selection for reference')
for r, g in b.groupby('ratio'):
    x = g.sort_values('val', ascending=False).iloc[0]
    print(f"  {r:<6g}  {x.kernel} g={x.gamma:g} T={x.temp:g} kl={x.kl:g} do={x.dropout:g}   val {x.val:.2f}  test {x.test:.2f}+-{x['std']:.2f}")
pooled = b.groupby(KEY).agg(val_sum=('val', 'sum'), n=('ratio', 'nunique')).query('n == 3').sort_values('val_sum', ascending=False)
print('\n##### shared config over the three densities, ranked by the pooled val (sum); test per density at the val-best dropout')
for cfg, row in pooled.head(8).iterrows():
    sel = b.set_index(KEY).loc[cfg].sort_values('ratio')
    cells = '  '.join(f"{x.ratio:g}: {x.test:.2f}+-{x['std']:.2f} (do {x.dropout:g})" for _, x in sel.iterrows())
    print(f"  {cfg[0]:4s} g={cfg[1]:<6g} T={cfg[2]:<4g} kl={cfg[3]:<3g}  pooled val {row.val_sum / 3:.2f}   {cells}")
top = pooled.index[0]
print(f'\n##### chosen: kernel {top[0]}, gamma {top[1]:g}, T {top[2]:g}, kl_weight {top[3]:g}  ->  BEST_HYPERPARAMS_DICT rows:')
for _, x in b.set_index(KEY).loc[top].sort_values('ratio').iterrows():
    print(f"    ('{SESSION}', {x.ratio:g}): ('{top[0]}', {top[1]:g}, {top[2]:g}, {top[3]:g}, {x.dropout:g}),")
print('\n##### marginal effect of each axis (mean over the other axes and densities of the val-best-dropout val / test)')
for ax in KEY:
    print(f'  {ax}: ' + '  |  '.join(f"{v:g}: val {g.val.mean():.2f} test {g.test.mean():.2f}" if not isinstance(v, str) else f"{v}: val {g.val.mean():.2f} test {g.test.mean():.2f}" for v, g in b.groupby(ax)))
