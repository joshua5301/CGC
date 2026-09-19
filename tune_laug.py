# ============ Cell 1 (single session) =============
# CGC-style label augmentation, labels only (--label_aug): the cell members are also seen at other propagation depths
# (A X and X views of the SAME nodes), the kernel teacher (fitted on A^2 X) is applied to those views and the posteriors are
# averaged into the cell label together with the main view. Partition and centres stay on A^2 X. Doubles / triples the
# label sample per cell without crossing cell boundaries (the failure mode of cell-level kNN smoothing).
# Prints the teacher's accuracy on each extra view (off-distribution input) and its argmax agreement with the main view.
import subprocess, re, json, os, time, itertools
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/cgc_tune'; os.makedirs(LOGDIR, exist_ok=True)
TAG = 'laug1'
LOG = f'{LOGDIR}/{TAG}.jsonl'
pd.set_option('display.width', 250)
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c label_aug_w /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'

BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans "
        "--landmark kmeans --h_pool all --ce_steps 1000 --probe_tol 1e-6 --head ce "
        "--label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 --cluster_feat last --expert_basis 3000 "
        "--conv_depth 2 --no_hyperpara 1 --lr 0.01 --epoch 1000 --eval_every 10 --dropout 0.5 --weight_decay 5e-4")
#         ds          ratio  kernel   gamma fn  mu   temps
CELLS = [('cora',     0.052, 'relu1', 0.01, 0, 2.0, [1.0, 0.5]),
         ('citeseer', 0.036, 'erf',   3.0,  1, 0.2, [1.0, 0.5, 0.25])]
#           name        aug     w
VARIANTS = [('none',      '',    1.0),
            ('+1hop',     '1',   1.0),
            ('+1hop w.5', '1',   0.5),
            ('+0,1hop',   '0,1', 1.0)]
DOWN = '0,0.1,0.3,0.5,0.7,0.9;5e-4,5e-3'
REPEAT = 5
PAT_D = re.compile(r'== down do=([\d.]+) wd=([\d.e-]+)(?: lr=([\d.e-]+))?: ([\d.]+) \+- ([\d.]+)\s+\(val ([\d.]+)\)')
PAT_G = re.compile(r'group diag: KL\(true\|\|q\) ([\d.]+) .*argmax agree ([\d.]+)%  H\(true\) ([\d.]+)  H\(q\) ([\d.]+)')
PAT_V = re.compile(r'label_aug view: teacher accuracy on this view ([\d.]+)%  \(main view ([\d.]+)%\)  agreement of argmax with the main view ([\d.]+)%')

def load():
    return pd.DataFrame([json.loads(l) for l in open(LOG)]) if os.path.exists(LOG) else pd.DataFrame()

def done(ds, r, name, temp):
    df = load()
    return len(df) > 0 and ((df.ds == ds) & (df.ratio == r) & (df.variant == name) & (df.temp == temp)).any()

def run(ds, r, kernel, gamma, fn, mu, temp, name, aug, w):
    extra = f"--label_aug '{aug}' --label_aug_w {w}" if aug else ''
    cmd = (f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --gamma {gamma} --feat_norm {fn} "
           f"--bregman {mu} --teacher_temp {temp} {extra} --repeat {REPEAT} --down_grid '{DOWN}'")
    t = time.time()
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    rows = PAT_D.findall(out); g = PAT_G.search(out); views = PAT_V.findall(out)
    if not rows or not g:
        print('FAIL', ds, r, name, temp, '\n', out[-2000:]); return
    with open(LOG, 'a') as f:
        for do_, wd_, lr_, te, sd, va in rows:
            f.write(json.dumps(dict(ds=ds, ratio=r, variant=name, aug=aug, w=w, temp=temp,
                                    drop=float(do_), wd=float(wd_), repeat=REPEAT, test=float(te), std=float(sd), val=float(va),
                                    kl_g=float(g[1]), agree=float(g[2]), h_true=float(g[3]), h_q=float(g[4]),
                                    view_acc=[float(v[1]) for v in views], main_acc=float(views[0][2]) if views else None,
                                    view_agree=[float(v[3]) for v in views])) + '\n')
    best = max(rows, key=lambda x: float(x[5]))
    print(f"{ds:8s} {r:<6g} {name:10s} T={temp:<4g}" + (f"  view acc {[v[1] for v in views]} (main {views[0][2]}) agree-with-main {[v[3] for v in views]}" if views else '') +
          f"  cell agree {g[2]}%  KL {g[1]}  H(q) {g[4]}  val {best[5]} (do={best[0]} wd={best[1]}) test {best[3]}±{best[4]}  ({round(time.time() - t)}s)")

print('laug1: CGC-style label augmentation (labels only) at cora 5.2% and citeseer 3.6%')

# ============ Cell 2: run =============
for ds, r, kernel, gamma, fn, mu, temps in CELLS:
    for (name, aug, w), temp in itertools.product(VARIANTS, temps):
        if not done(ds, r, name, temp):
            run(ds, r, kernel, gamma, fn, mu, temp, name, aug, w)

# ============ Cell 3: table =============
df = load()
b = df.sort_values('val', ascending=False).groupby(['ds', 'ratio', 'variant', 'temp']).head(1)
b = b.assign(cell=b.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f}", axis=1))
order = [v[0] for v in VARIANTS]
print('##### student test per (ds, T) x variant   [val-selected dropout/wd, repeat 5]')
print(b.pivot_table(index=['ds', 'temp'], columns='variant', values='cell', aggfunc='first').reindex(columns=order).to_string())
bb = b.sort_values('val', ascending=False).groupby(['ds', 'variant']).head(1)
print('\n##### T selected on val')
print(bb.assign(cell=bb.apply(lambda x: f"{x['test']:.1f}+-{x['std']:.1f} (T{x['temp']:g})", axis=1)).pivot_table(index='ds', columns='variant', values='cell', aggfunc='first').reindex(columns=order).to_string())
print('\n##### teacher accuracy on the extra views (T 1 rows): main / views / argmax agreement with main')
for _, x in b[(b.temp == 1.0) & (b.variant != 'none')].iterrows():
    print(f"  {x.ds} {x.variant:10s}: main {x.main_acc}  views {x.view_acc}  agree {x.view_agree}")
for m, title in [('agree', 'cell argmax agreement [%]'), ('kl_g', 'group KL(true||q)'), ('h_q', 'H(q)')]:
    print(f'\n##### {title}   [T 1]')
    print(b[b.temp == 1.0].pivot_table(index='ds', columns='variant', values=m, aggfunc='first').reindex(columns=order).round(3).to_string())
