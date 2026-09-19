# ============ Cell 1 (single cell; teacher only, ~3 min) =============
# Does the teacher's error grow with the distance to the nearest LABELLED node? If it does, a proximity-based confidence
# weight in the cell mean has a basis; if accuracy is flat across the bins, it only shrinks the effective sample size.
# Kernel teacher at the protocol-A settings; val+test nodes binned by hop distance (BFS) and by feature distance (quartiles).
# Plus the averaging diagnostic: within each distance quartile, error of the group-mean posterior vs the true label mean for
# groups of n similar nodes (n 5..80) - a 1/sqrt(n) decay is variance (averaging helps), a floor is bias.
import subprocess, re
subprocess.run('git -C /content/CGC pull', shell=True)
assert subprocess.run('grep -c avg_diag /content/CGC/scr/para.py', shell=True,
                      capture_output=True, text=True).stdout.strip() not in ('', '0'), 'Colab checkout is stale'
BASE = ("--gpu 0 --generate_adj 0 --raw_data_dir /content/data/ --clustering kmeans --landmark kmeans --h_pool all "
        "--ce_steps 1000 --probe_tol 1e-6 --head ce --label_mode kernel_mean --kernel_prior rkhs --cluster_obj l1 "
        "--cluster_feat last --expert_basis 3000 --refine_teacher kernel --conv_depth 2 --bregman 1 --teacher_temp 1 "
        "--no_hyperpara 1 --dist_diag 1 --avg_diag 1 --teacher_only 1")
RUNS = [('cora',     0.026,  'relu1', 0.1,  0),
        ('citeseer', 0.018,  'erf',   10.0, 1),
        ('arxiv',    0.0025, 'erf',   1e-4, 0)]
for ds, r, kernel, gamma, fn in RUNS:
    cmd = f"python main.py {BASE} --dataset_name {ds} --ratio {r} --label_kernel {kernel} --gamma {gamma} --feat_norm {fn}"
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/CGC'); out = out.stdout + out.stderr
    m = re.search(r'label-distance diag.*?(?=teacher_only: stopping)', out, re.S)
    ex = re.search(r'expert: train ([\d.]+)%(?:\s+val ([\d.]+)%)?(?:\s+test ([\d.]+))?', out)
    print(f'##### {ds} {r:g}  ({kernel} gamma {gamma:g} fn {fn})' + (f'  overall: train {ex[1]}% val {ex[2]}% test {ex[3]}' if ex and ex[3] else ''))
    print(m[0] if m else 'FAIL\n' + out[-1500:])
