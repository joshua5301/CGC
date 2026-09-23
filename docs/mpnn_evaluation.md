# MPNN surrogate, evaluated with the existing two-layer GCN

## Rho sweep after the full teacher search

Set `os.environ['GRIP_SWEEP_PRESET'] = 'rho'` before running the Colab cell.
It uses `R_rho=(1-rho)I+rho P_closed` with rho in {0, .05, .1, .2, .35, .5}
and mu in {.3, 1, 3, 10}. The teacher is fixed within each dataset: Cora
gamma=.01, T=.2; Citeseer gamma=.1, T=.2. Cora's T=.2 comes from the prior full
sweep's shared best teacher setting for mpnn/raw; this differs from pilot T=1.

The student remains the same two-layer GCN, uniform CE, dropout {.1,.5,.9},
student seeds {0,1,2}. With the two default datasets this is 48 condensations
and 432 student fits. GRIP is not retrained. rho=0 exactly reproduces the raw
ablation, so a separate raw method is unnecessary. Compare matched mu/dropout/
seed settings in summary.csv as well as validation-best settings per rho.

Results go to `GRIP_mpnn_rho` on Drive. The compact table includes each rho's
validation-best settings and ties. summary.csv includes rho; best_by_rho.json
contains a single validation-selected setting per rho, while best.json selects
across rho too. Config/cache keys include rho. This algorithm update changes
source fingerprints, so older experiment caches are not imported; within this
new sweep teacher logits are shared across rho/mu and reruns resume normally.

The corresponding standalone option is `--mpnn_rho`; the sweep option is
`--rhos`. rho=0 (and the supported rho=1 endpoint) are ablations rather than
uniform guarantees for arbitrary root-and-neighbor dependent layers. For
0<rho<1, the model-class envelope constant changes with rho. Objective values
across rho must not be interpreted as directly comparable certified risks.

```python
import os, subprocess
pull = subprocess.run(['git', '-C', '/content/GRIP', 'pull', '--ff-only'],
                      capture_output=True, text=True)
if pull.returncode:
    raise RuntimeError(pull.stdout + pull.stderr)
os.environ['GRIP_SWEEP_PRESET'] = 'rho'
%run /content/GRIP/colab_distance.py
```

## Original fixed-rho protocol

This is the default protocol in `colab_distance.py` and `sweep_distance.py`.
The earlier GCN-specific `distance` method remains an explicitly named legacy
option; it is NOT part of the default sweep.

## Condensation objective

Let B be binary incoming adjacency with exactly one self-loop per node,
P=diag(B 1)^(-1) B, and R=(I+P)/2. P is a comparison distribution over closed
neighborhoods, not the GCN propagation matrix. Duplicate edges are ignored.

```
D[v,j] = ||X[v] - C[j]||_2
J = mean_v { (R^2 D)[v,a[v]] + mu KL(F[v] || Yc[a[v]]) }
R^2 D = (D + 2 P D + P(P D))/4
```

Representatives C are raw-feature weighted geometric medians with weights
`(R.T)^2 S`. Labels Yc are cell means of the teacher F. Assignment is a batched
nonempty descent step. There is no OT solver, no GCN-specific mass correction,
no origin atom penalty, and no separate normalization of the two loss terms.
Float64 condensation is followed by float32 student training.

This implements the **optimizable surrogate** from
[the MPNN design](mpnn_identity_design.md), not a numerical certificate for all
MPNNs. Sensitivity constants, deletion remainders, degree/edge metadata costs,
and the final student residual are not estimated here. The theoretical
structural remainder is not silently treated as zero. Degree statistics are
diagnostics only, not a certified bound. Training has no added norm constraint.

## Evaluation fairness

All three default methods use the exact same existing `GCN(..., nlayers=2,
normalize=True)` and original evaluation graphs. The student's loss remains
the unchanged uniform mean soft-label CE in `src/utils.py:model_training`.
No cell-size weighting is used in either loss or sampling.

| Method | Condensation | Initial partition |
|---|---|---|
| mpnn | R^2 D + mu KL, raw-X representatives, Q=I | raw-X seeded k-means |
| raw | D + mu KL (K=0 ablation), raw-X representatives, Q=I | same raw-X seeded k-means |
| grip | existing GRIP partition on A_hat^2 X, identity edges | original GRIP initialization |

The mpnn/raw pair shares initialization for each seed to isolate propagation.
GRIP retains its original algorithm and its own initialization policy. Both
coefficient definitions and normalization differ from GRIP, so their grids are
documented separately. GRIP can return fewer than the requested budget if it
drops empty cells; node counts are recorded, not silently padded.

Teacher kernel, training labels, and A_hat^2 X features match the existing repo.
Default kernels: Cora ReLU; Citeseer erf. Training: width 256, 1000 epochs,
Adam lr .01, weight decay 5e-4, existing halfway learning-rate reduction,
best-validation epoch evaluated every 10 epochs. Dropout grid .1/.5/.9.
Student seeds 0/1/2 are reset for every setting and method. These are student
seeds, with one fixed condensation seed (0), not three independent condensations.

Selection uses mean validation across these seeds; report corresponding mean
test and sample standard deviation. Test is never used for configuration choice.
The theoretical cell-size-weighted student residual is NOT equated with uniform
loss; summary.csv logs the `m*max_cell/N` imbalance multiplier.

## Pilot and full sweep

Default cases: Cora 5.2% and Citeseer 3.6%.

| Setting | Pilot | Full |
|---|---|---|
| gamma | Cora .01 / Citeseer .1 | .01, .1, 1 |
| T | Cora 1 / Citeseer .2 | .2, .5, 1, 2 |
| mu for mpnn and raw | .1, .3, 1, 3, 10 | same |
| GRIP kl_weight | .1, .2, .5, 1, 2 | same |
| comparison depth / rho | 2 / .5; raw uses depth 0 | same |
| dropout / student seeds | .1, .5, .9 / 0, 1, 2 | same |

Across both datasets: pilot = 30 condensations / 270 student fits;
full = 360 condensations / 3240 student fits. No A100 runtime estimate is claimed.
To compare only mpnn versus GRIP, use `--methods mpnn,grip` (180 / 2160 fits).

After the user's Colab setup cell:

```python
import os, subprocess
subprocess.run(['git', '-C', '/content/GRIP', 'pull', '--ff-only'], check=True)
os.environ['GRIP_SWEEP_PRESET'] = 'pilot'  # 'full' for the joint grid
%run /content/GRIP/colab_distance.py
```

Outputs go to `/content/drive/MyDrive/GRIP_mpnn_identity`, separate from the old
GCN-specific experiment. Source/data/config fingerprints separate methods and
invalidate stale caches. Teacher logits, condensed graphs, stagewise objectives,
assignments, each student run, logs, summary.csv, and validation-selected
best.json are saved. Rerunning the same command resumes missing student runs;
changing repeat from 1 to 3 reuses the condensation and already finished seed.

The Colab cell displays a single updating progress item and one compact final
table (all validation-best ties, ignoring numerical noise at 10 decimal places).
Epoch/objective logs remain in the Drive log file; only the last 20 lines are
printed on failure. `best.json` retains the runner's existing single-winner
selection, while the compact table exposes ties without selecting by test score.
This display-only change does not invalidate the training/condensation caches.

Numerical defaults: representative batch 32, median iterations 30, outer
iterations 20, objective relative tolerance 1e-6. These are computational
controls, not accuracy grid dimensions. The full teacher/features/GCN still
need to fit GPU memory; representative batching alone is not a guarantee for
very large datasets.

Standalone runs:

```bash
python main.py --dataset_name cora --ratio 0.052 --edges mpnn_identity \
  --gamma .01 --T 1 --label_weight 1 --outer_iters 20 --gpu 0

# Short integration check; not a meaningful accuracy experiment
python sweep_distance.py --cases cora:0.052,citeseer:0.036 \
  --methods mpnn,raw,grip --mus 1 --baseline-kl .5 --dropouts .5 \
  --repeat 1 --epoch 20 --eval-every 10 --outer-iters 2 --median-iters 5 \
  --raw-data-dir ./data --output results/mpnn_smoke
```

Tests in `test_partition_mpnn.py` independently check closed-neighborhood
direction/duplicates/isolation, the dense multi-hop formula and adjoint,
K=0 ablation, matching initialization, nonempty monotone descent, batching,
CPU/CUDA agreement, and a common sensitivity-bound example for scalar
SUM/MEAN/MAX/softmax-attention updates with analytically bounded constants.
Those examples do not establish tight constants for arbitrary trained GNNs.

## Cell-size-weighted GRIP student diagnostic

Set `GRIP_SWEEP_PRESET=weighted` before running `colab_distance.py` for a Cora
0.052 GRIP-only full sweep. The grid is gamma {0.01, 0.1, 1}, T {0.2, 0.5, 1, 2},
GRIP KL coefficient {0.1, 0.2, 0.5, 1, 2}, dropout {0.1, 0.5, 0.9}, and student
seeds {0, 1, 2}: 60 condensations and 540 student fits of 1000 epochs.
Results go to `MyDrive/GRIP_cora_weighted_ce_full` with compact validation-best
output and resumable runs.

This explicitly requested diagnostic changes only the student training loss:
`sum_j (n_j / sum_k n_k) * CE(y_j, student_j)`. Counts come from the final
condensation assignment after empty-cell removal; they are not class weights.
The default remains uniform CE, selected by `--student-loss uniform`; the new
mode is `--student-loss cell-size`. The loss mode is in cache identities and
summary rows. Two-layer GCN, hidden size 256, optimizer, teacher, condensation,
and unweighted validation/test accuracy are unchanged.

Compare matching hyperparameters and seeds with the previous uniform full
sweep, as well as each sweep's validation-selected winner. Separate reruns can
have GPU numerical variation; a small difference alone does not establish a
bottleneck. This experiment does not turn the original GRIP bound into a
general-MPNN guarantee.

### Paired confirmation on fresh student seeds

`paired_ce.py` reads the saved weighted sweep's manifest and reuses its single
GRIP artifact at Cora 0.052, gamma 0.01, T 2, coefficient 0.5, condensation seed
0. It fixes dropout at 0.9 and trains uniform and cell-size CE on the exact same
features, labels and identity edges. Default student seeds are 3 through 12
(ten fresh seeds beyond the selection sweep), for 20 fits of 1000 epochs.
Each pair resets the initialization/dropout RNG. GPU kernels may remain
nondeterministic. Original data/split hashes must match the saved artifact.

Runs are cached separately by code, artifact, training configuration and seed;
increasing `--repeat` resumes existing fits. No additional hyperparameter
selection occurs. The compact output reports means/sample standard deviations
and paired differences (weighted minus uniform). `paired.csv` records every
fit, while epoch logs remain in the output directory. Default output is
`MyDrive/GRIP_cora_paired_ce`. This quantifies student-seed variability for one
fixed condensation, not variation across condensations or dataset splits.
