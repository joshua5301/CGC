# Fixed GCN coarsening with convex feature fitting

Run `coarsening_grip.py` after the existing Colab Drive/data setup. This compares
three condensed graphs with exactly the same GRIP assignment, cell count and
teacher-mean soft labels:

1. `grip`: original H2 geometric medians, identity edges.
2. `coarse_mean`: raw-X cell means, fixed coarsening edges.
3. `coarse_optimized`: same coarsening edges, optimized features.

The key controlled comparison is 3 versus 2. Comparison with 1 changes both the
features and edges. Assignment and edges are NOT jointly optimized in this version.

## Coarsening and normalization

P is the actual source-to-target PyG `gcn_norm` operator on the original graph.
Let Atilde contain the original raw weights and the remaining self loops inserted
by exactly the same PyG rule. Pool raw weights, including loops/internal edges:

    Ac = S^T Atilde S.

Q is `gcn_norm(Ac)`. The student receives Ac's raw weighted edges, including its
existing positive diagonal, and normalizes once inside the unchanged GCNConv.
There is no extra identity added to Ac, no row normalization, and no second
normalization of Q. This Q is a specified coarsening rule, not a claimed risk-
optimal operator. Unit tests compare both P and Q to actual GCNConv forwards,
including weighted existing self loops and isolated nodes.

## Feature objective and solver

    omega = P^T 1
    D_prop(C) = mean_v omega_v ||(PX)_v - (QC)_[a(v)]||_2
    minimize D_prop(C), subject to ||C_j||_2 <= R_X = max_v ||X_v||_2.

P, Q, S and PX are fixed. Initialization is raw-X cell means. No labels, validation
scores or test scores enter this optimization. This is a convex feature problem;
it does not add a student loss term or a new KL/structure tuning coefficient.

The float64 solver uses a smooth upper approximation sqrt(||r||^2+epsilon^2),
monotone restarted accelerated projected gradient, and backtracking majorization.
Default epsilon is 1e-4 times the initial weighted mean residual scale. The upper
approximation exceeds D_prop by at most epsilon*mean(omega). The solver retains
the iterate with smallest raw objective, including initialization.

For gradient g of the smooth objective and the product of radius-R_X balls,

    gap = <g,C> + R_X sum_j ||g_j||_2

upper-bounds smooth suboptimality by convexity. Consequently gap plus the smoothing
slack upper-bounds raw-objective suboptimality (up to floating-point arithmetic).
This certificate is recomputed for the returned iterate. Default stopping is at
1e-3 of the initial raw objective, with 1000 steps maximum. `max_steps` means the
requested gap tolerance was not reached; it does NOT mean an optimum was found.
The certificate refers to the float64 feature problem, not the float32 student,
the nonconvex training problem, risk, or generalization.

## Same-parameter GCN bound

For two-layer ReLU GCN logits, eval mode (dropout off), including learned biases:

    H = ReLU(P X W0 + b0),   Hc = ReLU(Q C W0 + b0)
    z = P H W1 + b1,         zc = Q Hc W1 + b1.

Hard S copies rows, so biases match and pointwise ReLU commutes with S. Then

    mean ||z_v-zc_[a(v)]||
      <= ||W1||_2 [||W0||_2 D_prop + M1 D_struct],
    M1 = max_j ||Hc_j||_2,
    D_struct = ||PS-SQ||_(1,1)/N.

D_struct is unchanged between coarse_mean and coarse_optimized. A common M1
upper bound follows CONDITIONALLY from a model-class norm bound and the feature
ball: M1 <= kappa0 * max_j sum_k Q[j,k] * R_X + beta0 when ||W0||_2<=kappa0 and
||b0||_2<=beta0. These model-class constraints are not imposed on the standard
GCN student. D_prop improvement alone is therefore not a certified decrease of
the full risk or of the post-fit bound (student weights and actual M1 may change).

## Risk diagnostics

At each fit's strict-best-validation checkpoint we restore that model and measure
original/coarse hidden representations, logits, weight norms, and the above bound.
F is the full-node teacher and Y the shared cell average. With pi_j=n_j/N:

    R_F(original) = L_uniform(Y,zc)
        + mean_v [CE(F_v,z_v)-CE(F_v,zc_[a(v)])]
        + sum_j (pi_j-1/m) CE(Y_j,zc_j).

This is an exact CE identity, not a generalization approximation. Both the signed
mass correction and prediction CE difference are saved, along with

    R_F <= L_uniform + sqrt(2)*logit_bound + signed_mass_correction.

It is a post-fit same-parameter teacher-risk bound. It does not certify true-label
test risk, an optimization gap for the trained student, or all MPNN architectures.
Diagnostics use the actual float32 features fed to GCN; solver certificates use
the stored float64 optimized features. Tiny normalization/rounding differences
are checked with numerical tolerances. Degenerate zero-validation runs retain
the original protocol's zero results and omit checkpoint diagnostics.

## Evaluation protocol

Default Cora .052, gamma=.01, T=5, mu=2, dropout=.9. T/mu come from the user's latest
full sweep's GRIP **selection-validation** winner, not from best test accuracy.
Options `--temperature` and `--coefficient` can select another fixed initialization,
but this runner does not tune them. The mu coefficient is used only to construct
the original GRIP partition, not in the new convex feature objective.

The unchanged model is GCN-2-256 with uniform soft CE, Adam lr=.01, wd=5e-4,
1000 epochs, optimizer reset to lr*.1 halfway, eval every 10 epochs and first
strict-best-validation checkpoint. Ten paired student seeds 3-12 give 30 fits.
Those seeds were used in earlier experiments; they are not advertised as untouched
confirmation seeds. CIs are pointwise paired t intervals, conditional on fixed
data, partition and labels. Tests check parity with existing model_training.

All details go to a Drive log. The cell shows one progress item, three diagnostic
rows (including mean best-checkpoint logit gap and student CE), solver status/gap
and three evaluation rows. Saved artifacts:

- teachers/*.pt: fixed teacher targets.
- condensed/*.pt: shared partition/labels, raw weighted edges, features, solver history.
- runs/*.json: per-seed scores and prediction/risk diagnostics.
- manifest.json, summary.json, student_runs.csv.

Caches fingerprint source, data, settings and artifact bytes. Re-running the cell
resumes. Optimization is atomic at completion; an interruption during optimization
restarts that one feature solve, while completed student fits are reused.

```python
import os, subprocess
subprocess.run(["git", "-C", "/content/GRIP", "pull", "--ff-only"],
               check=True, stdout=subprocess.DEVNULL)
os.chdir("/content/GRIP")
%run coarsening_grip.py
```

Local short-epoch runs are pipeline checks, not evidence of test performance.

## Citeseer 3.6%

`--dataset citeseer --ratio .036` uses the existing normalized-feature Citeseer
loader, 120-node budget and erf teacher kernel. Defaults are gamma=.01, T=.2,
mu=.1: the teacher/partition settings of the first reported validation tie from
the earlier Citeseer sweep. Dropout stays .9 for the current coarsening protocol;
this combination is not claimed to be a newly tuned Citeseer winner. All three
variants share these settings and student seeds 3-12 (30 fits total).
The Cora defaults remain unchanged.

```python
import os, subprocess
subprocess.run(["git", "-C", "/content/GRIP", "pull", "--ff-only",
                "https://github.com/joshua5301/GRIP.git", "main"],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
os.chdir("/content/GRIP")
%run coarsening_grip.py --dataset citeseer --ratio 0.036 --gamma 0.01 --temperature 0.2 --coefficient 0.1 --dropout 0.9 --repeat 10 --seed-start 3 --output /content/drive/MyDrive/GRIP_citeseer_coarsening_features
```

Completed fits resume from the output folder; Cora results use a separate folder.
