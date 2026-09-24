# Frozen Fisher clustering followed by independent student training

Run `fisher_grip.py` after `fisher_diagnostic.py`. This tests whether the diagnostic's
better local prediction-KL geometry improves a newly trained student, not just
the frozen models from which that geometry was estimated. Actual Cora training
is intended for Colab; local checks use only tiny synthetic graphs/tensors.

## Source and experimental controls

Reuse the diagnostic's exact Z=P^2 X, teacher probabilities F, GRIP centers C0,
cell labels Y0, assignment a0 and frozen student checkpoints. By default these
are Cora .052, gamma=.001, T=10, mu=1, dropout=.9, metric seeds 23-25. Source,
dataset, runtime, model hash and diagnostic report identities are verified.
No teacher or metric student is silently retrained. If the diagnostic artifacts
are missing or incompatible, finish/rerun the diagnostic cell first.

Three variants independently start from the same source:

1. `grip`: original centers, labels and assignment.
2. `euclidean_refined`: alternating squared-Euclidean geometry and label KL.
3. `fisher_refined`: the same optimizer with the averaged frozen input-Fisher metric.

Refinement keeps the original nonempty node budget and recomputes every cell
label as its mean F. Condensed edges remain identity. No student loss weights,
new edges, teacher temperature tuning or validation-based geometry selection.
The Euclidean control uses squared distances, whereas original GRIP uses
geometric-median Euclidean distances. Thus Fisher versus this control isolates
the geometry under the same quadratic refinement procedure; Fisher versus GRIP
compares the complete methods. There is no claim of identical original-GRIP
normalization or updates in the Euclidean control.

## Objective

For frozen students g_r, p_rv=softmax(g_r(Z_v)), input Jacobians J_rv:

    M_v = mean_r J_rv^T [diag(p_rv)-p_rv p_rv^T] J_rv
    D_F(v,c) = .5 (c-Z_v)^T M_v (c-Z_v)
    D_E(v,c) = .5 ||c-Z_v||^2.

Each method uses its own fixed geometry scale s_D=mean_v D(v,mean(Z)), floored at
1e-12; the shared label scale is s_KL=mean_v KL(F_v||mean(F)), floored at 1e-12.
Neither scale changes during refinement. For D=D_F or D_E:

    J = mean_v D(v,C_[a(v)]) / s_D
          + mu * mean_v KL(F_v||Y_[a(v)]) / s_KL
    subject to ||C_j||_2 <= max_v ||Z_v||_2.

Mu is inherited from the source triple (default 1) for this fixed experiment,
not claimed to be an exact risk-bound coefficient. The row-ball limits center
excursions in singular/poorly identified Fisher directions and is applied to
BOTH refinements. It does not enforce a local trust region, preserve all ReLU
gates or make the quadratic a global KL/CE bound. No ridge term is added.

The exact correction (p-F)^T (g(c)-g(x)) from the CE decomposition is NOT added
to the clustering objective. This experiment tests predictive-KL preservation
and transfer, not direct minimization of the original teacher CE. Signed CE
replacement changes and correction magnitudes are saved as diagnostics.

## Solver

Precompute low-rank factors B_v with M_v=B_v^T B_v. For a three-model Cora
ensemble they have shape N x (3*7) x d, roughly 0.65 GB in float64 for the
default dimensions, rather than N dense d-by-d matrices. The factors are fixed
at the original Z nodes, not recomputed as representatives move.

Each outer iteration:

1. Compute all node/center quadratic plus frozen-label KL costs in batches.
2. Sequential strictly improving assignments protect nonempty cells and retain
   ties, with deterministic node order.
3. Update Y to exact cell mean F.
4. Update C with monotone accelerated projected gradient on its fixed-assignment
   convex quadratic over row balls. Backtracking and acceleration restarts keep
   the geometry nonincreasing. Default cap is 100 steps, numerical gap tolerance
   1e-4 times the starting geometry value for that block.
5. Check full-objective descent; record moves, terms and center solver gap.

Default cap is 20 outer iterations. Tiny improvement with no moves stops early;
this is not a certificate of a global partition optimum. The center first-order
gap is only a numerical convex-block certificate. Raw surrogate objectives
across the two geometries have different scales and are not directly comparable.

At completion, evaluate actual KL, the quadratic, the signed teacher-CE
replacement change and correction magnitude under the SAME frozen ensemble
for all three variants. These common diagnostics can reveal whether quadratic
descent decreased actual KL or merely exploited the approximation.

## Independent retraining and evaluation

Train fresh GCN-2-256 students on each identity-edge condensed graph with uniform
soft CE, seeds 26-35. Dropout is inherited (.9), Adam .01, weight decay 5e-4,
reset to .001 halfway, 1000 epochs, evaluate every 10 epochs. Each variant has
10 training trajectories: 30 total. Metric and evaluation seeds must be disjoint.

Each trajectory is evaluated in TWO domains with its own first strict-best
validation checkpoint, so the scores need not use the same epoch:

- `graphless`: input Z=P^2 X, identity edges. This is the primary model class of
  the Fisher diagnostic. No further graph propagation is applied.
- `gcn_transfer`: original X and original graph, the existing raw-graph GCN
  evaluation protocol. This checks transfer to the actual nonlinear GCN.

Both use the same trained weights along each trajectory; evaluating the second
domain does not alter optimization or add fits. It is not valid to mix the best
validation score of one domain with the test score of the other. Each checkpoint
is saved separately. No test score selects a checkpoint or changes condensation.
Tiny tests verify parity against separate single-domain training helpers.

Per-fit results include original teacher CE, uniform condensed CE, and for the
graphless checkpoint, the held-out student's actual KL/quadratic/correction
diagnostics. These distinguish frozen-metric improvements from new-model
improvements. Full models are saved for later investigation.

Paired student-seed 95% t intervals are conditional on the fixed dataset split,
teacher, source partition, and frozen ensemble. Four planned Fisher contrasts
(versus GRIP and Euclidean, in each domain) also receive Bonferroni intervals.
No new-partition/generalization theorem is inferred from these seed intervals.

## Output and resume

The screen shows one updating progress item, three frozen-ensemble diagnostic
rows, and the three-method result table in each evaluation domain. Detailed
logs go to Drive. `summary.json` includes CIs and averaged diagnostics;
`student_runs.csv` contains scores; `runs/*.json` contains per-fit diagnostics;
`models/*.pt` contains both selected checkpoints; `condensed/*.pt` contains
centers/labels/assignments and optimization history; `manifest.json` ties them
to their configurations and source artifacts.

Completed condensed graphs and runs resume. An interrupted refinement restarts
that variant from the same GRIP initialization. Changing code/settings alters
cache keys. The source diagnostic files are read-only and the new experiment
uses a separate output folder.

```python
import os, subprocess

update = subprocess.run(
    ["git", "-C", "/content/GRIP", "pull", "--ff-only",
     "https://github.com/joshua5301/GRIP.git", "main"],
    capture_output=True, text=True,
)
if update.returncode:
    raise RuntimeError(update.stdout + update.stderr)
os.chdir("/content/GRIP")

%run fisher_grip.py --diagnostic-dir /content/drive/MyDrive/GRIP_cora_fisher_diagnostic --output /content/drive/MyDrive/GRIP_cora_fisher_clustering
```
