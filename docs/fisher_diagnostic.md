# Local KL/Fisher diagnostic on propagated inputs

This experiment checks an approximation before implementing new clustering.
It does not optimize a Fisher-based condensation objective or test student
retraining transfer. Cora .052 uses GRIP's latest validation-selected triple
gamma=.001, T=10, mu=1 from the coarsening sweep. All are overridable.

## Model and targets

Set Z=P^2 X once with the existing GRIP propagation. Teacher targets F are the
existing relu-kernel teacher fitted on training labels only. GRIP constructs
centers C, cell labels Y and partition a. Three independent students (seeds
23,24,25) train on exactly the same (C,Y), uniform soft CE, hidden 256, dropout
.9, Adam .01 with 5e-4 weight decay, reset to .001 halfway, 1000 epochs.

The student is the existing two-layer GCN with identity edges, hence an MLP.
Both training and evaluation use identity edges: evaluations are g(Z) and g(C),
not P ReLU(P X W0) W1. No ordinary raw-graph GCN accuracy comparison is made.
Freeze the final epoch; no validation/test labels select models or diagnostics.
The prepared teacher/partition hyperparameters came from an earlier validation
sweep, so the entire research process is not validation-independent.

Dropout is disabled for all measurements. Model weights and features are promoted
to float64 for small-KL numerical accuracy. The analytic two-layer evaluation is
checked against the actual identity-edge student (float32 tolerance). This is
a deterministic float64 evaluation of learned weights, not bitwise float32 risk.

## Exact identity and approximations

For p=softmax(g(x)), q=softmax(g(c)), teacher F and dz=g(c)-g(x):

    CE(F,g(c))-CE(F,g(x)) = KL(p||q) + (p-F)^T dz.

The signed correction is retained and verified numerically. KL alone equals the
CE increment only when the correction vanishes, including the special case F=p.
This is a replacement loss at one fixed model, not a bound for retrained risk.

For delta=c-x and J=dg(x)/dx:

    Q_input = .5 (J delta)^T [diag(p)-p p^T] (J delta)
    Q_logit = .5 dz^T [diag(p)-p p^T] dz.

Q_input is the local input-space Fisher approximation to KL. Q_logit uses the
exact nonlinear logit displacement, separating softmax curvature error from
input linearization error. J delta uses the exact two-layer ReLU formula without
constructing feature-by-feature Hessians. ReLU sign changes and base-point kinks
are recorded. Second-order local accuracy does not imply a global bound.

## Probes

For every original node, interpolate from Z_v to its assigned GRIP center using
t=.01,.03,.1,.3,1. Repeat with the raw cell mean in the Z space. Students are NOT
retrained for these interpolations or alternate endpoints. All-node probes use
teacher probabilities only; ground-truth val/test labels are never read for the
loss diagnostics. Report separately for each endpoint, t and student seed:

- Mean actual KL and input-Fisher quadratic.
- Relative MAE = mean(abs(Q-KL))/mean(KL), undefined if mean KL <= 1e-12.
- Relative MAE of Q_logit, Spearman correlation, ReLU gate crossings.
- Mean absolute correction / mean KL, signed correction, CE increment, full
  approximate CE-increment error and exact identity error.

The correction ratio uses mean absolute correction (not cancellation-prone
absolute mean). It may be large even when signed averages cancel. The cell-mean
endpoint is a sensitivity control, not an independently trained baseline.

For 256 deterministic random anchors, compare ALL actual GRIP centers at t=1.
Rank candidates by input-Fisher cost or squared Euclidean distance and compare
against exact KL. Save per-anchor cost matrices; report within-anchor Spearman,
top-choice agreement (accepting exact-KL ties) and excess KL of the chosen center
over the best exact-KL center. This is geometric candidate ranking only: it does
not include label-KL penalties, cell capacities or nonempty-cluster constraints.

The screen shows just the GRIP endpoint table and candidate-ranking summary.
Means across seeds are descriptive, not independent-node confidence intervals.
Small-step accuracy alone is insufficient: t=1 and the all-center ranking are
the relevant stress tests for clustering. Good rankings still do not certify
transfer to a newly trained student or genuine-label generalization.

## Files and Colab

`summary.json` contains all per-seed results and model-fit diagnostics;
`probes.csv` and `rankings.csv` provide tables. `diagnostics/*.npz` stores individual
node probes and all-center candidate costs. `models/*.pt` stores student weights.
`manifest.json` records exact settings and paths. Logs include training output.
Teacher/partition, students and per-seed diagnostics resume from completed files.
Hashes include source, dataset, settings and saved model/artifact bytes.

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

%run fisher_diagnostic.py --output /content/drive/MyDrive/GRIP_cora_fisher_diagnostic
```

Local checks use only tiny synthetic tensors/graphs: KL identity, autograd
Hessian agreement, local convergence, rank ties, and end-to-end cached resumption.
Actual dataset training is left to Colab.
