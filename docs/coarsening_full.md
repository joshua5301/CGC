# Cora fixed-dropout coarsening sweep

`coarsening_full.py` compares GRIP identity edges, raw cell means with fixed
coarsening edges, and the same edges with convex feature optimization. Cora .052
(budget 140), 2-layer GCN hidden 256, dropout .9 and uniform soft CE are fixed.
Training retains the existing 1000-epoch Adam schedule, validation checkpoint
selection and evaluation interval. The feature objective is unchanged; see
[coarsening_features.md](coarsening_features.md) for its theory and limitations.

## Equal search budgets

Each method gets the full Cartesian product:

- gamma: .001, .01, .1, 1
- T: .1, .2, .5, 1, 2, 5, 10
- mu: 0, .1, .2, .5, 1, 2, 5, 10

There are 224 triples, 672 method/settings and 2016 selection fits (seeds 0-2).
Gamma controls the teacher; T its label temperature; mu only the shared GRIP
partition. Mu is not a coefficient in the new convex feature objective.
Each triple shares the exact partition, labels and node count across methods.
Mean/optimized also share the exact coarse edges. Condensation seed is fixed at 0.
Each gamma fits a teacher once and reuses its logits across temperatures; common
kernel features are computed once. Each triple optimizes features once, reusing
all condensed graphs across student seeds. The feature solver has the previous
fixed 1000-step cap, 1e-3 relative gap tolerance and 1e-4 smoothing scale. Its
convergence status and numerical gap are saved; hitting the cap is not convergence.

## Selection and confirmation

Each method selects its own highest mean validation setting. Exact ties prefer
smaller gamma, then T, then mu. Test scores never enter this decision.
`selection.json` records the decision before confirmation fits.

Student seeds 13-22 are used for confirmation, separate from selection and from
the earlier 3-12 comparisons. Confirm the three winners and two matched controls:

- `mean_at_optimized`: coarse_mean at the optimized winner's gamma/T/mu.
- `optimized_at_mean`: coarse_optimized at the mean winner's gamma/T/mu.

Identical configurations are deduplicated: 30-50 confirmation fits. This gives
both a separately tuned comparison and a controlled feature-optimization
comparison at either method's selected configuration. Confirmation never
reselects a configuration by validation or test. Student seed intervals remain
conditional on the fixed data split and condensation seed, not new-dataset or
new-partition evidence. Earlier test results have already been inspected.

Four comparisons are recorded: tuned mean versus GRIP; tuned optimized versus
GRIP; optimized versus its matched mean; and matched optimized versus tuned mean.
Both pointwise paired t intervals and Bonferroni intervals for those four planned
test contrasts are saved. Pairing explicitly uses seed identity.

## Output and restart

The cell displays one updating progress line, five selection/confirmation rows
and two matched contrasts. Full training logs and per-fit JSON go to Drive.
`summary.csv` contains all selection settings and feature diagnostics;
`student_runs.csv` contains all selection/confirmation scores. `manifest.json`
records artifacts/configurations/solver diagnostics, `selection.json` selection,
and `confirmation.json` the paired comparisons. Best-checkpoint model/risk
diagnostics are calculated only for confirmation fits and saved in `runs/*.json`.
They are same-parameter diagnostics, not a generalization certificate.

Teacher logits, condensed bundles and completed student fits are cached with
source/configuration/data hashes. Restarting the identical cell resumes them.
An interrupted feature solve restarts that triple; completed triples are reused.
The new runner uses a separate folder from the previous fixed-setting experiments.
Changing source or solver settings invalidates these caches. This runner does
not import previous experiments' caches, which used different identities.

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

%run coarsening_full.py --output /content/drive/MyDrive/GRIP_cora_coarsening_full
```

CLI grid/seed/epoch options allow small pipeline checks. Defaults above are the
full experiment. Short-epoch local checks are not performance evidence.
