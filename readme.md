# Original Repo

https://github.com/XYGaoG/CGC/tree/main

`--sgc_refine` runs GRIP until assignments stabilize, then refines its cells with
a mixture of normalized feature distance and centered-SGC-logit squared distance:
`(1 - beta) * feature_distance + beta * output_distance + kl_weight * label_KL`.
The distance weights sum to one; the KL coefficient is unchanged. Candidates preserve the
initial cell count and are accepted only when a freshly fitted SGC lowers the
original teacher-label CE by more than `--refine_tolerance`. A rejected candidate
is discarded. SGC is used only for refinement. The accepted condensed features
and labels enter the unchanged two-layer GCN evaluation with uniform soft CE,
dataset-specific dropout defaults and original-graph validation/test evaluation.

`--refine_beta 0.1 --refine_rounds 5 --sgc_ridge 0.001` controls the output-distance share,
outer iteration cap and uniform-soft-CE SGC regularization. `--sgc_steps 1000`
and `--grip_steps 1000` cap the solvers; failure of the SGC gradient check or
GRIP assignment stabilization stops execution. Geometric medians retain GRIP's
finite iterations; candidate center descent is also approximate. Accepted
measured SGC teacher risk decreases, without a GCN-risk or global-optimum guarantee.
Beta must lie in [0, 1]; beta=0 uses feature distance only and beta=1 uses output
distance only. Normalization is fixed within each candidate optimization.

## Risk-based partitioning in Colab

`src.risk_partition.risk_partition` builds a partition without GRIP initialization.
It uses bound-derived D² seeding, then proposes node moves on the GPU. Each batch
is accepted only when the recomputed global objective decreases; rejected batches
are split recursively. Features are centered and RMS-scaled during condensation,
then representative features are restored to the original space.

`src.risk_experiment.run_experiments` is an importable experiment API, not a CLI.
It reuses the dataset loader, kernel teacher, and two-layer GCN. Evaluation always
uses uniform soft-label CE. The original-graph GCN computation uses pre-normalized
sparse adjacency; the condensed graph has self-loops only. Optuna selects settings
using validation results, and test is evaluated only for the selected settings.
Final repeats vary the GCN seed on one fixed condensed dataset.

After cloning the repository and preparing `/content/data/`, use one Colab cell:

```python
%cd /content/GRIP
!git pull --ff-only origin main
%pip -q install optuna

import importlib
import src.risk_experiment as experiment
importlib.reload(experiment)

results = experiment.run_experiments(
    datasets={"cora": [0.013, 0.026, 0.052]},
    output_dir="/content/drive/MyDrive/GRIP_results/risk_v1",
    space={"B": {"low": 0.01, "high": 100.0, "log": True},
           "teacher_kernel": ["erf", "relu"],
           "gamma": [0.01, 0.1, 1.0], "T": [0.2, 0.5, 1.0, 2.0],
           "basis": [3000], "dropout": [0.1, 0.5, 0.9],
           "lr": [0.01], "weight_decay": [5e-4]},
    n_trials=100,
)
results
```

Search parameters support categorical lists or Optuna range dictionaries for `B`,
`teacher_kernel`, `gamma`, `T`, `basis`, `dropout`, `lr`, and `weight_decay`.
Ranges use `suggest_int` for `basis` and `suggest_float` for numerical parameters
other than `basis`. Use a categorical list for `teacher_kernel`.
A one-element list fixes a setting. Keys omitted from `space` use the supplied
defaults. `teacher` overrides `teacher_kernel`, `gamma`, `T`, and `basis`, with
`kernel` and `temperature` accepted as aliases; a searched parameter takes
precedence over this fixed default. Otherwise existing dataset/ratio defaults apply.
`partition` controls `max_sweeps`, `block_size`, `atol`, and `rtol`.
Dataset/ratio pairs use the existing `BUDGET` and `BEST_HYPERPARAMS_DICT` tables.

Teacher logits are reused across temperatures. The cache keeps one kernel feature
matrix on the GPU and up to four fitted logit matrices on the CPU per dataset.
Condensed-cache keys include B and every teacher setting, so changing gamma or T
cannot reuse a condensed dataset produced by another teacher configuration.
Selected teacher and student settings are included in `best.json` and `summary.csv`.

Each configuration stores its Optuna database, condensed tensors, best settings,
and evaluation CSVs in a separate output directory. Increasing `n_trials` resumes
the same study. `converged=False` reports a solver iteration limit, not convergence.

`B` controls the condensation objective; it does not constrain the evaluated GCN.
The linear-student, mass-weighted-CE bound does not certify this GCN evaluation.
The implementation has only been statically checked locally, not executed or
smoke-tested. See [the derivation and design](docs/risk_bound_partition_design.md).

## Controlled partition ablation

`src.risk_analysis.run_ablation` accepts a previous `summary.csv` path or a
DataFrame containing dataset, ratio, B, teacher_kernel, gamma, T, basis, dropout,
lr and weight_decay. It freezes these settings; this diagnostic does not run
another hyperparameter search. Risk checkpoints share one optimization trajectory
per partition seed. `risk_terminal` means the final state at convergence or the
specified sweep cap; inspect `converged` before calling it converged. A requested
checkpoint after convergence reuses the converged state and reports its actual sweep.

The optional GRIP comparator uses the same teacher and student configuration with
explicit `kl_weight` and `grip_steps`. It is a controlled comparator, not a separately
tuned benchmark. FAISS initialization now accepts a seed; existing callers retain
the original default 1234. GRIP may remove empty cells, so actual node counts are
reported. Its different objective is not labeled as risk J.

All students use two-layer GCN and uniform soft CE. Defaults use new student seeds
200–209 and validation only. `evaluate_test=True` explicitly enables test evaluation
at each seed's best validation checkpoint; diagnostic checkpoint selection should
remain validation-only. Existing configurations were selected for the original
method, so this is a conditional ablation, not an unbiased algorithm ranking.

Outputs include resumable per-case runs and artifacts, aggregate `runs.csv`,
`history.csv`, `summary.csv`, `paired.csv`, and PNG figures. Paired differences are
in validation percentage points with Student-t 95% intervals across GCN seeds,
computed separately for each partition seed. These conditional intervals do not
measure dataset uncertainty and have no multiple-comparison correction. Figure
error bars show seed standard deviations. Cell mass total variation against uniform
weights describes a possible objective/evaluation mismatch, not its causal effect.
Timing excludes teacher training and includes checkpoint copying; CUDA work is
synchronized before partition timing starts. No local training or smoke tests were run.

## Independent tuning and crossed settings

`src.method_comparison.compare_methods` runs separate Optuna TPE studies for risk
and GRIP with equal trial counts and identical common search spaces, search GCN
seeds, and training schedules. Common parameters are teacher_kernel, gamma, T,
basis, dropout, lr and weight_decay. Method-specific spaces contain B for risk
and kl_weight for GRIP. Search and final GCN seeds must be disjoint. All evaluation
uses two-layer GCN and uniform soft CE; this API never evaluates test accuracy.

The selected teacher/student settings from each study are crossed with both
partition methods. Each method retains its own selected B or kl_weight across
the two settings. Crossed configurations are fixed, with no extra tuning or
selection. Solver caps are identical between search and crossed evaluation;
convergence and actual condensed node counts are reported. Equal trials do not
imply equal wall time or guarantee either global hyperparameter optimum.

`tuned.csv` stores independently selected settings, `cross.csv` the four
method/settings combinations, `cross_runs.csv` paired GCN runs, and `paired.csv`
risk-minus-GRIP validation differences in percentage points with conditional
Student-t 95% intervals. Intervals cover GCN seed variability at one fixed
partition seed, not data splits, partition variability or selection uncertainty.
Validation is reused for tuning; new GCN seeds do not make it held-out data.

`initial_configs` optionally maps risk/grip to lists of dictionaries containing
dataset, ratio and params. Known good settings can be enqueued within the search
space and count toward the same trial budget. Missing parameters are sampled.
Increasing n_trials resumes the stored studies; changing the revision or protocol
creates separate caches. No local training or smoke tests were run.

## Contribution of the moment term

`src.objective_ablation.compare_objectives` accepts selected risk configurations
and evaluates initial, variance-only and combined partitions. It generates and
saves one initial assignment and the post-initialization random-generator state
per partition seed. Both optimizers copy that assignment and restore that state,
so initial membership and per-sweep node order are shared. GPU reductions are not
forced to be bitwise deterministic. Teacher labels are generated once and cached.

The variance-only objective is B² V / 4; the combined objective adds 2 B ||E||.
B, teacher/student settings, solver caps and numerical tolerances stay fixed.
Both use arithmetic feature representatives and mean teacher labels. The shared
initialization still uses teacher labels: this tests the moment term during
optimization, not all uses of labels or a separately tuned k-means baseline.

The objective column reports the method's optimized quantity. The bound_J column
always evaluates B² V / 4 + 2 B ||E||, including for variance-only results. Its
monotonicity is guaranteed by acceptance checks only in the combined mode.
Summary and paired CSVs report validation accuracy in percent and paired differences
in percentage points. The combined-minus-variance row measures the incremental
effect under the fixed selected configuration and iteration budget. The 95%
Student-t intervals cover GCN seed variability separately per partition seed;
they do not establish generalization across partitions or data splits. All runs
use two-layer GCN and uniform CE, with no test evaluation or hyperparameter search.
Interrupted student evaluations resume from saved rows. No local model execution
or smoke tests were performed.

## Uniform-mass transport condensation

`run_experiments(method="transport")` uses `src.uniform_transport` instead of hard
partition moves. The transport plan has row masses 1/N and column masses 1/m.
Representatives are c=m Pi^T H and y=m Pi^T Q. Thus the mass-weighted representative
loss in the linear-student derivation is exactly uniform CE. This addresses the
mass mismatch; it does not establish a nonlinear GCN or test-risk guarantee.

Initialization uses the existing bound-derived D² seeds, followed by a balanced
transport solve with the joint feature/label distortion. Optimization linearizes
J=B² V/4+2B||E|| and uses an entropy-regularized transport oracle followed by
backtracking on the unregularized J. Costs are centered by rows and columns and
RMS-scaled before Sinkhorn. Entropy epsilon controls the oracle, not an added term
in J. Four attempts with progressively smaller epsilon are allowed by default.
Sinkhorn output is rounded to the requested marginals by row/column downscaling
and a residual outer product. Relative marginal errors are checked at 1e-7.
Oracle histories include errors before rounding and the mass filled by rounding.

`partition` accepts max_sweeps (outer updates), block_size (seed assignment),
epsilon (default .05), sinkhorn_steps (200), sinkhorn_tolerance (1e-6),
oracle_retries (4), line_steps (20), atol and rtol. A stalled approximate oracle
does not certify stationarity: status is oracle_stalled or iteration_limit and
converged remains false. J history records accepted decreases only. Diagnostics
include mass_tv and relative row/column marginal errors. Counts are effective
fractional masses N/m, not integer cluster memberships.

The dense float64 plan costs 8Nm bytes; oracle workspaces require several such
arrays. For arxiv at 454 representatives the plan alone is about 0.62 GB. The full
plan is not written to Drive; representative tensors, objective/oracle histories,
settings and marginal diagnostics are saved in the standard condensed artifact.
Evaluation remains two-layer GCN with uniform soft CE. Only syntax and diff checks
were performed locally; run numerical and model evaluation in Colab.

