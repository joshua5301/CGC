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

## Corrected uniform-CE hard partition

`run_experiments(method="corrected")` retains variable-size hard cells and optimizes
Ju=B²(V+C)/4+2B||Eu||. C is the size-weight mismatch weighted by squared augmented
centroid norm; Eu uses uniform representative weights and class-centered labels.
An appended constant coordinate accounts for affine logits and label-marginal
error, and is removed from the output features. This is a bound-derived correction,
not a free balance coefficient. See [the derivation](docs/uniform_ce_correction.md).

The method shares initialization and the exact batch acceptance mechanism with
risk partitioning. It adds no new sweep dimensions. Existing B, teacher and student
settings may be fixed or searched through the same Optuna interface. Summary files
include variance_term, correction_term, moment_term and mass_tv; the first three
sum to J_final. The first few candidate move deltas are verified against complete
objective recomputation at runtime in Colab. Local checks are syntax-only.

## Optional mass-weighted student CE

`run_experiments(loss_weighting="mass")` is an explicit diagnostic alternative to
the default uniform loss. Each cell CE is weighted by counts/counts.sum(), with
no extra division by the number of cells. The same weighting is used during
Optuna search and final student repeats. Student architecture stays two-layer GCN;
validation/test accuracy remains the ordinary node accuracy. Partition objectives
and initialization are unchanged. Use method="risk" for the original uncorrected
partition when studying its mass-weighted training assumption.

The weighting is stored in protocol, cache identity and summary, so uniform and
mass experiments cannot reuse student scores. Direct calls to _train_student with
mass weighting must supply counts; otherwise they fail rather than silently use
uniform CE. All existing ablation/comparison APIs retain uniform CE by default.
Comparisons of independently tuned best results measure the full training protocol;
attributing a difference solely to weighting requires matching search settings or
evaluating the same condensed artifact under both losses. Local verification was
limited to syntax and diff checks, without model execution or smoke tests.

## GCN-aware hard partition and label calibration

`run_experiments(method="gcn_aware", aware={...})` refines an ordinary risk hard
partition using frozen GCN probes and calibrated soft labels. Student loss must
be uniform CE. The first stage uses init_sweeps risk sweeps (default 30); setting
init_sweeps=0 starts from the existing D² initialization, never from GRIP.

Each probe trains a two-layer GCN on original train labels for a fixed epoch count.
Validation/test labels and scores are not used. With the first layer frozen,
original hidden features are A ReLU(A X W0+b0), whereas synthetic features are
ReLU(C W0+b0). Matching uses dropout-free forwards and includes the final bias via
an appended constant. A regularized head is fitted to the original teacher Q with
LBFGS. The reported head_stationarity_max diagnoses the finite head fit; it is
not assumed to be an exact optimum. Probe seeds must differ from evaluation seeds.

The objective is the mean squared Frobenius difference of last-layer CE gradients
across probes, with ordinary uniform averaging on each dataset. The same head
regularizer cancels in the gradient difference. Fixed-feature label calibration
uses projected gradient descent on a convex quadratic with row-simplex constraints.
An eigenvalue-based step bound and a Frank-Wolfe gap provide its stopping rule.
label_converged and label_gap report whether the inner tolerance was attained.
No KL or arbitrary label-anchor penalty is added.

Partition proposals use the derivative of this actual objective with respect to
centroids and the first-order effect of moving a node. Each batch is fully rescored
with labels fixed, accepted only on a decrease, or recursively split. Label
calibration is repeated after each round. This preserves nonempty cells and
arithmetic feature means; labels are no longer forced to be cell means. Proposal
scores are approximations, while acceptance scores are exact for the frozen-probe
objective up to floating-point error. This is not exhaustive move enumeration.

partition.max_sweeps controls refinement rounds; partition.block_size controls
proposal batches. aware options include init_sweeps, probe_seeds ([1000,1001]),
probe_epochs (100), probe_lr (.01), probe_dropout (.5), head_ridge (.001),
head_steps (200), label_steps (1000), label_tolerance (1e-8), and proposal_nodes
(8192). At most proposal_nodes random nodes are considered per round, against all
destination cells. Increasing this to the dataset size considers all source nodes.
A no-improvement round is proposal_stalled, not a local/global optimality certificate;
converged remains false. The iteration cap is reported separately.

J_initial/J_final now mean the gradient-matching objective, identified by
objective_name; they cannot be compared numerically to earlier risk J. Artifacts
store frozen probe states, head diagnostics, label gaps, objective history and final
assignments. Final GCN training still starts from scratch on representative features
and calibrated labels with uniform CE. No local training or smoke tests were run;
only AST parsing and diff checks were performed. The theoretical claims and limits
are in [the analysis](docs/gcn_aware_improvement_analysis.md).

## Covariance Frobenius risk partition

`run_experiments(method="risk_fro")` optimizes
J_F=B²||S||F/4+2B||E||F, where S is within-cell feature covariance and E is the
original mass-weighted feature/label moment error. This replaces the scalar trace
relaxation in the original risk method. For the same partition and B, the spectral
bound is at most J_F, and J_F is at most the original trace bound. See
[the tightening derivation](docs/risk_bound_tightening.md).

The new mode preserves the original feature normalization, D² initialization,
mean features/labels, fixed nonempty cell budget and moment term. It adds no new
search parameter. B should still be selected using validation; changing the feature
term changes its practical tradeoff. This implementation is the Frobenius variant,
not the spectral-eigenvalue or joint trust-region variant described in the analysis.

S is computed as X^T X/N minus the weighted centroid second moment in float64.
Candidate moves use its exact rank-two Frobenius norm change and the existing
moment-error update. Batches are accepted after full objective recomputation and
bisected otherwise. verify_deltas=True checks the first eligible moves against full
recomputation during Colab execution. End-of-run checks enforce ||S||F<=trace(S)
within numerical tolerance. Keeping covariance costs O(d²) memory and additional
matrix products, so this is more expensive than trace-only partitioning.

Summary columns include covariance_fro, covariance_trace, feature_term,
moment_term, trace_bound and trace_to_fro. feature_term+moment_term=J_final;
trace_bound evaluates the old bound on the NEW partition, not an old experiment.
trace_to_fro is the feature-term tightening factor at that partition. A smaller
upper bound does not guarantee higher GCN accuracy. The original theorem's
mass-weighted constrained-linear-student assumptions remain; evaluation defaults
to two-layer GCN and uniform CE. Local verification was limited to AST parsing
and diff checks. No local training, imports of model modules or smoke tests ran.

# GRIP-cost greedy initialization

`grip_init='kmeans++'` selects feature-space centers with vanilla D-squared
sampling (`sklearn.cluster.kmeans_plusplus`, `n_local_trials=1`), then supplies
them to the same FAISS K-means training used by `kmeans`. FAISS iteration and
subsampling settings, subsequent GRIP updates, and student evaluation are
unchanged. `grip_seed` controls the seeding. This is feature K-means++ followed
by K-means, not the feature-plus-KL greedy initializer. The K-means++ guarantee
concerns feature squared distortion, not final GRIP cost or validation accuracy.

GRIP sweeps use `grip_seed=1234` by default, matching the implicit FAISS seed
in the original `main.py` path. Earlier sweeps passed the experiment seed
(usually 0) into FAISS instead. Set `grip_seed=0` to reproduce those sweeps.
Teacher generation still uses `seed`; student runs use their explicit seed
lists. Greedy initialization is deterministic for fixed features and labels
and does not use `grip_seed`. The protocol and summary record this setting.

For exhaustive search use `run_experiments(..., search='grid', space=GRID)`.
Every value in `GRID` must be a nonempty list. All Cartesian-product combinations
are evaluated; `n_trials` does not limit grid search. This path does not import
or call Optuna. Each completed combination is saved atomically and skipped on
resume with the same protocol. Changing the grid, its key order, revision, or
evaluation settings creates a separate run. Search uses mean validation accuracy;
the selected configuration uses the existing final validation/test evaluation.

Use `run_experiments(..., method='grip', grip_init='greedy')` for exact greedy
node selection under GRIP's feature-plus-KL cost. `grip_init='kmeans'` retains
the original initialization. See [details](docs/grip_greedy_initialization.md).

