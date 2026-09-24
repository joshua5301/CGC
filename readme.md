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

After cloning the repository and preparing `/content/data/`, install `optuna` in
the notebook, then call:

```python
from src.risk_experiment import run_experiments

results = run_experiments(
    datasets={"cora": [0.013, 0.026, 0.052]},
    output_dir="/content/drive/MyDrive/GRIP_results/risk_v1",
    space={"B": {"low": 0.01, "high": 100.0, "log": True},
           "dropout": [0.1, 0.5, 0.9]},
    n_trials=20,
)
results
```

Search parameters support categorical lists or `suggest_float` keyword dictionaries
for `B`, `dropout`, `lr`, and `weight_decay`. `teacher` can override `kernel`,
`gamma`, `temperature`, and `basis`; otherwise the existing dataset/ratio defaults
are used. `partition` controls `max_sweeps`, `block_size`, `atol`, and `rtol`.
Dataset/ratio pairs use the existing `BUDGET` and `BEST_HYPERPARAMS_DICT` tables.

Each configuration stores its Optuna database, condensed tensors, best settings,
and evaluation CSVs in a separate output directory. Increasing `n_trials` resumes
the same study. `converged=False` reports a solver iteration limit, not convergence.

`B` controls the condensation objective; it does not constrain the evaluated GCN.
The linear-student, mass-weighted-CE bound does not certify this GCN evaluation.
The implementation has only been statically checked locally, not executed or
smoke-tested. See [the derivation and design](docs/risk_bound_partition_design.md).

