# Convex relaxation of the original Risk objective

For centered, RMS-normalized features X and fixed teacher probabilities Q, let
Z=A diag(n)^(-1) A^T for a nonempty m-cell hard partition A. The original objective is exactly

    J(Z) = alpha/N tr(X^T(I-Z)X) + beta/N ||X^T(I-Z)Q||_F,
    alpha=B^2/4, beta=2B.

Relax the partition constraint to Z PSD, Z elementwise nonnegative, Z1=1,
tr(Z)=m. This is a convex conic problem (SDP plus a norm cone), following the
Peng–Wei partition-matrix relaxation. The moment extension here follows from
our objective; existing k-means approximation guarantees do not automatically apply.

`src/risk_sdp.py` solves it with CVXPY/SCS in float64 on CPU. Auxiliary variables
Y=ZQ and R=X^T(Q-Y)/N avoid directly expanding each moment entry into N^2 coefficients.
There is no feature truncation or low-rank factorization of the optimization variable.
The dense PSD projection remains expensive. `max_nodes` is an explicit size guard,
not automatic subsampling. A100 is used for teacher/partition/student work, not this SCS solve.

## Numerical lower bound

The solver's primal objective, including an `optimal_inaccurate` result, is not
reported as a lower bound. For any ||U||_F<=1, define

    c = alpha ||X||_F^2/N + beta <U,X^TQ>/N,
    C = (alpha XX^T + beta sym(XUQ^T))/N.

For any y,t,M with M>=0 and D=sym(y1^T)+tI-C-M PSD,

    J(Z) >= c - 1^T y - m t

for every feasible relaxed Z. We recover U from the moment equality's dual,
project it inside the unit ball, symmetrize and clip M to be nonnegative, and
increase t by max(0,-lambda_min(D)) plus a numerical margin. Taking the maximum
with zero is valid because the relaxed objective is nonnegative: feasible Z
has spectrum in [0,1]. Dual arrays are saved for independent inspection.

This is a numerically repaired dual bound, not an interval-arithmetic certificate.
The safety margin covers ordinary eigensolver error heuristically; it is not a
formal floating-point error proof. Invalid finite checks or a bound exceeding an
observed feasible partition beyond tolerance raise errors rather than hiding the gap.

`gap_upper = J_hard - lower_bound` bounds hard-partition suboptimality in exact
arithmetic. A small gap is informative; a large gap can reflect optimization error
or relaxation looseness. `relaxed_value` is a diagnostic only. Row, trace,
nonnegative, PSD and moment residuals and solver status are saved separately.
`optimal_inaccurate` is retained and visibly reported, never relabeled optimal.

## Discretization and comparison

Top-m eigenvectors scaled by square-root eigenvalues provide a rounding embedding.
D-squared seeding and nearest-seed assignment generate nonempty partitions; there
is no Lloyd refinement in the rounding. The existing exact-Risk move solver then
refines each partition. Rounding need not preserve the relaxed objective value.

`run_sdp_study` compares surrogate initialization and SDP rounding using the same
number of partition seeds and the same subsequent move seeds. This equalizes
candidate counts, not wall-clock cost: SDP cost is reported separately. The best
partition per method is selected by J only. Optional full-data student evaluation
uses the existing two-layer GCN, uniform CE and validation-selected epochs. No test
labels are used for selection; this diagnostic does not evaluate test accuracy.

Default `sample_nodes=256, sample_clusters=8` is an explicitly separate subset
problem. Teacher probabilities and propagated features come from the full graph,
then the sampled feature rows are renormalized for both optimizers. The subset
indices, N and m are saved. These bounds are NOT bounds for the full dataset;
the configured dataset ratio is only metadata for that diagnostic. Full-graph
student evaluation is disabled in subset mode. Use `sample_nodes=None` and raise
`sdp['max_nodes']` explicitly for the original full problem and its original budget.

Protocol-hashed directories store the teacher, relaxation, dual witness, partitions,
per-seed results and optional student evaluations. Completed stages are reused.
Changing code revision or settings creates a new case.

References:
- https://optimization-online.org/wp-content/uploads/2005/04/1114.pdf
- https://arxiv.org/abs/2104.11542
- https://www.cvxpy.org/tutorial/constraints/index.html

Tests are intended for Colab: `python -m pytest -q tests/test_risk_sdp.py`.
They compare the matrix objective with exhaustive hard partitions, check repaired
dual witnesses against exhaustive optima, cover m=1/N, and verify rounding and
Risk refinement. Local development does not execute these numerical tests.

## Full-data grid search

`src.risk_sdp_grid.run_sdp_grid` exhaustively searches teacher_kernel, gamma, T,
basis, B, dropout, lr and weight_decay. Each teacher/B combination uses the full
dataset and its original BUDGET entry, never a subset. Dropout/lr/weight_decay
combinations reuse the same SDP and partitions. Teacher probabilities are cached
across B values. Every method selects a partition by minimum J over the configured
partition seeds, then selects hyperparameters by mean validation over search seeds.
Only after all candidates finish and selected.csv is saved are the winning settings
evaluated with separate final student seeds, including test accuracy at the
validation-selected epoch. Final repeats measure student randomness conditional
on the selected partition, not variability over new partitions.

The default comparison includes surrogate and SDP initialization, independently
selected on validation under the same grid. This is not a GRIP baseline. Results
and individual student runs resume from protocol-hashed files. SCS time_limit_secs
limits solver time, not CVXPY compilation, rounding, refinement or evaluation.
An optimal_inaccurate result remains eligible and its status is reported; the
returned hard partition is feasible, but the relaxation may not have converged.
