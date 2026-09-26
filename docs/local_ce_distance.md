# Local CE preservation by node distances

`src.local_ce_distance.run_local_ce_comparison` reuses a completed
`node_distance_comparison` result: four distance matrices, teacher probabilities,
and the original frozen student logits. It adds the matched mean embedding
`(alpha I + (1-alpha) P)^L X` with the same neighbor convention and settings as
probability OT. No student training, OT recomputation, or representative selection
is performed. Source graph, teacher and student fingerprints are checked.

Local analysis now includes all architectures listed in the original saved probe
by default, including GIN even if the intermediate representative comparison
only evaluated GCN/SAGE. Optional `models` restricts this explicitly. Additional
distance matrices can be passed without replacing existing comparators; their
hashes and construction metadata enter the result protocol. `health.csv` reports
teacher-agreement and output-confidence diagnostics for every included student.

For teacher target q_i and student output p_i, the directed matrix is

    C_ij = |CE(q_i, p_j) - CE(q_i, p_i)|.

This holds the source target fixed. It is not the difference between each node's
own loss. Although D is symmetric, C generally is not; both directions are
included. This is a directed replacement diagnostic, not a metric-fitting
optimality certificate or a Lipschitz upper bound.

## Calibration and evaluation

Split the same saved 512 nodes once, using seed 2026: 128 calibration nodes,
384 evaluation nodes. No node appears on both sides. These are calibration/eval
splits for distance scaling, not the dataset's train/validation/test split.
Some evaluation nodes may have been sampled for student supervision; their count
is recorded per run. The source teacher already used validation checkpoint
selection. No further labels or model selection are introduced here.

For each distance, divide by the median calibration-pair distance, using the
existing positive-median fallback. For each frozen student and distance fit ONE
nonnegative slope on ALL ordered off-diagonal calibration pairs:

    a* = argmin_(a>=0) sum |a D_ij - C_ij|.

The exact minimizer is a weighted median of C_ij/D_ij with weights D_ij for
positive distances. Zero distances contribute constant error and are retained
in evaluation. If all calibration distances vanish, choose a=0. No slope is
refitted at individual thresholds or using evaluation CE. The auxiliary score
thus tests whether a globally calibrated relationship also holds locally.

## Local evaluation

All methods use the same evaluation nodes but choose their own close pairs.
Diagonal pairs are excluded throughout.

- Quantiles: thresholds at 1%, 2%, 5%, 10% of off-diagonal evaluation distances.
  Computing these thresholds uses geometry only. Include all threshold ties;
  record actual `pair_fraction`, `query_coverage` (fraction of source nodes with
  a selected neighbor) and `zero_distance_fraction`.
- k-NN: k=1,5,10,20 other evaluation nodes per source. Every source has the same
  weight. Exact ties use stable sorting in the common randomly permuted split
  order, not separate random choices for each method or student.

Primary measurements are the selected-pair mean and p95 of C. The auxiliary
measurement is mean |a* D-C| on those same selected pairs. A small prediction
residual alone does not imply safe replacement: C itself may be large.

Relative versions divide each quantity by the mean C over ALL ordered evaluation
pairs for that student. This denominator is shared across methods and cutoffs;
it is not the selected-pair mean. The p95 uses this same mean denominator, not an
all-pair p95. A near-zero denominator yields NaN and a degeneracy flag.
Multiplying any input distance by a positive scalar does not change the results.

Summaries average over saved student runs and report descriptive standard
deviations. These crossed subset/model seeds are not independent replicates.
Paired differences compare each run's relative mean CE change against S²X at
the same cutoff. Different methods' selected pairs can differ; this is a
comparison of the neighborhoods they choose, not errors on identical pairs.

## Outputs

CSV files contain per-run metrics, calibration slopes, summaries, and paired
differences. The protocol stores split node IDs and the source configuration.
`distances.npz` and each `ce_<student>.npy` use the FULL saved probe order, recorded
in `probe_ids.npy`; select rows/columns by the protocol split IDs when reusing
them. No cross-calibration/evaluation pairs enter slope fitting or scoring.

`plot_local_ce_comparison` saves quantile and k-NN figures. Each has rows for
relative mean CE change, relative p95 CE change, and relative calibration MAE;
columns separate architectures. Shading shows run SD, not confidence intervals.

Run `tests/test_local_ce_distance.py` in Colab for matrix orientation,
weighted-median fitting, distance-scale invariance, threshold ties, k-NN coverage,
constant-output handling, and the matched mean operator. No numerical tests or
training are run locally.
