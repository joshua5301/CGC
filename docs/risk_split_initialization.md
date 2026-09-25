# Risk-driven divisive initialization

`risk_partition(..., init='split')` is available for the original combined
objective only. `init='surrogate'` remains the default and preserves the previous
weighted-space D-squared initializer. Features use the existing centering and
RMS normalization; teachers and the objective are unchanged.

Start with one cell. Each splittable cell generates cuts at 25%, 50%, and 75%
of sorted projection ranks. Projection directions are an approximate principal
feature direction, an approximate principal teacher-label direction (eight power
iterations each), and two random directions in the weighted feature/label
space. Candidates are cached while their parent cell is unchanged. They are
heuristic proposals, not an exhaustive search over all bipartitions.

For child mean differences a and b and weight w=n1*n2/(N*(n1+n2)),
V_new=V-w*||a||^2 and E_new=E-w*a*b^T. All cached candidates are rescored against
the CURRENT global E at every step. The lowest exact combined-objective delta
is chosen. No stale ranking is reused across splits. Full sufficient-statistic
recomputation verifies each selected delta with a numerical tolerance.

If every candidate increases J, the least increasing candidate is used to reach
the exact requested budget. Thus splitting is not claimed to be monotone.
Artifacts record every split's objective, delta, and number of positive deltas.
All cells are nonempty by construction. Subsequent node moves use the existing
solver without changing its objective or acceptance rule.

`move_seed` optionally separates node-move ordering from initialization RNG.
The comparison runner uses 100000+partition_seed for both methods. This keeps
the per-sweep node permutations identical while both methods continue, unlike
sharing a generator whose state depends on initialization. Existing callers
with move_seed=None retain the old RNG behavior. Therefore comparison results
need not exactly reproduce historical runs with coupled RNGs.

`compare_risk_initializations` fixes teacher, B, dropout, training settings, and
student seeds for each density. It saves teacher outputs, condensed artifacts,
and every validation run for resumption under the same protocol. No test score
is computed. Both methods use two-layer GCN with uniform soft-label CE and the
existing evaluator. It reports initial/final J, V, moment norm, convergence,
timing, and paired validation differences. Partition-level means are the units
for partition variability; student repeats are not treated as independent
partition samples. Fixed historical Risk settings favor the existing method's
tuning context; this is an initialization comparison, not independently tuned
method benchmarking.

Default experiment: Citeseer historical Risk settings, five partition seeds,
ten shared student seeds, and at most 100 refinement sweeps. Nonconverged
results remain explicitly flagged. Larger graphs may make divisive seeding
expensive; this implementation is intended first for Cora/Citeseer.
