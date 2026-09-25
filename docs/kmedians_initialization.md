# Paired K-means / Euclidean K-medians initialization

Four experimental initialization names are supported: `uniform_kmeans`,
`uniform_kmedians`, `distance_kmeans`, and `distance_kmedians`.
Uniform seeding samples distinct nodes without replacement. Distance seeding
starts with a uniform node and samples later nodes proportional to their
nearest Euclidean distance, not squared distance. Selected indices are excluded;
if all distances vanish, remaining indices are sampled uniformly.

Both center solvers use the exact same seed function and full-data float64
assignment loop. Thus within a seeding scheme and seed, initial node indices
match. The indices are saved in each candidate artifact. Experimental K-means
updates arithmetic means; K-medians updates approximate geometric medians using
the existing 30-step routine. This is not coordinate-wise L1 K-medians. A
per-cell unsquared-cost comparison retains the best of the old center,
arithmetic mean, and proposed geometric median, avoiding an increase from
an inaccurate median solve. No exact median optimality is claimed.

Empty cells are repaired identically for both solvers by assigning a
maximum-residual node from a cell with at least two nodes. Stable repaired
assignments terminate the feature stage, capped by `feature_steps` (default
100). Feature convergence and iteration counts are recorded. Candidate
selection requires feature-stage convergence when this diagnostic is present.

These options deliberately use a shared experimental feature solver instead
of comparing FAISS float32 K-means against a different-precision median solver.
The existing `kmeans`, `kmeans++`, and `greedy` options are unchanged; the
original FAISS seed 1234 remains a separate reference. Subsequent GRIP updates
and the existing student evaluator are unchanged.

Use `run_initialization_study` with the four initialization names. Compare
discovery validation within each density and partition seed before inspecting
selected-candidate confirmation, which is not an unbiased all-seed method mean.
Paired differences between center solvers hold starting centers fixed; comparing
uniform and distance seeding within K-medians measures the seeding change.
The mean initial SSE is always measured about arithmetic cell means for a
consistent diagnostic; initial_feature measures unsquared distance about GRIP's
geometric-median representatives.
