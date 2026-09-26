# Exact computation-tree distances

`src.tree_distance.exact_tree_distances` computes all node-to-node computation-tree
distances, the inner distance used by TMD. It does not compute graph-to-graph TMD's
outer transport, and it does not train a teacher, encoder, or student.

For expansion depth zero, `D_0(i,j) = ||x_i-x_j||_2` and `b_0(i) = ||x_i||_2`.
At each subsequent depth, pad the smaller neighbor multiset with blank trees to
equal cardinality. Match these two multisets using an exact linear assignment:

`D_l(i,j) = D_0(i,j) + weight * min_permutation sum_r C[r, permutation(r)]`.

Real-real costs come from `D_(l-1)`, real-blank costs from `b_(l-1)`. The blank
distances obey `b_l(i) = ||x_i||_2 + weight * sum_u b_(l-1)(u)`, summed over
neighbors. This is unnormalized transport with unit atom masses. Dividing by
the neighbor count or dropping unmatched neighbors would change the method.
Repeated blank atoms make this square assignment equivalent to the aggregated
blank mass in the reference OT implementation. The implementation eliminates
duplicate blank rows/columns exactly: if the right side is larger, first pay
the blank cost for every right atom, then match real atoms with adjusted costs
`real_cost - right_blank_cost`. The left-larger case is symmetric. This yields
a rectangular assignment of size degree(i)-by-degree(j), while preserving all
unmatched-atom costs through the baseline. It is not a rectangular match that
simply discards unmatched neighbors. Roundoff-negative transport totals are
clipped to zero. No entropic or sampled OT is used.

References: [original TMD implementation](https://github.com/chingyaoc/TMD/blob/master/tmd.py),
[SciPy assignment solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).

Our `max_depth=3` means three neighbor expansions, equivalent to the reference
implementation's `L=4` with constant `w=weight`. Features and outgoing neighbor
sets are used directly. Duplicate edges are deduplicated. `self_loops=False`
removes self-loops; `True` adds one per node. No automatic symmetrization or feature
normalization is performed. Cora's Planetoid graph is already undirected. Zero
features may coincide with blank features, so no strict-metric claim is made.

## Storage and restart

Each `distance_l.npy` is an N-by-N matrix. `blank_l.npy` stores its N blank-tree
distances. Default float64 uses about 56 MiB per Cora matrix. Depths 0 through 3
use about 224 MiB for the distance matrices. Only previous-depth distances are
needed for matching; arrays are accessed through memory maps.

Only upper-triangular pairs are solved, with results mirrored. Rows are flushed
before `depth_l.json` checkpoints advance. Interrupted batches are recomputed.
`protocol.json` fingerprints features, edges, dtype, self-loops and weight;
changing these requires a separate directory. Increasing `max_depth` extends
the existing cache. Changing checkpoint frequency or benchmark sample count
does not invalidate completed distances. Use one writer per output directory.

`summary.json` records cumulative computation time and files for each completed
depth. Times include block writes/flushes, exclude the optional preflight benchmark,
and cannot include work lost in an interrupted block. Existing completed depths
are loaded rather than recomputed. A persistent output directory survives runtime
replacement; local `/content` storage does not. Drive memory maps may be slower
than local disk, especially for frequent checkpoints.

## Preflight timing

Before each new matching depth, the implementation times uniformly sampled
distinct node pairs (with replacement), plus a separate sample involving nodes
in the highest degree quantile. Only the uniform sample is extrapolated to all
N(N-1)/2 pairs. `on_benchmark` receives the depth, mean pair microseconds,
high-degree pair microseconds, and estimated minutes for a full depth.

These are CPU matching estimates, not end-to-end guarantees: writes, checkpoint
overhead, changing Colab CPU performance, and sample variability matter. The
progress bar provides actual throughput during the full run. No GPU is needed
for the exact SciPy matching implementation. Benchmark pairs are timing probes;
all pairs are still evaluated in the resulting matrices.

The distance computation uses only graph structure and features. This artifact
alone does not establish usefulness for classification or graph condensation;
it enables those comparisons without an embedding approximation confound.

## Verification

In Colab run `python -m pytest -q tests/test_tree_distance.py` before computing
Cora. Tests compare all pair and blank distances through depth 2 against an
independent recursive exhaustive matcher, check node permutations and duplicate
edges, simulate interruption/resume, reject stale caches, and check self-loops.
No local numerical tests or graph experiments were run during implementation.
