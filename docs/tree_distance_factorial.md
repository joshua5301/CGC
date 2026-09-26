# Depth-one feature normalization / OT aggregation experiment

This experiment follows the raw Cora tree-distance diagnostic. It holds expansion
depth at one, reuses the original raw-feature distance cache, and compares:

| Features | Neighbor matching cost | Method |
| --- | --- | --- |
| Original | Sum | `raw_sum` |
| Original | Mean | `raw_mean` |
| Per-node L2 normalized | Sum | `l2_sum` |
| Per-node L2 normalized | Mean | `l2_mean` |

Zero feature rows remain zero. Normalization uses no labels. Original-feature
Euclidean (`raw_euclidean`), normalized-feature Euclidean (`l2_euclidean`), the
unchanged original GRIP `S^2 X` distance, and a degree-only distance are controls.
The GRIP baseline is not renormalized in this experiment.

Let `D0` be root Euclidean distance, `D1_sum = D0 + w * OT_sum`, and
`m(i,j) = max(1, degree_i, degree_j)`, with degrees matching the cache's self-loop
convention. The mean variant is exactly

`D1_mean(i,j) = D0(i,j) + (D1_sum(i,j) - D0(i,j)) / m(i,j)`.

Only the neighbor transport term is divided; root distance is unchanged. At
depth one, dividing the objective by a positive pair-specific constant leaves
its optimal matching unchanged. Hence mean costs can be derived exactly from
sum costs, with no second OT solve. Tiny roundoff-negative differences between
`D1_sum` and `D0` are clipped to zero. If both nodes are isolated, OT is zero.

This derivation is specific to depth one. At greater depths, changing to mean
aggregation also changes lower-level costs and must be incorporated recursively.
Dividing an existing deeper final matrix is not the recursive mean variant.
The mean variant still penalizes blank matches; it is not entirely degree-free.
It is an experimental dissimilarity, not a claim of the original TMD metric
properties or stability bounds. Exact matching is retained in all four arms.

## Computation and outputs

`src.tree_distance_factorial.run_tree_distance_factorial` validates the raw cache,
builds only the normalized-feature depth-0/1 cache, and evaluates all eight
methods under the same train-only candidate and validation-query protocol.
It inherits weight, self-loop convention, and storage dtype from the raw cache.
The existing raw cache, including deeper matrices and its summary, is unchanged.
The normalized cache can resume interrupted computation. No test labels are used.

Full normalized sum matrices are stored in their own directory. Derived mean
validation-to-training blocks, together with all controls and node IDs, are saved
in `validation_blocks.npz`. Full mean matrices need not be stored: they can be
reconstructed from their respective full depth-0/1 matrices and the degrees.

The standard sweep, best settings, class recalls, and degree diagnostics are
saved as in [the first diagnostic](tree_distance_analysis.md). `effects.csv`
additionally reports percentage-point contrasts at each identical k/voting:

- Normalization effect under sum: `l2_sum - raw_sum`.
- Normalization effect under mean: `l2_mean - raw_mean`.
- Mean-aggregation effect on original features: `raw_mean - raw_sum`.
- Mean-aggregation effect on normalized features: `l2_mean - l2_sum`.
- Interaction: difference between the two mean-aggregation effects.

These are controlled algorithmic contrasts on this validation split, not
statistical significance claims. Compare the complete k curves, class recalls,
and exact-degree-matched AUC as well as the maxima. If normalized raw Euclidean
distance alone explains an improvement, it should not be attributed solely to
tree structure. This experiment does not evaluate condensed GCN training.

Colab tests cover normalization/zero vectors, the unchanged root term, factorial
contrasts, exhaustive padded mean matching, reuse of the original cache, and
unused-label independence. Local verification was limited to static checks.
