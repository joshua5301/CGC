# FSW-GRIP with mass-weighted CE

This experimental method uses a frozen, label-free Fourier Sliced-Wasserstein
encoder for GRIP's feature distance. It does not replace the student GCN.
The teacher remains the existing kernel teacher on propagated features, fitted
only on training labels. Validation selects configurations; test labels are used
only for evaluating the selected configuration.

For each direction and frequency, sort the projected neighbor features and
compute `2 (1 + frequency) integral Q(t) cos(2 pi frequency t) dt`. The integral
is evaluated exactly for the empirical quantile step function. A zero-frequency
coordinate and an explicit neighbor-count coordinate are included. Empty
neighborhoods have zero aggregate and count. Self-loops are omitted from this
encoder; the root feature is retained explicitly.

Each layer concatenates the previous representation, FSW coefficients, and
neighbor count. There is no learned classifier or dimension-reducing MLP in the
encoder. Root and coefficient blocks use positive scalar RMS normalization
fitted on the original graph, and degree uses the original maximum degree.
These scales and the seeded directions/frequencies remain fixed when encoding
the condensed graph. With input dimension D, depth L, and width M, output
dimension is D + L (M + 1).

Source: [FSW-GNN, equations 8–11 and theorem 3.5](https://arxiv.org/html/2410.09118v2).
This is an FSW-inspired finite-width encoder, not a claim that our chosen width,
depth, or concatenation architecture satisfies that paper's sufficient theorem
conditions. No exact TMD, certified distortion constant, or guarantee for every
GNN is claimed. Random directions are fixed across all configurations except
when their explicit encoder settings change; student seeds do not change them.

## Partition and graph realization

GRIP minimizes its existing normalized Euclidean-distance plus KL objective in
the FSW coordinates. Its geometric-median representatives need not correspond
to realizable trees. We therefore construct a real graph:

1. Aggregate intercluster edge counts, keep the strongest `neighbors` connections
   per node, and symmetrize their union. The resulting graph is unweighted;
   symmetrization can make degrees exceed `neighbors`. Ties use cluster order.
2. Initialize raw synthetic features with cluster means of raw original features.
3. With topology, assignments, and encoder fixed, minimize
   `sum_k (n_k/N) ||encoder(G')_k - center_k||` over synthetic features using Adam.
4. Retain the lowest reconstruction-loss iterate, including initialization.

This restricted graph realization is a heuristic optimizer with a measurable
residual, not an exact tree barycenter. There is no guarantee of an exact inverse
embedding. The graph topology is not optimized by gradients. The frozen
encoder does not use label-dependent gradients or any student weights.

Let z_i be source embeddings, c_k GRIP centers, and z'_k realized embeddings.
The saved diagnostics are:

- `embedding_fit = mean_i ||z_i - c_a(i)||`;
- `realization_initial/final = sum_k (n_k/N) ||c_k - z'_k||`;
- `embedding_direct = mean_i ||z_i - z'_a(i)||`;
- `embedding_upper = embedding_fit + realization_final`.

The triangle inequality implies `embedding_direct <= embedding_upper` up to
floating-point error. This is an embedding-space inequality, not a measured TMD
bound. A tree-risk guarantee additionally requires a lower-Lipschitz constant
valid for both original and synthetic trees and an appropriate student stability
assumption. `J_initial/final` refer only to the normalized GRIP partition cost.

## Student and experiment protocol

The output contains raw-dimensional `x`, average teacher labels `y`, cluster
`counts`, and a standard symmetric GCN propagation matrix with unit self-loops.
The existing two-layer GCN trains on this actual graph with
`sum_k (n_k/N) CE(q'_k, prediction_k)`. Evaluation uses the original graph.
The mass weighting matches the clustered empirical measure; it does not by
itself establish the missing embedding-distortion guarantee. Empty clusters
follow existing GRIP behavior: removed, with actual/requested counts reported.

`run_experiments(method='fsw_grip', loss_weighting='mass', search='grid', ...)`
accepts grid keys `fsw_depth`, `fsw_width`, and `fsw_frequency` in addition to
GRIP's teacher, KL, dropout, and optimizer parameters. The `fsw` argument sets
fixed encoder seed and graph-realization `neighbors`, `steps`, and `lr`.
Encoder embeddings are cached for the current setting. Condensations are
persisted independently of student hyperparameters and include encoder state,
assignment, centers, graph edges, and reconstruction diagnostics. Keep the
output directory and protocol unchanged to resume completed trials.

No node-pair distance matrix is computed. Persistent embeddings use O(Nd)
space; projection sorting is batched by degree and direction. Edges and neighbor
indices use O(E) storage. Large degrees and autograd workspaces still cost memory.
The dense K-node quotient/GCN matrices use O(K^2), and GRIP uses blocked N-by-K
distances. This implementation targets small citation graphs first.

Run `python -m pytest -q tests/test_fsw.py tests/test_grid_search.py` in Colab.
Tests cover exact quantile coefficients, permutation equivariance, cardinality,
feature gradients, identity reconstruction, the triangle inequality, and a
manual graph-aware mass-CE student update. No local numerical tests were run.
