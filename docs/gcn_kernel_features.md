# Exact two-layer GCN kernel features

`src.gcn_kernel_features.run_gcn_kernel_study` adds two analytic kernel distances
to the existing frozen-student local CE comparison. The NTK is the main candidate;
the NNGP isolates the contribution of random forward features. Both are label-free.

## Architecture and derivation

Let S be PyG's GCN normalization of the input graph with self-loops, d the input
width, and h the hidden width. Fix this scalar-output architecture:

    U = SX / sqrt(d)
    f = S phi(UW) a / sqrt(h),   phi(t) = sqrt(2) max(t, 0).

Entries of W and a are independent standard Gaussian variables at initialization.
Both parameter groups are trainable, with equal Euclidean parameter metric.
There are no biases, dropout, output activation or jumping connections. This is
two graph propagation layers with ONE intervening ReLU, not two ReLU blocks.
It is not the infinite-width limit of the previous PyG default Glorot/bias
empirical NTK as parameterized there. The finite students stay unchanged.

Set C = UU^T, r_ij = sqrt(C_ii C_jj), rho_ij = C_ij/r_ij,
theta_ij = arccos(rho_ij). Gaussian integration gives

    Q_ij = r_ij [sin(theta_ij) + (pi-theta_ij) cos(theta_ij)] / pi
    D_ij = (pi-theta_ij) / pi
    K_NNGP = S Q S^T
    K_NTK  = S [Q + C elementwise_mult D] S^T.

Q is the contribution from derivatives with respect to a; C*D is the
contribution from W. If either preactivation variance is zero, Q and D are set
to zero (ReLU derivative convention at zero). This does not alter the C*D term.

For query nodes I, the final sandwich uses S[I,:] on BOTH sides. All original
nodes participate in propagation and the inner covariance; the probe-induced
subgraph is never substituted. The implementation builds the full N-by-N C in
float64 but only the query-by-query final kernels. It is intended for Cora-scale
graphs; it is not a scalable all-pairs construction for Reddit or arxiv.

## Explicit features

For each probe kernel K, compute K=V Lambda V^T and Phi=V sqrt(Lambda).
All nonnegative eigenvalues are kept; only roundoff-negative values are clipped,
and materially negative eigenvalues raise an error. Thus Phi Phi^T reconstructs
the probe kernel, with reconstruction error reported. Euclidean distances in Phi
equal sqrt(K_ii+K_jj-2K_ij), up to numerical roundoff.

There is no per-node feature normalization, rank selection, random projection,
initialization seed, teacher fitting, label temperature or validation tuning.
The coordinates are an exact factorization on the evaluation probe set, not a
learned out-of-sample embedding and not synthetic raw GCN input features.
`gcn_two_layer_kernels(..., ids=None)` also supports a full-node kernel when needed.

## Evaluation

The runner can import the eight extra distances from a preceding `local_ce`
folder and verifies their recorded hashes and frozen-student hashes. This makes
15 methods including the five original comparators. Without that folder, it
compares the two new distances with the five original methods. Original saved
GCN, SAGE and GIN students are all evaluated; no student is retrained. Calibration
split, thresholds and k values are inherited from the previous local evaluation.

Kernel matrices, explicit features, distances and probe IDs are saved to
`gcn_kernel_features/<hash>/features.npz`. `diagnostics.csv` reports eigenvalues
and reconstruction error. Local CE tables and plots use the existing evaluation
pipeline. Distances are scaled by calibration-pair medians, with CE scale fitted
only on calibration nodes. Quantile selection and kNN selection remain separate.
Small CE change is not itself evidence of improved condensed-data accuracy.

## Verification in Colab

`tests/test_gcn_kernel_features.py` checks known ReLU Gaussian moments, analytic
NTK against explicitly differentiated finite GCN angular quadrature, node
permutation equivariance, full/probe kernel agreement, feature reconstruction,
distance identities and CUDA/CPU agreement. Angular quadrature in two input
dimensions uses radius sqrt(2) and readout coefficients with a^2=1: the required
radial and readout second moments match Gaussian moments for these kernels.
No numerical tests or training are run locally.

References: [GNTK](https://arxiv.org/abs/1905.13192),
[node GNTK](https://arxiv.org/abs/2110.03763). The formulas above are specialized
directly to the explicitly stated two-layer GCN rather than treating a generic
GNTK block count as PyG's number of convolution layers.
