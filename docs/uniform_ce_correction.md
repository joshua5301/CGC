# Uniform-CE correction for hard partitions

Let x be the centered, RMS-normalized input and z=[x;1]. The constant coordinate
accounts for affine logits. Assume q is a probability vector. For m nonempty
cells, define pi_j=n_j/N, u=1/m, c_j=mean(z_i), y_j=mean(q_i), and delta_j=pi_j-u.
The evaluated representative loss in this derivation is uniform over cells.

Define

    V = (1/N) sum_i ||z_i-c_a(i)||²
    C = sum_j |delta_j| ||c_j||²
    Eu = (1/N) sum_i z_i (q_i-1/K)^T
         - (1/m) sum_j c_j (y_j-1/K)^T
    Ju = B²/4 (V+C) + 2B ||Eu||_F.

For affine softmax logits W^T z with ||W||_F<=B, write

    A_W(z) = logsumexp(W^T z)
    F_W(z) = A_W(z)-log(K)-(1/K) 1^T W^T z.

The logsumexp Hessian has operator norm at most 1/2, hence
0 <= F_W(z) <= B²||z||²/4. The exact original-minus-uniform risk difference is

    R(W)-Ru(W) = G(W) + sum_j delta_j F_W(c_j) - <W,Eu>,

where G(W) is the mass-weighted Jensen gap and 0<=G(W)<=B²V/4.
For any two admissible W1,W2, the difference of the signed F terms is at most
B² sum_j |delta_j| ||c_j||²/4. If W_hat minimizes Ru within epsilon of its
infimum on that same norm ball and W_star minimizes R, then

    R(W_hat)-R(W_star) <= Ju + epsilon.

This is a conservative, model-uniform envelope of the missing loss discrepancy,
not an exact GCN loss correction. It removes the mass-weighted-training assumption
but retains a fixed normalized feature space, teacher targets, and a constrained
affine softmax class. The actual two-layer GCN is unconstrained and has no test-risk
guarantee from this result. B is fixed during optimization. The added constant
coordinate contributes one to ||c_j||²; its within-cell variance is zero, and the
last row of Eu captures label-marginal mismatch. Returned features exclude it.

## Optimization

The implementation keeps the original D² initialization in the unaugmented feature
and label space. It uses hard nonempty cells, arithmetic feature means and mean
teacher labels. No transport, equal-size constraint or new penalty coefficient is
introduced. Candidate moves include exact changes of V, C and Eu. The Eu change is
the sum of the removed and added cell outer-product changes, each weighted by 1/m;
it cannot use the old mass-weighted rank-one covariance update.

GPU candidate deltas use outer-product inner products to avoid allocating an
N-by-m-by-d-by-K tensor. Batch proposals are accepted only after full recomputation
of Ju; rejected batches are bisected using the existing solver. The first eligible
block checks four move deltas against independently recomputed full objectives in
Colab by default (verify_deltas=True). No local numerical or model execution was
performed. Syntax and diff checks are the only local validation.

The convergence flag uses the existing no-accepted-moves stopping rule and solver
tolerances; a sweep cap is not convergence. Full GCN evaluation always uses uniform
soft CE. Ju values should not be compared directly with the original J: the
statistics and the affine model class differ.
