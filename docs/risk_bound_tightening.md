# Tightening the original risk bound

Scope: the original fixed-feature, mass-weighted representative CE theorem.
Both comparison students lie in the same Frobenius ball ||W||F<=B. Teacher labels
are fixed probability vectors. None of the results below automatically extends
to unconstrained two-layer GCN or uniform representative CE. No numerical or model
experiments were run for this review.

## 1. Exact starting point

For arithmetic cluster means, define S as the within-cell feature covariance,
E as the original-minus-mass-weighted representative feature/label moment, and
D(W)=R(W)-Rp(W). Then

    D(W)=G(W)-<W,E>,
    G(W)=mean_i KL(p_W(c_a(i)) || p_W(h_i)).

The orientation of KL is centroid prediction to individual prediction. This is
the logsumexp Bregman identity; the linear terms vanish on averaging within cells.
It is an identity for Jensen gap, not a proposed extra teacher-label KL penalty.

For epsilon-optimal condensed training in the same ball,

    R(W_hat)-R(W_star) <= osc(D)+epsilon,
    osc(D)=sup_W D(W)-inf_W D(W).

This oscillation is the relevant discrepancy, rather than twice an absolute
loss-error bound. The original derivation already exploits one-sided G>=0.

## 2. Trace relaxation is avoidable

The logsumexp Hessian gives

    0 <= G(W) <= (1/4) tr(W^T S W).

Since S is positive semidefinite,

    J_spectral = B² lambda_max(S)/4 + 2B||E||F
    J_frobenius = B² ||S||F/4 + 2B||E||F
    J_trace = B² tr(S)/4 + 2B||E||F

are all upper bounds on the compression part, and

    J_spectral <= J_frobenius <= J_trace.

The trace term can overstate the spectral term by tr(S)/lambda_max(S). For an
illustrative S=sigma² I_d these feature terms differ by factors d versus 1;
Frobenius gives sqrt(d). This is not a measurement of the experimental datasets.

A Frobenius implementation retains cheap exact rank-two updates. For a node move,
let deltaS=-ka ua ua^T+kb ub ub^T, using the original removal/addition coefficients.
Then

    ||S+deltaS||F² = ||S||F²
      -2ka ua^T S ua + 2kb ub^T S ub
      +ka²||ua||^4 + kb²||ub||^4 -2ka kb(ua^T ub)².

This needs covariance statistics, not a new balance penalty or hyperparameter.
Maximizing the remaining eigenvalue may shift distortion to other directions;
a pointwise smaller upper bound does not imply a better optimizer or accuracy.
Power-iteration Rayleigh quotients are lower bounds on lambda_max, so substituting
one without an upper certificate does not retain an upper-risk guarantee.

## 3. Keep feature/moment direction coupling

The usual spectral bound maximizes the quadratic and linear terms separately.
Define instead

    U(S,E)=max_{||W||F<=B} [(1/4)tr(W^T S W)-<W,E>].

Because sup D<=U and inf D>=-B||E||F,

    J_joint=U(S,E)+B||E||F

is a valid bound and J_joint<=J_spectral. It uses whether E aligns with the
high-variance eigenspaces, rather than discarding those directions. This is not
the unsupported claim max(f+g)=max(f)+max(g).

Diagonalize S=V diag(s_r) V^T and let a_r=||(V^T E)_r||². The norm-ball quadratic
maximization is a trust-region subproblem with the exact scalar dual

    U = inf_{nu>lambda_max(S)/4}
        [nu B² + (1/4) sum_r a_r/(nu-s_r/4)].

Boundary optima use a limiting value/pseudoinverse and the usual hard-case range
condition. In particular E=0 yields U=B² lambda_max(S)/4. For any strictly feasible
nu, the displayed expression itself is an upper certificate; approximate primal
maximization alone supplies a lower bound and is insufficient for certification.
For an interior solution, its derivative condition is

    B² = (1/4) sum_r a_r/(nu-s_r/4)².

Thus the coupling can be retained with an eigendecomposition and a scalar solve,
without an adversarial GCN or adding balance constraints. Per-partition evaluation
is straightforward; evaluating this for every relocation can be more expensive
than the Frobenius alternative, especially in high feature dimension.

## 4. A curvature lower bound can tighten the second side

Softmax is invariant to common class shifts. Restrict without loss to W1=0, since
class-centering decreases the Frobenius norm and leaves predictions unchanged.
Also E1=0. Suppose ||h_i||<=R. Along all node/centroid segments and all admissible
W, logit spread is at most sqrt(2)BR. Therefore

    kappa = 1 / [1+(K-1)exp(sqrt(2)BR)]

is a valid probability lower bound. On the class-contrast subspace the softmax
Hessian is at least kappa I. Consequently

    (kappa/2)tr(W^T S W) <= G(W) <= (1/4)tr(W^T S W).

Let Lk=min_{||W||F<=B}[(kappa/2)tr(W^T S W)-<W,E>]. Then

    J_curvature=U-Lk <= U+B||E||F = J_joint.

Equivalently, -Lk is the minimum over tau>=0 of

    tau B² + (1/4)tr(E^T[(kappa/2)S+tau I]^{-1} E),

with the appropriate limiting/pseudoinverse convention. This is another scalar
dual. For large BR, kappa can be essentially zero, so this improvement may be
negligible. A measured Hessian at one centroid cannot replace a bound over every
relevant segment and admissible W without changing the theorem's assumptions.

## 5. Which constants are actually loose?

The global logsumexp Hessian constant 1/2 is sharp: two equally probable classes
have curvature 1/2 in their contrast direction. The Taylor factor 1/2 then gives
the coefficient 1/4. Uniform class probabilities give smaller curvature for K>2,
but the original theorem does not restrict all predictions to that distribution.

The coefficient 2B on ||E|| cannot simply be halved either. For small B,
G(W)=O(B²) while -<W,E> varies by 2B||E|| across the ball whenever E!=0.
Using curvature or more restrictive model information is different from changing
that coefficient without justification.

## 6. Recommended mathematical next step

Prioritize covariance geometry and quadratic/linear coupling. The unconditional
chain is

    J_joint <= J_spectral <= J_frobenius <= J_trace.

The curvature-lower-envelope version can tighten it further but may be numerically
weak for a large global ball. All inequalities compare a fixed partition, B and
feature space. They do not order the test accuracies of their minimizing partitions.
The original method's initialization surrogate is still valid as an upper bound
on J_trace, hence on these smaller objectives, but is not necessarily a good
initializer for their distinct minimizers. Actual GCN/uniform-loss mismatch remains.

Trust-region background (the quadratic subproblem machinery is established work):
Fortin and Wolkowicz, The Trust Region Subproblem and Semidefinite Programming,
https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/2004fortinwolkTRS.pdf
