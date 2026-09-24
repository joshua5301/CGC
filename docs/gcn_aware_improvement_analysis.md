# GCN-aware improvement: theoretical analysis

Status: derivation and proposed algorithm, not implemented or experimentally
validated. No local training or numerical smoke tests were run.

## 1. What the results do and do not establish

At the selected arxiv configuration, combined partitioning outperformed variance
alone by about 0.38 validation percentage points in one fixed-partition-seed
experiment. Equal-mass transport and a conservative uniform-loss correction both
reduced accuracy. Independently tuned mass-weighted CE on Cora did not consistently
improve validation over the earlier uniform runs; their search spaces differed.
These do not isolate a single causal bottleneck. In particular, none establishes
that a more conservative global upper bound will improve nonlinear GCN accuracy.

## 2. A graph/nonlinearity error omitted by H2 clustering

Use the actual normalized propagation A from the evaluated GCN. Ignoring biases
and dropout temporarily, original logits are

    A ReLU(A X W0) W1.

An edgeless network evaluated on H2=A²X instead produces

    ReLU(A²X W0) W1.

For nonnegative A and U=AXW0, the hidden discrepancy is exactly

    K(W0) = A ReLU(U) - ReLU(AU)
          = (A|U| - |AU|)/2 >= 0 entrywise.

Row stochasticity is not needed: ReLU is positively homogeneous. Mixed signs
across neighbors produce a potentially nonzero discrepancy. Clustering H2 with
arithmetic representatives does not explicitly model this term. This identity
alone does not quantify its impact on the observed data.

For the actual biased model the final-layer input is

    Zo = A ReLU(A X W0 + 1 b0^T),

whereas for self-loop-only representatives C it is

    Zs = ReLU(C W0 + 1 b0^T).

These are the quantities an architecture-aware objective should compare. Simply
moving a bias through A is invalid when A1 != 1. For the derivations below append
a constant coordinate to both Zo and Zs to represent the final bias. Dropout is
off; its training-time randomness is not covered by these statements.

## 3. Exact excess-risk decomposition

Let Ro(theta) be original empirical teacher-label CE and Rs(theta) be uniform
representative CE, using their actual respective graph forwards. Set D=Ro-Rs.
If theta_s minimizes Rs within epsilon on a common parameter set Theta, and
theta_o minimizes Ro on that set, then

    Ro(theta_s)-Ro(theta_o) <= osc_Theta(D) + epsilon,
    osc_Theta(D) = sup_Theta D - inf_Theta D.

An additive constant in D has no effect. This makes variation of the loss
discrepancy a more direct target than its absolute value.

On a convex radius-rho ball around theta0, where D is C² and its Hessian is
Lipschitz with constant M, Taylor expansion gives

    osc(D) <= 2rho ||grad D(theta0)||
              + rho² ||Hessian D(theta0)||op + M rho³/3.

This is a conditional local bound: both relevant minimizers must lie in the ball.
ReLU gate crossings invalidate the smooth Taylor assumptions in general. Finite
probes or sampled Hessian directions do not certify these suprema. Therefore this
formula motivates gradient/curvature preservation but is not a global GCN proof.

## 4. A stronger, implementable statement for a frozen first layer

Fix W0,b0, and thus Zo,Zs. Let Q be the fixed original teacher targets and Y the
synthetic soft labels. For a final affine head W (bias included), define

    Fo(W) = mean_i CE(Q_i, softmax(Zo_i W)) + lambda/2 ||W||F²
    Fs(W;Y) = mean_j CE(Y_j, softmax(Zs_j W)) + lambda/2 ||W||F²,

with lambda>0. Both are lambda-strongly convex. The original objective is globally
L-smooth, with the valid bound

    L <= lambda + (1/2) lambda_max(Zo^T Zo / N).

Let Wo and Ws be their exact minimizers. At Wo define

    Po = softmax(Zo Wo), Ps = softmax(Zs Wo)
    g(Y) = Zs^T(Ps-Y)/m - Zo^T(Po-Q)/N.

The identical regularizer cancels in the gradient difference. Strong convexity
of Fs and smoothness of Fo imply

    ||Ws-Wo||F <= ||g(Y)||F / lambda,
    Fo(Ws)-Fo(Wo) <= L ||g(Y)||F² / (2lambda²).

Proof: grad Fo(Wo)=0, so grad Fs(Wo)=g(Y). Strong monotonicity of grad Fs gives
lambda||Ws-Wo|| <= ||grad Fs(Wo)||. Apply the smoothness inequality for Fo at its
minimizer. If g(Y)=0, Wo is also the unique minimizer of Fs. A small lambda can
make the nonzero upper bound loose. Approximate head fits need stationarity-error
terms; the exact-minimizer statement must not be silently applied to finite fits.

This is an empirical regularized-head result, not a full-GCN training, population,
accuracy or test guarantee. The final evaluation protocol remains unchanged.

## 5. Label calibration is a convex problem

For fixed representative features and probe weights, Ps and Zs are fixed. Thus

    minimize_Y || Zs^T(Ps-Y)/m - Go ||F²
    subject to Y_j >= 0, sum_k Y_jk = 1,
    Go = Zo^T(Po-Q)/N

is a convex quadratic over a product of simplices. Cluster-mean labels are a
feasible starting point. A converged solve cannot increase this matching objective
relative to those labels. No equal-cell-mass constraint is imposed; the actual
uniform synthetic CE is represented directly. Exact matching may be infeasible.

For multiple frozen GCN probes/checkpoints, sum their squared residuals using a
shared Y. This remains convex, but improves an aggregate proxy, not every probe
or every possible student. Full-parameter CE gradients are also affine in Y for
fixed features and fixed parameters, enabling a larger convex label subproblem.

The frozen-head Hessian contains terms

    (diag(p_i)-p_i p_i^T) tensor (z_i z_i^T),

and is independent of labels. Label calibration alone therefore cannot repair
hidden geometry or curvature mismatch. Feature or partition updates are needed
for that. Hessian-vector products avoid materializing the full matrix.

## 6. Proposed algorithm, not another balance penalty

1. Keep ordinary nonempty hard cells, the current budget and uniform student CE.
2. Construct a small set of train-only GCN first-layer probes. Fix their seeds and
   training checkpoints without consulting test labels. Use the exact evaluation
   graph normalization. Cache their original-graph hidden features and targets.
3. Initialize representatives with the existing risk partition, or its D² seeds
   when evaluating a from-scratch variant; neither requires GRIP initialization.
4. Solve the convex simplex label subproblem at fixed representative features.
5. Update partition or representative features against the actual probe gradient
   discrepancies, recalibrating labels. Only the surrogate improvement is assured.
   Partition updates preserve arithmetic representatives; free feature updates
   instead change the method into synthetic-feature condensation.
6. Freeze the condensed data and evaluate fresh two-layer GCNs with uniform CE.

The minimal next implementation would calibrate labels and then optimize
representatives using a few frozen probes, without adding curvature matching in
the first version. It directly models nonlinear propagation and the student's
uniform loss. Multiple probes address some single-probe dependence but cannot
eliminate it. Probe generation and feature optimization introduce real compute
costs; no speed or accuracy improvement has been established.

## 7. Related work and novelty limits

Gradient matching and curvature matching are existing condensation principles.
The derivations above do not establish a novel algorithm merely by adopting them.
Potential research contribution would require a demonstrably useful graph-aware,
partition-constrained realization and efficient optimization, with a separate
prior-art review.

- Zhao et al., Dataset Condensation with Gradient Matching:
  https://arxiv.org/abs/2006.05929
- Jin et al., Graph Condensation for Graph Neural Networks:
  https://arxiv.org/abs/2110.07580
- Shin et al., Loss-Curvature Matching for Dataset Selection and Condensation:
  https://proceedings.mlr.press/v206/shin23a.html
- Loukas and Vandergheynst, Spectrally Approximating Large Graphs with Smaller Graphs:
  https://proceedings.mlr.press/v80/loukas18a.html

A learned Mahalanobis metric or replacing trace by spectral norm alone would not
fix the propagation/nonlinearity discrepancy. A tighter scalar bound can have a
different optimizer without improving task performance. Such changes should not
be presented as automatically superior.
