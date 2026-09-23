# Directional correction of teacher probabilities before GRIP

This is the first (held-out residual correction) proposal, not the ensemble
uncertainty-set proposal. The downstream model remains the existing two-layer
GCN, hidden 256, identity condensed adjacency, uniform soft-label CE.

## Construction

Randomly split only the original training indices, using a fixed seed and no
label-based stratification: half for fitting a kernel teacher and half for
calibration. The held-out teacher never sees calibration labels when fitting.
Features H=A_hat^2 X and kernel basis can use unlabeled graph features as in the
existing transductive GRIP protocol. No validation/test label is used to split,
fit, select neighbors or construct corrected probabilities. Validation metrics
are evaluated only after construction; test is used only in student evaluation.

For calibration node u form r_u=onehot(y_u)-F_u. At each target v, average r_u
over k nearby calibration nodes, add that average to F_v, and project onto the
probability simplex with the Euclidean projection. This changes the direction
of the prediction, not just its entropy. Calibration nodes may use their own
known training label. Original GRIP then condenses the corrected probabilities
with its original normalized feature-distance + KL objective. No new clustering
penalty or weighted student loss is added.

## Conditional bound and neighbor choice

Write e_v=p_v-F_v. Assume e is L_e-Lipschitz in H. Conditional on features and
the fitted teacher, assume calibration labels are independent draws from their
conditional class probabilities. Selection must be independent of those labels.
Graph labels are not automatically conditionally independent; dataset sampling
or class-balanced splits also require care. These are assumptions, not properties
proved for Cora by this code.

For uniform weights on k feature-selected neighbors:

    ||p_v - corrected_F_v||_2
       <= L_e * mean_neighbor_distance
          + sqrt(C * log(2*N*C*M/delta) / (2*k)).

Proof: the mean residual differs from e_v by at most the Lipschitz bias plus
the mean centered label noise. For each coordinate, Hoeffding bounds weighted
Bernoulli noise; union-bound over C coordinates, N targets and M candidate k.
The coordinate bound gives the displayed L2 noise term. Euclidean simplex
projection cannot increase distance to p_v, which lies in the simplex. This
is simultaneous over the candidate k, so minimizing this bound over k is valid
under the assumptions. It is a conditional expected-label estimate, not a
guarantee of each observed hard label or of GCN test accuracy.

Implementation parameter `slope` means L_e = slope / s, where s is the median
positive nearest-other calibration-feature distance, or 1 when all coincide.
Candidates are {1,2,4,8,16,32,64}, clipped/deduplicated to the calibration size,
plus all calibration nodes. k minimizes the bound, using features only. Delta
defaults to .05. Slope zero selects all calibration nodes (global residual
correction control); it is not evidence that the residual is truly constant.

The slope is an assumed sensitivity, NOT a proven or calibrated estimate. The
output explicitly marks coverage uncertified, saves each selected k and bound,
and reports the fraction of bounds below the trivial simplex diameter sqrt(2).
A large bound can be vacuous. Do not relabel the candidate grid as guaranteed
true-label coverage or use the smallest bound from invalid slope assumptions.
In the local Cora integration check (default 70 calibration labels), slopes
.5, 2 and 8 produced no bounds below sqrt(2). This diagnostic depends on the
teacher/features/split, not the number of student epochs. The implementation is
therefore a conditional-bound prototype; its current default spatial corrections
do not deliver a nontrivial numerical certificate on that check. Slope zero's
smaller bound additionally requires a spatially constant residual, which is not
established. Tightening/validating this theory remains work, even if accuracy
improves in Colab.

If student predictions obey q_c >= tau > 0, the conditional estimate yields
|CE(p_v,q_v)-CE(corrected_F_v,q_v)| <= ||p_v-corrected_F_v||_2 * ||log q_v||_2
<= bound_v * sqrt(C)*log(1/tau). This can be combined with a valid GRIP
teacher-target bound. The actual unconstrained GCN has no common enforced tau;
the runner does not claim a full numerical risk certificate. Original SGC
representation analysis also remains SGC-specific. Improving teacher estimation
is a contribution candidate, not by itself a new general-MPNN condensation theorem.

## Colab diagnostic

Run `residual_grip.py`: fixed Cora .052, gamma .01, T 2, GRIP coefficient .5,
dropout .9. Six conditions: full training-label teacher, split teacher without
correction, and residual correction with slopes {0,.5,2,8}. The first is the
original GRIP control; the second isolates the correction effect from the
reduced teacher training set. All use the same kernel feature construction and
condensation seed. Ten student seeds 3-12 give 60 student fits of 1000 epochs.
Each method has its own fixed condensation; initial student seed is paired.
The random calibration split is fixed, so uncertainty across splits is not
measured by these student-seed standard deviations.

Output: `MyDrive/GRIP_cora_residual_teacher`. Teacher probabilities, fit/calibration
indices, corrected targets, k, bounds, condensation artifacts and per-seed fits
are saved. `summary.json` reports paired differences versus both baselines;
the compact table shows differences versus the split teacher. Intervals are
pointwise Student-t intervals over student seeds, not multiplicity-corrected.
Do not select a slope using test scores. `--repeat` can be increased to extend
cached seeds. Epoch logs remain in a timestamped file; output is compact.
