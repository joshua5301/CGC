# Robust teacher-label compression in GRIP

This is a conditional, theory-derived objective implemented as an opt-in
`--methods robust` in `sweep_distance.py`. It does not claim a certified true
risk bound for the current GCN student. In particular, the distance-based
uncertainty radius is an explicit heuristic, not a calibrated coverage bound.

## Objective and exact inner problem

For teacher probability F_v and supplied L1 radius epsilon_v in [0,2], set
U_v = {p >= 0, sum(p)=1, ||p-F_v||_1 <= epsilon_v}. The exact label cost is
phi_v(y) = max_{p in U_v} -p.log(y). The optimized objective is

    mean_v [||H_v-C[a_v]|| / s_H
            + coefficient * (phi_v(Y[a_v]) - H(F_v)) / s_KL].

H = A_hat^2 X; s_H and s_KL use the original GRIP global median/label scales
(with a numerical floor). Thus phi-H(F) = KL(F||Y) + Delta(F,Y,epsilon).
Delta is nonnegative and depends on the assigned label, unlike adding a
fixed node-wise uncertainty penalty, which would not affect the partition.
The tunable coefficient/scales make this a surrogate, not the numerical
certificate with exact theoretical coefficients.

The exact adversary moves up to epsilon/2 probability mass from low-cost
classes to a highest-cost class. It respects simplex capacity, handles ties,
and can stop before exhausting the radius once that class has mass one.
It costs O(C log C) for a node/representative pair; no graph OT solver is used.
Independent linear-program tests check optimality, not just feasibility.

## Conditional risk connection and remaining assumptions

Assume all true conditional label distributions p_v are in U_v and a uniform
representation comparison CE(p,g_v) <= CE(p,q_j)+L*d(v,j) holds. Then

    R_true <= mean_v [L*d(v,a_v) + phi_v(Y[a_v])]
              + sum_j (n_j/N) max_c log(Y[j,c]/q[j,c]).

The last term is an explicit downstream fit residual, NOT the original
KL(Y_j||q_j) residual. Uniform CE training does not automatically bound it
tightly. The implementation leaves downstream training unchanged and does
not numerically certify this full inequality for the two-layer GCN.
An SGC feature bound is not a general-MPNN bound. This is expected CE over
conditional labels, not a finite-test accuracy guarantee.

## Radius policy: experimental until independently justified

Only H and the TRAIN mask enter `distance_radii`; no validation/test labels
are used. r_v is Euclidean distance to the nearest training feature. s is the
median positive leave-one-out training nearest-neighbor distance (fallback 1
if all training anchors coincide). The prototype uses

    epsilon_v = floor + (cap-floor) * r_v/(r_v+s).

The floor defaults to zero, which assumes trust at the training anchors; it
does NOT establish teacher correctness there. Set a nonzero floor if desired,
with all caps >= floor. `--robust-radius-mode constant` provides the essential
control for generic robust smoothing versus the specific distance hypothesis.
Distance-accuracy correlation alone does not prove a posterior L1 error bound.
Neither empirical calibration nor simultaneous coverage is implemented here.
The core `robust_partition` accepts an arbitrary radius tensor so a future
justified radius construction can replace the heuristic without changing it.
Artifacts explicitly record `radius_certificate=False, risk_certificate=False`.

## Optimization

Each setting starts from the same seeded, original GRIP solution. For all-zero
radii the original partition function is returned without any refinement:
features, labels and assignments match the baseline exactly.

Positive-radius updates alternate:

1. Labels: finite projected subgradient steps on the convex robust CE in
   logits, within [-32,32], including initial and uniform candidates. Retain
   the best iterate for each cell. This is approximate optimization, NOT a
   closed-form mean or an exact minimizer. A box-optimum subgradient gap bound
   is logged; it can be loose, especially at ties. The final reported gap is
   for the final fixed partition's label problem.
2. Assignment: exact worst-label cost for all representative candidates,
   batched at 512 nodes by `--batch-size` representatives. Preserve a member
   of each current cell; accept only cheaper assignments.
3. Centers: geometric median proposals with per-cell nonincrease checks.

The initial GRIP may remove empty cells as before; refinement preserves the
resulting number. Every outer phase records and checks the exact objective.
Teacher probability flooring/normalization is numerical, and final student
features/labels are float32 as in the existing runner.

## Colab diagnostic

Set `GRIP_SWEEP_PRESET=robust` and run `colab_distance.py`. It uses Cora 0.052,
gamma .01, T 2, GRIP coefficient .5, dropout .9, caps {0,.01,.03,.1}, seeds
{0,1,2}, 1000 epochs and 20 outer rounds with 150 label steps per round.
That is 4 condensations and 12 student fits; student remains the same two-layer
GCN with uniform soft-label CE. Cap zero is the paired-initialization baseline.
This small diagnostic is not a full hyperparameter sweep.

Output: `MyDrive/GRIP_cora_robust_labels`, with full logs, resumable artifacts,
objective histories, radius quantiles, solver-gap diagnostics and summary.csv.
The Colab display reports results separately for each cap, using validation
only for any remaining selection. No performance improvement is assumed.
