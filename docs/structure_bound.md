# Structure-aware partition from the message-passing risk bound (`--edges structure_identity | structure_median`)

Opt-in path on top of GRIP; the default path is untouched. No OT / LP / Sinkhorn. Reference implementation in
`src/partition_struct.py` (float64 numpy, sequential exact moves).

## Setting and bound

Faithful student: bias-free `Z[l+1] = sigma_l(P Z[l] W_l)` on the original graph and `Zc[l+1] = sigma_l(Q Zc[l] W_l)`
on the condensed one, same weights, last layer linear. `P` is the row-stochastic operator the student uses
(`build_transition`: binarised edges, diagonal removed, rows normalised, isolated node -> `P[t,t] = 1`); `Q` is the
condensed operator. `S` one-hot assignment, `B = P S`.

With `E_l = ||Z[l] - S Zc[l]||_{2,1}` (sum of row norms), `R_S = ||P S - S Q||_{2,1}` and `c_P = max_u sum_t P[t,u]`
(row-stochasticity alone does **not** give `||P M||_{2,1} <= ||M||_{2,1}`; `P = [[1,0],[1,0]]`, `M = [1,0]^T` doubles it):

    P Z - S Q Zc = P (Z - S Zc) + (P S - S Q) Zc
    E_{l+1} <= a_l ( c_P E_l + R_S ||Zc[l]||_2 ),    a_l = Lip(sigma_l) ||W_l||_2
    E_K <= A E_0 + Bcoef R_S,   A = prod_l abar_l c_P,   Bcoef = sum_l abar_l M_l prod_{s>l} abar_s c_P

Node-wise CE is sqrt(2)-Lipschitz in the logits, hence for the teacher-target risk

    R_F(g) <= mean_t H(F_t) + [alpha D_H + beta D_S + D_KL] / N + sum_j n_j KL(Yc_j || g_c[j]) / N
    D_H = sum_t ||H_t - Hc[a_t]||,  D_S = sum_t ||B_t - Q[a_t]||,  D_KL = sum_t KL(F_t || Yc[a_t]),
    alpha = sqrt(2) A,  beta = sqrt(2) Bcoef.

Explicit constants (`bound_coefficients`): with `R = max_t ||H_t||`, `Hc` in that ball, `Q` row-stochastic,
`M_l = sqrt(m) R prod_{s<l} abar_s`. The bound is a surrogate for the teacher risk, conditional on common weight /
representation bounds across trainings; a decrease of `J` is not a claim about test risk, and `beta = 0` is an
ablation, not a bound-preserving variant.

## Objective and blocks

`J = (alpha D_H + beta D_S + mu D_KL) / N`, no per-term normalisation (`--struct_coef surrogate`: alpha / beta / mu from
`--root_weight / --neighbor_weight / --label_weight`; `--struct_coef bound`: alpha, beta from the explicit constants with
`K = 2`, `--struct_abar`, mu = 1). Teacher `F` = the existing kernel teacher on `A^2 X` (training labels only), smoothed
once by 1e-12 and fixed; structural features `H` = raw `X` (the faithful student's input); initial partition = GRIP's.

| block | update | exactness |
|---|---|---|
| assignment | node order fixed by seed; for every node all `m` destinations are scored with the exact `Delta J` of the frozen centres: `Delta D_H`, `Delta D_KL` from precomputed rows, `Delta D_S` over `U = {u : P[u,t] != 0} + {t}` (for `u != t` only coordinates `a`, `b` of the residual change -- squared-norm increments then sqrt; row `t` recomputed densely against every `Q[b]`, including a self transition); best strictly improving move applied immediately; a singleton cell never loses its node | exact coordinate descent for frozen centres |
| Hc | per-cell geometric medians (Weiszfeld + Vardi-Zhang, best iterate kept) | cell cost non-increasing |
| Yc | cell means of `F` | forward-KL minimiser |
| Q | `identity`: fixed `e_j`; `learned_median`: per-cell geometric medians of the current `B` (convex iterates stay in the simplex, checked, never renormalised) | cell cost non-increasing |

After every sweep `B == P S` is re-verified by a fresh sparse product and the incremental `J` is compared with a full
recomputation. The loop stops on a small full-objective improvement, not on "no move".

## Students in `main.py`

- `--struct_student faithful` (bound applies): condensed graph `(Hc, Q, Yc)`, student `PropGNN` (bias-free, mean
  aggregation with the given matrix, no self-loops, no renormalisation), served on `(X, P)` via `attach_transition`.
- `--struct_student transfer` (empirical): only the learned partition is used -- representatives are the geometric
  medians of `A^2 X` over the new cells, `A' = I`, GCN as in the main table. Same evaluation as GRIP; the bound does not
  cover it.

## Checks (`python -m pytest tests/test_partition_struct.py -q`)

1. row-stochastic counterexample: `||P M||_{2,1}` 1 -> 2, `c_P = 2`
2. single-move `Delta J` == full recomputation (directed, undirected, self transitions of isolated nodes, both Q modes)
3. caches (`B`, residuals, norms) consistent after applied moves
4. full runs monotone, cells non-empty, `Q` and `Yc` in the simplex
5. identity mode leaves `Q`; learned mode: medians do not cost more than means
6. `beta = 0` reduces to the feature + KL objective
7. layer recursion and CE bound hold numerically for a random 2-layer bias-free student
8. `PropGNN` with the identity weight reproduces `P @ H`; no hidden self-loops or renormalisation
9. geometric median: coincident points, simplex preserved

## Positioning

Minimising `||P S - S Q||` is the lumpability defect of a Markov chain (Kemeny-Snell) and is close to restricted
spectral / coarsening criteria (Loukas 2019); the coarsening criterion itself is not new. The claim is limited to: the
three terms of `J` come from one risk bound for the stated student class, and `J` is decreased monotonically without
training. The teacher is fitted, so "training-free" refers to the condensation, not the whole pipeline.

## Results

cora 5.2 % (m = 140), `structure_identity`, teacher / GRIP initial partition of the main table, 3 seeds, dropout grid {0.5, 0.9}
(local run; c_P = 46.9 on cora's row-stochastic P, so the explicit bound constants are large: alpha 3113, beta 4392).

| coefficients | student | J (init -> final) | D_S (mean) | moves | test | val |
|---|---|---|---|---|---|---|
| alpha 1, beta 0, mu 1 | transfer (A^2 X, A'=I, GCN) | 3.865 -> 3.834 | 0.53 -> 0.65 | 223, 76, 32, ... | 84.30 +- 0.46 | 81.53 |
| alpha 1, beta 1, mu 1 | transfer | 4.399 -> 4.194 | 0.53 -> 0.30 | 474, 69, 5, 0 | 84.03 +- 0.25 | 81.73 |
| bound (3113, 4392, 1) | transfer | 14210 -> 13113 | 0.53 -> 0.28 | 553, 94, 9, 1, 0 | 84.10 +- 0.20 | 81.67 |
| alpha 1, beta 0, mu 1 | faithful (raw X, PropGNN on P) | 3.865 -> 3.834 | | | 84.03 +- 0.40 | 81.13 |
| alpha 1, beta 1, mu 1 | faithful | 4.399 -> 4.194 | | | 83.53 +- 0.67 | 81.27 |

Every run is monotone stage by stage and the incremental J matches the full recomputation (`full_J`). Condensation
takes 20-35 s on a laptop CPU (a sweep 0.5 s, the medians 1.5 s). On this instance the structure term changes the
partition (D_S halves) without changing downstream accuracy beyond the seed noise; the sweep over densities /
coefficients / Q modes is the next step.
