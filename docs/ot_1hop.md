# 1-hop neighbourhood-OT extension (`--edges ot_1hop`)

Opt-in reference implementation of the neighbourhood-OT objective on top of GRIP. The default path
(`--edges none`) is untouched.

## What is implemented

`src/partition_ot.py`

| block | update | exactness |
|---|---|---|
| assignment | sequential moves, strictly cost-decreasing (tol 1e-9), a node never leaves a singleton cell, ties keep the cell; exact pair W1 only for cells whose lower bound (distance of neighbourhood means) beats the incumbent | coordinate descent under the non-empty constraint |
| labels | `Y_cond[j] = mean F[cell j]` | forward-KL minimiser |
| adjacency + transport | one LP per cell over the common row `p_j` and every member coupling `Gamma_t` (HiGHS, unregularised) | global optimum of the block |
| features | weighted geometric medians, weights `omega[u,r] = alpha 1[a_u=r] + beta sum_t Gamma_t[u,r]` over **all** couplings; batched Weiszfeld with the Vardi-Zhang coincident-point step; each column keeps its best iterate (rollback) | fixed-coupling cost never increases |

Objective `J = mean_t [alpha ||H_t - H'_{a_t}|| + beta W1(nu_t, nu'_{a_t}) + mu KL(F_t || Y'_{a_t})]`, evaluated
exactly after every block (`w1_from='fresh'` re-solves one transport per node; `'gammas'` reuses the LP couplings,
valid only right after the LP block). No adaptive normalisation, no entropic regulariser, no sampling.

- `P`: `build_transition` binarises edges, removes the diagonal, row-normalises; a node without neighbours gets `P[t,t] = 1`
  (recorded in the returned metadata). `check_transition` validates an externally supplied `P`.
- `F`: teacher probabilities on `A^2 X` (the existing kernel teacher, training labels only), smoothed once with 1e-12 and
  renormalised, then fixed. `Y_cond` is clipped only inside `log`.
- Structural features `H` = raw `X` (the student's raw input); initial assignment = the GRIP partition on `A^2 X`.
- Output `P_cond` is row-stochastic and directed; edge list convention `P[j, r]` -> source `r`, target `j`
  (`transition_to_edges`). The student for this mode (`SAGE` in `src/models.py`) uses the given matrix as is:
  `x <- W_root x + W_nbr (P x)`, no self-loop insertion, no renormalisation; the original train/val/test graphs get the
  same `P` rule through `attach_transition`. `test_operator_direction` checks the sparse form equals `P @ H`.
- Guard: the largest cell LP must have at most `--max_lp_variables` variables (default 2e6); variable / constraint counts
  and the coupling storage estimate are printed before solving. Nothing is sampled or approximated when the guard trips —
  the run stops.

Flags: `--edges ot_1hop --root_weight alpha --neighbor_weight beta --label_weight mu --outer_iters k --max_lp_variables n`.
Tolerances are module constants (`MOVE_TOL`, `LP_FEAS_TOL`, `OBJ_TOL`, `EPS_PROB`, `COINCIDENT`, `MEDIAN_ITERS`).

## Checks (`python -m pytest tests -q`, 7 passed)

1. fixed OT toy: masses .5 at {0, 4} vs {1, 3} -> W1 = 1
2. cell LP: `p_j >= 0`, sums to 1, every coupling has the right marginals (residual < 1e-7), LP value <= the
   feasible fixed-`p_j` solution
3. weighted median: fixed-coupling cost non-increasing, coincident start point handled (heavy point stays the median)
4. tiny full run: finite J, no empty cells, stage-wise non-increase, valid probability rows
5. beta = 0: W1 term identically 0, `P_cond` untouched, still monotone
6. operator direction: `SAGE` sparse form == `P @ H`; no self-loops, rows sum to 1
7. isolated nodes / singleton cells / one-node graph valid

## Synthetic demonstration (`python demo_ot_synthetic.py`, N 60, d 4, C 3, m 6, alpha = beta = mu = 1)

All variants scored with the same J (same alpha/beta/mu, F, H, P):

| variant | root | W1 | KL | J | cells (min,max) | density | time |
|---|---|---|---|---|---|---|---|
| A  root+KL descent, isolated graph (`P_cond = I`) | 1.496 | 2.511 | 0.255 | 4.262 | (1,16) | 0.17 | 0.05 s |
| B  A + mean cell-transition `P_cond` | 1.496 | 1.969 | 0.255 | 3.720 | (1,16) | 0.97 | 0.48 s |
| C  A's partition/features, LP adjacency | 1.496 | 1.926 | 0.255 | 3.677 | (1,16) | 0.75 | 0.23 s |
| D  A's partition, LP adjacency + features alternated | 1.506 | 1.900 | 0.255 | 3.660 | (1,16) | 0.75 | 4.9 s |
| E  full alternating descent | 1.595 | 1.775 | 0.186 | 3.557 | (1,18) | 0.67 | 10.0 s |

E's 21 logged stages are monotone (J 3.858 -> 3.557); LP residuals ~1e-16. The LP adjacency is sparser than the
mean transition (density 0.75 vs 0.97), as expected of an LP vertex.

## Real graph: cora 1.3 % (m = 35)

`python main.py --dataset_name cora --ratio 0.013 --edges ot_1hop --outer_iters 3 --repeat 3` (CPU, teacher / hyperparameters
of the main table: relu, gamma 0.01, T 1, kl 2; alpha = beta = mu = 1; H = raw X, initial assignment = GRIP partition).

| stage | J | root | W1 | KL | notes |
|---|---|---|---|---|---|
| init | 8.012002 | 3.953413 | 3.992965 | 0.065624 | 2708 exact pair OTs |
| 1 assign | 8.009138 | 3.948625 | 3.992104 | 0.068410 | 132 moves, 23356 pair OTs (8.6 / node after pruning), 140 s |
| 1 labels | 8.008905 | | | 0.068177 | |
| 1 adjacency | 7.993794 | | 3.976993 | | 35 LPs, max 35105 variables, residual 1e-15, 18 s |
| 1 features | 7.978993 | 3.952039 | 3.958777 | | fixed-coupling cost 7.9256 -> 7.9113, 4 s |
| 2 assign / labels / adjacency / features | 7.977490 / 7.977314 / 7.976920 / 7.974420 | | | | 95 moves, 74 s / 24 s / 3 s |
| 3 assign / labels / adjacency / features | 7.973952 / 7.973862 / 7.973525 / 7.972308 | | | | 46 moves, 79 s / 26 s / 4 s |
| polish | 7.972270 | 3.948754 | 3.950875 | 0.072641 | |

Monotone over all 17 stages; total condensation 545 s (assignment blocks ~300 s, exact J evaluations ~120 s, LPs ~80 s,
medians ~10 s). `P_cond` has 66 nonzeros out of 35 x 35 (about two neighbours per condensed node, LP-vertex sparsity).

Downstream (same `SAGE` student, same raw `H`, `F`, initialisation, dropout grid of the main table, 3 seeds):

| condensed graph | test acc |
|---|---|
| beta = 0: root + KL descent on raw X, `P_cond` = mean cell transition (427 edges) | 81.67 +- 0.06 |
| beta = 1: full alternating descent (66 edges) | **83.13 +- 0.25** |
| reference, main table: GRIP on `A^2 X`, `A' = I`, GCN student | 84.3 +- 0.3 |

The OT adjacency + feature blocks add +1.5 over the same pipeline without them; the GCN-on-propagated-features number
is a different student and feature level, listed only for reference. Validation was used for the dropout choice only;
alpha / beta / mu were not tuned.

## Limitations / blockers

- Exact pair OT through `scipy.optimize.linprog` costs 5-15 ms per call (solver overhead dominates at these sizes);
  the assignment block needs one exact OT per node plus the pruned candidates (cora 1.3 %: 8.6 per node), the exact J
  evaluation another N. Measured on cora 1.3 %: ~100 s per assignment block, ~40 s per exact evaluation, ~20 s per LP
  block. This is the bottleneck; POT's network simplex (`ot.emd`) would be the drop-in exact replacement but is not
  a dependency of the repository.
- The per-cell LP has `m * sum_{t in cell} deg(t) + m` variables; arxiv / reddit densities are far beyond the guard.
- Downstream accuracy is a separate transfer question (the student that preserves the row-stochastic operator is
  `SAGE`; the main-table GCN is not evaluated on `P_cond`). No downstream claim is made from J alone.
- This is the 1-hop normalised neighbourhood-OT objective, not tree mover's distance; no guarantee for arbitrary GNNs.
