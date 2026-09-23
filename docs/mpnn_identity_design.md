# Design proposal: Q=I condensation for a bounded class of local MPNNs

Status: a conditional derivation. The optimizable surrogate is implemented in
`src/partition_mpnn.py`; see `mpnn_evaluation.md` for evaluation. The complete
certificate is not numerically computed or empirically validated. It does not use
the evaluation student's GCN normalization matrix. No novelty claim is made.

## Scope and assumptions

Fix maximum depth K, bounded input/hidden-state domains and a class of local,
permutation-invariant message-passing layers. Each layer has uniform budgets:

- a_l: changing the root state by e changes the layer output by at most a_l e;
- b_l: changing one retained neighbor state by e changes it by at most b_l e;
- eta_l: deleting one neighbor changes it by at most eta_l.

These bounds must hold over the intermediate multisets used by the comparison,
not only the original graph's observed inputs. For bounded Lipschitz sum/mean,
max, and softmax-attention layers finite budgets can be established. This is a
class defined by sensitivity conditions, not a claim about unrestricted MPNNs,
hard discontinuous attention, arbitrary positional encodings, or global layers.
There is no assumption that aggregation is a mean or that messages are linear.

The derivation below first uses a common nonempty neighborhood convention with
one incoming neighbor per condensed node (its self-loop). Architectures with
different self-loop conventions can be represented on closed neighborhoods
with a self-edge flag and a common layer map that interprets that flag.

For degree-normalized GCN, the state must additionally carry fixed degree
metadata, e.g. log closed-degree. This lets a local layer compute both endpoint
normalization factors. Treat this as metadata for the proof, not additional
learned student inputs. Edge flags/attributes similarly need their substitution
costs. Omitting these would NOT prove the result for symmetric-normalized GCN.

## A fixed, feasible comparison, with no optimal transport solver

Let B be the unweighted adjacency of the common closed-neighborhood graph,
d_v=sum_u B[v,u]>=1, and P[v,u]=B[v,u]/d_v. P is a comparison measure, NOT an
assumption about the student's aggregation operator. The condensed graph is I.

Compare node v with condensed node j by retaining any one original neighbor u,
deleting the other d_v-1 neighbors, replacing the retained state with the state
of j, and replacing the root state. Triangle inequalities give:

    e_l[v,j] <= a_l e_{l-1}[v,j]
                 + b_l e_{l-1}[u,j] + eta_l (d_v-1).

This holds for every u, hence also after averaging uniformly over retained u:

    E_l <= (a_l I + b_l P) E_{l-1} + s_l 1_m^T,
    s_l[v] = eta_l (d_v-1).

The averaging is over valid edit comparisons. It does not replace SUM, MAX,
or attention inside the student with MEAN. It is a fractional feasible matching
to the single real condensed child, with the surplus matched to deletion slots.

For edge metadata with substitution coefficient c_l, add
`c_l sum_u P[v,u] ||edge_tag[v,u]-edge_tag_self||` to s_l[v].
With a product norm on hidden state and node metadata, initialize
`E_0[v,j] <= ||x_v-c_j|| + r_0[v]`, where r_0 is the distance between original
fixed metadata and the isolated condensed node's metadata. These extra terms
depend on the original graph, but not the representatives or partition.

## Separation into an optimizable term and a structural remainder

Define D(C)[v,j]=||x_v-c_j||, M_l=a_l I+b_l P, and M=M_K ... M_1.
Then

    E_K <= M D(C) + r_K 1_m^T,
    r_l = M_l r_{l-1} + s_l.

All model-class budgets must be fixed uniformly across the trainings under
consideration for r_K to be a condensation-independent constant. If one only
measures budgets after training, this is a per-model certificate instead.

Using a common envelope a_l,b_l<=L_l gives the convenient choice

    R=(I+P)/2,  C_K=prod_l (2 L_l),
    E_K <= C_K R^K D(C) + r_K 1_m^T.

For K=2, the geometric objective is thus based on

    R^2 D = (D + 2 P D + P^2 D)/4.

This compares root, one-hop, and two-hop raw-feature distances, without
averaging raw features before taking the norm. The geometry does not depend on
GCN/SAGE/GIN/GAT weights or on a chosen evaluation architecture. Its depth does.

A fixed rho in (0,1) is possible with R_rho=(1-rho)I+rho P and envelope
`c_l=max(a_l/(1-rho),b_l/rho)`. The overall certificate scale then changes with
rho. The first version uses rho=1/2, not a separate hyperparameter sweep.

## Risk and surrogate objective

For fixed teacher F and Yc[j] equal to its mean within cell j:

    R_F(g) <= mean H(F)
       + sqrt(2) C_K mean_v [R^K D(C)]_{v,a[v]}
       + mean_v KL(F[v] || Yc[a[v]])
       + sqrt(2) mean_v r_K[v]
       + sum_j (n_j/N) KL(Yc[j] || g_c[j]).

Logits are compared in Euclidean norm; a Lipschitz readout, if present, is
included in C_K and r_K. This is teacher-target risk on the original nodes,
not true-label population risk or a guarantee that test accuracy improves.

The proposed practical objective is

    J = mean_v { [R^K D(C)]_{v,a[v]} + mu KL(F[v] || Yc[a[v]]) }.

After dividing the certificate by sqrt(2) C_K, the exact relative KL coefficient
would be 1/(sqrt(2) C_K). Freely tuning mu gives a bound-derived surrogate.
No weight constraints or weighted loss are silently added to the evaluation
student. Its existing uniform soft-label CE is retained. With
eps_unif=mean_j KL(Yc[j]||g_c[j]), the theorem's weighted residual is at most
`(m max_j n_j/N) eps_unif`; log this imbalance multiplier.

## Efficient block optimization

1. Initialize m nonempty cells (same initialization across compared objectives).
2. Compute D(C) in representative batches and apply R K times using sparse
   matrix products. Assign nodes using propagated distance plus root-label KL,
   protecting nonemptiness and retaining ties to prevent gratuitous moves.
3. Set each Yc to the mean of F in its cell.
4. Let S be one-hot membership. Compute weights omega=(R.T)^K S. Update c_j
   to the weighted geometric median of ALL x_u with weights omega[u,j].
5. Recompute J and retain only non-increasing updates; stop at a tolerance/cap.

The center identity is

    sum_v [R^K D]_{v,a[v]} = sum_{u,j} [(R.T)^K S]_{u,j} ||x_u-c_j||.

No tree expansion, OT solve, student training, or dense P^K is needed during
condensation. Propagation costs O(K |E| m), initial distances O(N m f), and
median iterations O(I_med N m f), with working distance/weight storage O(N b)
for representative batch size b. Dense feature storage is additional. Median
updates can dominate; this is not an assertion of O(|E|) total runtime.

## What Q=I cannot solve

The structural remainder is unavoidable in general. Let every x_v=1, c_j=1,
and compare a node with d incoming neighbors against an isolated self-loop.
A one-layer SUM outputs d versus 1, although every propagated feature distance
is zero. Any valid bound for a class containing SUM must account for d-1.

In THIS construction the remainder is independent of C and assignment because
all condensed nodes have the same singleton topology and the deletion budgets
are uniform. It cannot be optimized away. This does not prove every possible
Q=I bound has a constant structural term, or that Q=I cannot perform well on a
task. It does mean our inexpensive common certificate may be loose on dense
or heterogeneous-degree graphs, and the surrogate need not reward topology
preservation beyond the distributions of features it propagates.

An optimized retained-child matching could tighten the edit bound, but may
encourage representatives to explain only convenient branches. A sparse Q or
multiple child prototypes is a different extension if a small uniform bound
for branching-sensitive models is necessary; it is outside this Q=I design.

## First validation, before a sweep

- Numerically verify the layer edit assumptions and whole-network inequality
  on small SUM, MEAN, MAX, attention and degree-normalized examples, including
  constant features and irregular degrees. Derive actual class budgets rather
  than fitting constants to the observed errors.
- Keep the evaluation student as the existing two-layer GCN with uniform loss.
  Compare propagated-feature GRIP, K=0 raw-feature clustering, and K=2 distance
  propagation with identical teacher, initialization policy, and training seeds.
- Log geometric cost, KL, cell imbalance, and (where certified budgets are
  available) structural remainder separately. A decreasing surrogate alone is
  insufficient evidence for a useful uniform bound.

Background, not a claim these papers prove this particular design:

- Chuang and Jegelka, Tree Mover's Distance (2022):
  https://arxiv.org/abs/2210.01906 . The main bound is for GIN; Appendix B makes
  additional architectural assumptions and its GCN example is mean-style.
- Jain et al., Subsampling Graphs with GNN Performance Guarantees (2025):
  https://arxiv.org/abs/2502.16703 . Related use of TMD bounds for data reduction.
