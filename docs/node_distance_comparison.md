# Frozen-output comparison of four node distances

`src.node_distance_comparison.compare_node_distances` reuses the trained logits
from `gnn_distance_probe` and all-node teacher probabilities saved in
`teacher_predictions.npz`. It does not train or replay a student. This avoids
mixing the earlier accuracy-audit replays with the original model outputs.
The input graph/features are checked against the original distance fingerprint.
Predictions and teacher targets are fingerprinted in the new report.

The default evaluates GCN and mean GraphSAGE, with GIN available through
`models`. All source subset/model seeds are included. No held-out labels are
read here. The source GCN teacher was selected using validation accuracy;
the entire pipeline therefore is not validation-free.

## Distances

All four use the same saved probe nodes. Raw input features are retained;
there is no per-node L2 normalization. For each distance, divide by the median
over the same unordered probe pairs. A zero median falls back to the median of
positive pairs, or one if all pairs coincide. This scaling cannot affect
nearest-neighbor ranks or the selected medoids.

1. **GRIP S²X:** Euclidean distance in the original symmetric-normalized,
   self-looped two-step propagated features. Supply these through the existing
   `normalize_adj_sparse` implementation, alongside X and SX.
2. **Multiscale:** concatenate X, SX, ..., S^L X, dividing each block by its
   probe-pair median distance and the concatenation by sqrt(L+1).
3. **Probability OT:** let P_i be the uniform distribution on incoming
   neighbors. Default: remove self-loops; an isolated node has P_i=delta_i.
   Set d_0(i,j)=||x_i-x_j||/s_0, then

       d_(l+1)(i,j) = alpha d_l(i,j) + (1-alpha) W_(d_l)(P_i,P_j).

   Default alpha=0.5 and L=2. Both marginals have total mass one. Equal support
   sizes use exact assignment divided by support size. Unequal sizes use
   POT's unregularized network simplex, allowing fractional couplings. This
   is NOT the earlier blank-padded matching divided by maximum degree.
   For example, one copy versus two identical copies has zero transport cost.
4. **Neighborhood MMD:** begin with z_i=x_i. At each layer construct Gaussian
   random Fourier features phi_l(z), then average them over P_i. Apply the
   nonlinearity BEFORE averaging. Concatenate the previous z and the neighbor
   mean, each normalized by its probe-pair distance median and weighted by
   sqrt(alpha) and sqrt(1-alpha). This retains the root branch recursively.
   Gaussian bandwidth is the previous z's probe-pair median; 512 cosine/sine
   coordinates and seed 2026 are fixed defaults. This is a finite-feature
   approximation to neighborhood MMD, not exact Gaussian MMD or probability OT.

No metric parameters or RFF seeds are selected by CE/validation/test results.
Changing width, seed or depth is an explicit new comparison. S²X always stays
at depth two; other methods use the requested L (default two).

Probability OT computes the neighbor dependency closure of the requested probe
nodes backward through L layers. Intermediate distance matrices are stored only
on those node sets, with float64 row checkpoints. This is exactly the same
recursion as a full-graph matrix, restricted to the requested final pairs.
The closure can still approach the whole graph: this is a Cora-scale experiment,
not an all-pairs solution for Reddit. Exact OT runs on CPU even on a Colab A100.
Progress bars show measured throughput; no fixed runtime is assumed. Timing
tables report original construction time, including completed checkpoint work,
not the time to reload a cache. S propagation time is excluded.

## Representative protocol

Defaults: 13 and 70 representatives of the same 512-node probe pool. Students
were trained on their saved 70-node subsets of the entire graph. These two
budgets describe DIFFERENT objects. Neither representative budget is a complete
2708-node condensation experiment; 13/512 only approximates 70/2708 numerically.

For every distance/budget use the same ten restart seeds and up to 100
alternating k-medoids steps. Select the restart by its own average distance
objective, never model outputs or labels. This is local alternating k-medoids,
not a global optimum or a PAM swap algorithm. Tie handling keeps the current
medoid and assigns each medoid to itself, preventing empty clusters.

Two comparisons are returned:

- `own_representatives`: each distance chooses representatives and assignments.
- `S2X_representatives`: keep S²X's representatives, change only assignments.

The second comparison isolates assignment geometry under a common representative
set. It intentionally favors the baseline's choice of representatives, so read
it alongside the first comparison. Cluster labels are mean teacher targets;
weights are node counts divided by pool size. All methods use the same rule.

## CE measurements

Let q_i be the saved teacher target, p_i the saved student's softmax output,
r(i) its assigned representative, and delta_i=CE(q_i,p_r(i))-CE(q_i,p_i).

    risk_gap = |mean_i delta_i|
    E_CE = mean_i |delta_i|
    E_robust = mean_i max_c |log p_ic - log p_r(i)c|

Also report the signed risk change, both risks, the logit-range bound and the
class-centered logit bound. For these fixed outputs and arbitrary soft targets,

    risk_gap <= E_CE <= E_robust
             <= mean_i range(z_i-z_r(i))
             <= sqrt(2) mean_i ||center(z_i-z_r(i))||_2.

The representative risk equals cluster-mass-weighted CE with cluster-mean
targets EXACTLY, by CE's linearity in its target. This needs no linear-student
assumption. The inequalities do not assert that any of the four input distances
is a universal upper bound for an arbitrary trained GNN.

`all_probe` is the primary scope matching the earlier analysis.
`outside_student_train` removes that run's directly supervised probe nodes as
an extra diagnostic, without changing representatives or mappings. It averages
only the remaining nodes; the exported cluster weights/labels describe the full
probe scope. This is not a validation/test accuracy score.

Smaller E_CE is better. Pair each run with the SAME architecture, student subset,
student seed, budget, and scope for S²X. Negative `delta_E_CE` favors the alternative.
`win_fraction` counts strictly negative differences. Reported standard deviations
describe these crossed runs, not independent-sample confidence intervals.
Nearest-neighbor probability-TV gaps and class-centered-logit gaps provide a
second comparison that does not require representative selection.

`plot_node_distance_comparison` saves and returns CE bar plots (own versus fixed
representatives), paired CE differences, and neighbor-TV curves. CSVs, mappings,
cluster labels/weights, restart objectives, settings and distance caches are
saved. There is no learned condensed graph or student retraining here.

## Limits and checks

Mean-aggregation Lipschitz arguments motivate probability OT. The chosen alpha
does not certify the learned self/neighbor weight norms. Standard GCN's
symmetric degree normalization also differs from P; degree-mass terms would
be needed for a corresponding bound. MMD controls an RKHS function class,
not all Lipschitz nonlinear GNNs, and finite random features add approximation
error. The multiscale comparator only retains linear propagations.

Run `tests/test_node_distance_comparison.py` in Colab. It checks fractional OT,
duplicate-neighbor invariance, dependency closure equivalence, interruption and
resume, nonlinear-before-mean behavior, distance rescaling, nonempty medoids,
the CE inequality chain and mass-weighted identity, and a saved-output pipeline
with plot generation. Local validation is limited to syntax and diff checks.

References: [POT exact EMD](https://pythonot.github.io/all.html#ot.emd2),
[Weisfeiler-Lehman distance](https://arxiv.org/abs/2202.02495),
[MMD](https://www.jmlr.org/papers/v13/gretton12a.html),
[random Fourier features](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html).
