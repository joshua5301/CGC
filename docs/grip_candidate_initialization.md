# Matched candidate-pool initialization

`run_experiments` accepts `grip_candidates='train'` or `'random'` with
`method='grip', grip_init='kmeans'`. Both pools have the number of training
nodes. The random pool samples all nodes without replacement, including
training nodes when selected, using `grip_candidate_seed`. It is not stratified.

Both variants greedily select K distinct candidate node IDs to minimize
the sum, over all graph features H, of squared distance to the nearest selected
center. The first center minimizes total squared distance. Subsequent centers
minimize the remaining coverage cost. Ties use ascending node ID. This is a
greedy approximation, not a global optimum. The distance matrix uses N times
the candidate count memory and is intended for small citation graphs.

The selected features initialize the existing CPU FAISS K-means. Centers then
move freely, followed by unchanged GRIP and unchanged teacher targets.
True class values are never used in center selection; only training membership
defines the train pool. No hard-label replacement or class quota is applied.

The same cached K-means assignment is used for all hyperparameters within a
case. `candidate_initialization.pt` records pool IDs, selected IDs, greedy
coverage history, and the K-means assignment. Protocol hashes include pool
source and seed. K must not exceed the training-node count. At K=120 for
Citeseer, every candidate is used; coverage selection only determines order.

Use the same grid and student seeds in both arms and select each best setting
using validation only. Test is evaluated only for the selected setting.
One random-pool seed is a controlled comparison, not evidence of robustness
over random pools. The existing runner reports convergence and actual budget;
it does not exclude nonconverged or reduced-budget runs from selection.

Local numerical tests were not run. Run `tests/test_grip_candidates.py` in Colab.

## Direct GRIP initialization

Set `grip_candidate_refinement='grip'` to skip K-means entirely. Candidate
selection is unchanged. Each selected node supplies its feature and teacher
probability as the initial representative. The first assignment minimizes
the existing normalized distance plus KL cost for the current teacher and mu.
Selected nodes are assigned to their own zero-cost representative, resolving
duplicate-point ties and roundoff. This guarantees occupied cells only at the
first assignment; subsequent GRIP updates and empty-cell removal are unchanged.
Then geometric medians and mean teacher labels are computed and the existing
GRIP loop runs. Training labels never replace teacher probabilities.

Only selected IDs are cached across teacher configurations in direct mode;
the initial assignment is recomputed for each gamma, temperature, kernel and mu.
The refinement mode is recorded in the protocol and summary. Dataset-specific
output directories allow separate Colab sessions to run Cora and Citeseer.
