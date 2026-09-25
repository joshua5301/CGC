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
