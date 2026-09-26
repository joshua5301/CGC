# Validation-only distance diagnostics

`src.tree_distance_analysis.analyze_tree_distances` compares cached exact rooted
tree distances with raw Euclidean distance, the existing GRIP `S^2 X` Euclidean
distance, and a degree-only control. This is a distance diagnostic, not a GCN
condensation performance experiment. No teacher or student is fitted.

Only training nodes serve as neighbor candidates. True validation labels score
the diagnostic and choose k/voting settings. Test labels and labels of other
nodes are never read by the analysis logic. The full unlabeled graph is still
used for transductive structural features. The cache fingerprint, matrix shape,
dtype, row completion, diagonal, and train-validation symmetry are checked.

The caller supplies `propagated_features` using the repository's
`normalize_adj_sparse` and two propagations on the original raw features. Tree
depth zero is the raw-feature baseline loaded from the existing cache. The
degree-only distance is `abs(log1p(degree_i) - log1p(degree_j))`, with degrees
excluding self-loops. No tree distances are recomputed.

## Classification and neighborhood checks

Every method uses the same k grid and the same uniform/inverse-distance voting
options. Inverse-distance weights are rescaled by the nearest distance for
numerical stability without changing the vote; if any selected neighbor has
zero distance, only selected zero-distance neighbors vote. Equal class scores
are resolved by the nearest member of a tied class, then training node ID for
equal distances, rather than always favoring the smallest class label.

- `val_accuracy`: percentage of validation nodes classified correctly.
- `val_macro_recall`: mean recall over classes represented in validation.
- `neighbor_purity`: percentage of selected neighbors with the query's label.
- `random_reference_purity`: expected purity under uniform training-node sampling.
- `neighbor_log_degree_gap`: mean absolute log-degree gap of selected neighbors.
- `random_reference_log_degree_gap`: the corresponding mean over all candidates.

`sweep.csv` preserves all settings. `best.csv` chooses validation accuracy within
each method, breaking exact ties by smaller k, then voting name. `per_class.csv`
contains every setting's class recall, train/validation support, and predicted
class counts; `best_class.csv` contains the selected settings only. Also inspect
the fixed-voting, same-k comparison instead of relying solely on the best rows.
These validation-selected maxima are exploratory scores, not unbiased estimates
of performance on new data. k-NN is deterministic; repeating student seeds is
irrelevant here. Test remains untouched for a later locked experiment.

## Degree controls

`diagnostics.csv` reports descriptive Spearman correlations between pairwise
distance and (a) absolute log-degree gap and (b) summed log-degrees, using all
validation-to-training pairs. Those pairs share nodes, so these are not treated
as independent observations and no significance p-values are reported.

`degree_matched_auc` compares same-class versus different-class training
candidates of exactly equal degree, separately for each validation query. It is
the fraction of such positive/negative comparisons where the same-class node
is closer, counting distance ties as one half. Comparisons are pooled within a
query, then query scores are averaged equally. Queries without eligible pairs
are excluded, and coverage is reported in `matched_queries` and
`matched_comparisons`. The degree-only control has AUC 0.5 whenever comparisons
exist. Values above 0.5 indicate label-relevant ranking beyond candidate degree
alone within these strata; this is not a causal isolation of every confounder.

The experiment can tell whether a distance supports class-relevant neighborhoods.
It cannot by itself establish that a partition or reconstructed condensed graph
will train a better GNN.

Run `python -m pytest -q tests/test_tree_distance_analysis.py` in Colab. Tests
check voting, zero distances, tie handling, degree-controlled AUC, scale
invariance, unused-label independence, and rejection of mismatched caches.
Only static checks were run locally.
