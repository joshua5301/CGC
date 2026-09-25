# Paired GRIP student confusion analysis

`run_grip_confusion_study` reuses saved converged seed-cost partitions at equal
budgets. It trains the existing two-layer GCN with uniform soft CE and the supplied
fixed dropout/lr/weight decay. The same student seeds are used for both partitions.
No new clustering, teacher fit, or hyperparameter selection occurs.

The existing `_train_student` now optionally returns the model restored to its
highest-validation checkpoint; the default three-value API remains unchanged.
Test inference happens only after training and checkpoint selection. Saved labels
and predictions permit resumption without retraining, in protocol-hashed folders.

Confusion rows are actual classes and columns predicted classes. Recall is TP / true
support; precision is TP / predicted support and is NaN if never predicted. All
reported percentages use 0–100 units. Accuracy contribution is TP / total split
size times 100, so its classwise differences sum to the total accuracy difference.
Mean confusion counts may be fractional because they average student runs.

Paired differences use partition_seeds[1] minus partition_seeds[0], normally
1234 minus 0. Negative error_1_to_0/error_3_to_0 means fewer such mistakes.
Means and standard deviations describe student randomness conditional on two fixed
partitions; they are not variation over independently generated partitions.
This is a diagnostic of a previously proposed mechanism, not a new test-based
selection rule. It cannot by itself establish a causal effect of class allocation.
