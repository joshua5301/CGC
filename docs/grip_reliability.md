# Distance-based label reliability for GRIP

The teacher probabilities remain fixed. Weighted GRIP minimizes mean feature
distance / original distance scale plus mu times mean w_i KL(q_i || s_a(i)) /
original KL scale. Weights are positive and normalized to mean one. Centers remain
unweighted geometric medians; representative labels are weighted arithmetic means.
Student training is the existing two-layer GCN with uniform soft CE.

The study currently accepts transductive datasets, including Cora, Citeseer and
ogbn-arxiv. It rejects inductive datasets rather than mixing feature spaces.
Distance is mean squared Euclidean distance in the existing H=A^2 X representation
to k labeled training nodes. Self neighbors are excluded for training nodes.
No validation or test node is used as a labeled distance reference.

Validation is stratified into calibration and selection subsets. The teacher is
fit using training labels only. Increasing isotonic regression maps distance to
teacher Brier error using calibration labels only. Predictions outside its fitted
distance interval use boundary values. Raw weights are clip(1/(tau+error), lo, hi),
then divided by their mean; the final normalized weights need not lie in [lo, hi].
This estimates expected prediction error, not a certified inverse noise variance.

Each partition seed is compared separately. Both baseline and reliability methods
search the same mu/dropout grid, selection nodes and student seeds. Reliability
additionally searches k/tau, so it uses a larger search budget. Baseline uses the
unchanged partition path. Only converged full-budget partitions are eligible.
Fresh student seeds evaluate the selected setting; checkpoint selection uses the
selection subset, and test is evaluated only for the selected settings. Validation
numbers are not directly comparable to earlier experiments using all validation
nodes. Calibration parameters must not be chosen using test metrics.

Outputs include complete protocols, exact split IDs, teacher probabilities,
distances, fitted isotonic curves, weights, cached partitions, per-trial validation,
per-student final results, and paired reliability-minus-baseline differences.
Diagnostics compare held-out selection Brier prediction MSE against a constant
calibration-mean predictor, and report distance/error Spearman correlation.
These diagnostics are not used by the weight fit. Negative correlation or no MSE
improvement challenges the increasing-distance error assumption.

Different reliability weights define different J objectives, so cross-method J
values are not evidence of improvement. The original normalization scales are
held fixed within a teacher configuration. This change does not establish a full
GCN risk guarantee or prove the label-noise model. Only syntax and diff checks are
run locally. Numerical and regression tests are supplied for Colab.
