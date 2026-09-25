# Heuristic distance weighting with full GRIP tuning

Use run_experiments(method='grip_distance', search='grid'). The teacher kernel,
gamma, T, basis, kl_weight, dropout, lr and weight_decay use the existing search
API. Additional grid parameters are distance_k and distance_power.

Let d_i be mean squared distance in H=A^2 X to k labeled training nodes, excluding
the node itself. Let s be the median positive d_i (one if all distances are zero).
Weights are proportional to (1+d_i/s)^(-power), normalized to mean one. Evaluation
uses stable log weights with a numerical floor of 1e-12. Power zero bypasses the
weighted path exactly. No calibration labels, isotonic fit, tau or alpha are used.
This is a heuristic reliability model, not a proven inverse noise variance.

The GRIP feature term and its original normalization remain unchanged. The node
weight multiplies KL and determines the weighted representative label mean. The
same two-layer GCN and uniform CE evaluate both methods. Full validation is used
for settings and checkpoint selection; it is no longer split for calibration.
Do not compare these validation numbers directly to split-validation studies.

Tune original GRIP and weighted GRIP separately over the same teacher/student
grid and seeds. Weighted GRIP has additional distance choices and thus a larger
search budget. Test does not select settings. Teacher/partition caches and resumed
grid search use the existing protocol-hashed output folders. Feature distances are
cached per dataset and k. Returned convergence and requested_nodes should be
inspected: the existing run_experiments evaluator reports capped or reduced-budget
partitions without automatically excluding them. Local checks are syntax/diff
only; execute numerical tests in Colab.
