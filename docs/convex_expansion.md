# Expanding the representative convex hull

This experiment loads a previous convex_representatives result, verifies its
partition/data/teacher, and reconstructs the previous input from its coefficients.
It first refines the within-cluster hull and then starts the all-node hull from
that refined solution. Both use identical Frank-Wolfe and Armijo backtracking
settings, fixed teacher readout-input targets and uniform reconstruction loss.
Student labels, partition and student hyperparameters remain fixed.

Each representative minimizes squared distance between ReLU(xW₁+b₁) and its
target. A linear oracle searches every permitted S²X row in blocks, selecting
the smallest gradient inner product. The update is (1-alpha)x + alpha*u, with
alpha in [0,1]. All nodes are candidates in the global run; active combinations
may be sparse but the candidate pool is never subsampled. There is no dense
K-by-N coefficient optimizer or N-by-N distance matrix. Saved vertex IDs and
step sizes, together with the initial combination, reconstruct the coefficients.

The stopping statistic is max_j <grad f_j(x_j), x_j-s_j> / max(1, ||target_j||²).
Default tolerance is 1e-6, with at most 20,000 additional steps per scope.
The ReLU objective is nonconvex and nonsmooth: a small gap is a first-order
stopping diagnostic using the chosen ReLU derivative, not a global optimum
certificate. It can miss improvements across activation boundaries. Status is
stationary, iteration_limit, or line_search_stalled; no iteration-limit result
is labeled converged. Histories include loss and gap. Oracle/search arithmetic
uses float64, while final representatives are cast to float32 for student training.

Progress checkpoints every 250 steps resume interrupted solves. Changes to the
solver configuration create separate result directories. Comparison includes
median, the previous 1000-step within-cluster result, refined within-cluster,
and refined global hull. Paired student seeds measure validation/test changes;
test never controls reconstruction or solver stopping. Larger hull guarantees
only a no-worse feasible optimum, not a better student. Here monotone accepted
steps also preserve the refined starting reconstruction objective numerically.

Run tests/test_convex_expansion.py on Colab for exhaustive block-oracle coverage,
restricted support, convex update reconstruction, monotone loss, stopping status,
and completed-cache reuse. No local numerical tests are required.
