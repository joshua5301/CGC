# GRIP with an explicit contamination model

Teacher probabilities are deterministically converted to integer counts with a
fixed total n by largest-remainder rounding. Let u_i=counts_i/n. This is an explicit
proxy observation model: it does not claim teacher softmax outputs are independent
multinomial samples. Observations n controls both rounding and likelihood strength.

Conditional on fixed node features, counts follow a mixture of Mult(n,s_j) and
Mult(n,b), with clean prior rho_i. The background b is the mean of observed label
vectors on calibration nodes misclassified by the original teacher, smoothed with
five uniform pseudo-observations by default. If none are misclassified, it is
uniform. Increasing distance to labeled training nodes is mapped to decreasing
teacher correctness by isotonic regression on calibration nodes only; rho is
clipped to [.05,.95]. Correctness is a proxy for clean-component membership, not
an observed ground-truth contamination indicator. Neither b nor rho is updated
using clustering residuals. A single background captures aggregate class bias,
not every instance-dependent or multimodal error mechanism.

After dropping the common multinomial coefficient and observation entropy,
the assignment cost is

    ||x_i-c_j||/s_x + mu/s_q * (-logsumexp(
        log(rho_i)-n KL(u_i||s_j),
        log(1-rho_i)-n KL(u_i||b)))/n.

The common removed terms cancel in posterior probabilities. This is a scaled,
shifted exact multinomial mixture NLL, not posterior-weighted KL alone. The clean
posterior is softmax of the two log components. EM labels are posterior-weighted
means of u_i. The posterior is not normalized to have mean one and has no alpha
or tau. Relative log weights are stabilized separately per cluster during updates.

Feature-only FAISS initialization uses the same seed as baseline GRIP. Assignment
moves use the full mixture cost while preserving nonempty cells. Geometric median
updates are accepted per cell only if distance decreases. The full objective is
checked after every generalized EM iteration; increasing steps are rejected.
Stopping requires unchanged assignments, stable labels and a small objective change.
This is a local stopping rule, not a global optimality certificate. The saved
history, posterior mean and fraction below .1 expose collapse or stalled solves.

Three methods are evaluated: original GRIP, GRIP on rounded observations, and the
mixture. Rounded GRIP and mixture use the same observation-based normalization
scales. The n search can select different n for the two methods, so selected-row
comparisons are comparisons of tuned methods rather than a fixed-n intervention.
Per-trial files allow matched-n comparisons. Cross-method objective values are not
directly comparable. The original baseline has fewer search combinations.

The experiment reuses the existing two-layer GCN and uniform soft CE. Validation
is stratified into calibration and selection subsets; only selection chooses
settings and checkpoints. Test is used only for final evaluations. Search and
final student seeds are disjoint. Full-budget converged partitions alone are
eligible; failed candidates remain visible in trials.csv and partition artifacts.
Only transductive datasets are currently supported. Protocol-hashed outputs are
resumable, with exact split IDs and calibration parameters saved.

This implements a model-based robust clustering proposal. It does not establish
an unbiased noise estimator or a full-GCN risk guarantee. No local training or
numerical tests were run; run the supplied tests in Colab.
