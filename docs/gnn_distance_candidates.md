# Nonlinear node-distance candidates

`src.gnn_distance_candidates.run_candidate_distance_study` adds eight distances
to the five existing local-CE comparators. Defaults evaluate the original saved
GCN, GraphSAGE and GIN students, ten runs per architecture. No student is
retrained and no exact OT matrix is recomputed. The distance-generating networks
are newly initialized, frozen, label-free models, separate from evaluated students.

## Candidates

### Random GNN features: rf_gcn, rf_sage, rf_gin

Use the existing two-layer `ProbeGNN` architecture with hidden and output width
128, dropout disabled, and eight initialization seeds 2000 through 2007. The
last outputs are embedding channels, not class probabilities. For model m let
h_m(i) be its width-w output at original-graph node i. Concatenate

    phi_RF(i) = [h_1(i), ..., h_M(i)] / sqrt(M w).

Euclidean distance in this embedding averages squared representation differences
over channels and initializations. It retains per-node amplitude; there is no
per-node L2 normalization or removal of degree effects. Each architecture supplies
one distance, evaluated against ALL three student architectures. This is a finite
random-feature kernel, not an exact infinite-width Gaussian-process kernel or a
bound for all trained GNNs. Initializers are the existing PyG/PyTorch defaults.

### Projected empirical NTK: entk_gcn, entk_sage, entk_gin

Use scalar-output two-layer networks with hidden width 64 and initialization
seeds 3000 and 3001. For each initialization, draw 64 independent Rademacher
directions r in the ENTIRE trainable parameter vector and compute

    phi_theta(i)_r = <gradient_theta f_theta(i), r> / sqrt(R).

Concatenate the two sketches and divide by sqrt(number of initializations).
Since E[r r^T]=I, the sketch inner product is an unbiased estimator of the
finite-width scalar-output empirical NTK. Squared sketch distance estimates
squared gradient distance. The finite sketch, and taking square roots to form
a distance, introduces approximation error. Individual sketches are cached.

JVPs use automatic differentiation, not finite differences. The implementation
uses `torch.autograd.functional.jvp` (double backward) with `functional_call` to
substitute parameters. GCN normalization caches are populated before computing
derivatives; graph normalization is independent of parameters. GIN epsilon
parameters and trainable biases are included. Directions have covariance I in
the modules' native parameterization; there is no parameter-group reweighting.

This is NOT the analytic infinite-width GNTK. It measures local parameter
sensitivity at initialization, not nonlinear feature learning throughout training.
The scalar random readout is independent of dataset classes and teacher labels.
Direction RNG seed for network seed s is projection_seed+s (default 4000+s).

### sum_mmd

Reuse the original recursive RFF mean-embedding construction, replacing each
neighbor average by a sum. Keep root weighting, width, RFF seed, depth and
self-loop policy from the original comparison. Apply the same label-free
bandwidth and block-median rules in each layer. Later bandwidths/scales naturally
change because previous sum embeddings differ. This controls the construction
rule, not every numerical bandwidth in the recursion.

Unlike mean MMD, duplicate neighbor mass affects the embedding. Scalar block
normalization does not remove relative degree differences between nodes. As in
the old mean comparator, an isolated node uses itself as its neighborhood; this
is not literal GIN's empty-neighbor convention. This comparator tests mass
retention and is not an exact GIN representation.

### scattering

With the same row-stochastic neighbor operator P as probability OT, build

    Z = [P^2 X, P |X-PX|, |PX-P^2 X|].

Absolute values are elementwise. Divide each block by its probe-pair distance
median, concatenate, and divide by sqrt(3). All three blocks use at most two-hop
information. This is a small scattering-inspired node embedding, not the full
published graph scattering transform and not a claimed instance of its stability
theorem. It adds rectified band-pass information to a low-pass representation.

The neural candidates and scattering use two message-passing/hop levels. Sum MMD
uses the original comparison's depth (two in the provided Cora cell). Widths,
initializations and all block rules are fixed without selecting by validation,
test or CE-preservation scores. Numerical experiments belong in Colab only.

## Evaluation and provenance

The local evaluation is unchanged: 128 calibration / 384 evaluation nodes by
default, quantiles 1/2/5/10%, k=1/5/10/20, directed CE replacement changes, and
one L1-optimal nonnegative scale fit on all calibration pairs per student/distance.
Distance normalization uses calibration pairs. As with earlier mean MMD and
multiscale embeddings, label-free block bandwidth/scale estimates can use the
full probe geometry; no evaluation CE enters distance construction.

The earlier comparison protocol listed GCN and SAGE, but its original probe
protocol included GIN. `saved_student_fingerprints` now enumerates the original
probe's saved subset/seed grid. Previously fingerprinted files must match; newly
included original GIN files receive fingerprints in the new result protocol.
It never loads the `accuracy_audit` replay directory. Missing files raise an error
instead of silently omitting a student or retraining it. Passing `models` limits
the architectures explicitly; otherwise local analysis uses all original models.

`health.csv` records training CE/teacher agreement when available, probe teacher
agreement/CE, confidence and entropy. These are diagnostics against teacher soft
labels, NOT ground-truth validation/test accuracy. GIN is reported even when its
health diagnostics are worse. The GCN teacher's architectural bias remains.

The candidate generator caches each neural initialization and the other distance
matrices under hashes of inputs, settings, framework versions and device. TF32 is
disabled. Initialization RNG states are restored after generation. Student seed
overlap with neural distance initialization seeds is rejected. Independence of
seed values is a protocol separation, not a claim that architecture priors match
all students. Cached timing is original embedding construction time; neural
pairwise Euclidean distance construction is excluded from that timing.

The result includes CSVs, full probe-order distance/CE matrices and two heatmap
figures (quantile and k-NN). Each figure has student architectures in columns and
mean CE change, p95 CE change and calibration MAE in rows. Colors share a scale
across architectures within each metric row. All scores use the same per-student
all-evaluation-pair CE mean denominator. Smaller is better. No method is declared
best using these diagnostics and then evaluated on the same pairs as a new
generalization test; these remain exploratory comparisons.

## Checks

Run `tests/test_gnn_distance_candidates.py` and `tests/test_local_ce_distance.py`
in Colab before the experiment. Tests compare projected NTK features to explicit
parameter gradients for GCN/SAGE/GIN, include tiny CUDA interface checks, check
RF permutation equivariance and RNG preservation, sum-versus-mean mass retention,
the scattering formula, cache reuse, and enrollment of original GIN outputs.
Local checks are syntax/diff only; the Colab cell shows full pytest output if a
check fails. Candidate defaults are exploratory, not empirically validated winners.

References: [random graph features](https://proceedings.mlr.press/v119/zambon20a.html),
[GNTK](https://proceedings.neurips.cc/paper/2019/file/663fd3c5144fd10bd5ca6611a9a5b92d-Paper.pdf),
[diffusion scattering](https://arxiv.org/abs/1806.08829),
[PyTorch JVP](https://docs.pytorch.org/docs/2.8/generated/torch.autograd.functional.jvp.html).
