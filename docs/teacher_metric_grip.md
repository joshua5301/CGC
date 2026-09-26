# GRIP condensation with frozen teacher geometry

`src.teacher_metric_grip.py` runs a resumable Cora grid search using the saved
GCN teacher from the output-preservation experiments. It verifies the original
graph and reproduces saved teacher probabilities before extracting all-node
features. It does not change existing teacher probabilities or student caches.

The compared embeddings are original GRIP `S²X`, the trained GCN readout input
`S ReLU(SXW₁+b₁)`, and its logits. All use the same frozen GCN teacher soft labels
`softmax(logits/T)`. Thus `s2x` controls the partition geometry, but is not the
original kernel-teacher GRIP baseline. Gamma and basis are not parameters of
this fixed teacher. Hidden/logit Euclidean distances match the preceding
diagnostics up to a global factor, removed by GRIP distance normalization.

Existing GRIP minimizes normalized Euclidean distance plus normalized forward
KL, updating geometric medians and mean soft labels. Teacher-space centers
cannot be passed to a student that expects original input features. For each
resulting cluster, the experiment therefore computes a geometric median in
original GRIP `S²X` space. The student uses those original-dimensional features,
cluster mean labels, and an identity condensed adjacency, exactly as in the
existing GRIP evaluator. This is a concrete condensation heuristic, not an
inverse of the teacher embedding or a guarantee that teacher distances survive
realization. The original-space baseline uses its existing GRIP medians.

The student remains a two-layer GCN trained with uniform CE. Grid selection uses
mean best-checkpoint validation accuracy across search seeds. Every mode's best
configuration is frozen before any final test evaluation. Final seeds are
disjoint from search seeds; reported standard deviations are population SDs
across student seeds, conditional on one fixed teacher and partition seed.
The teacher itself used validation checkpoint selection; no test labels enter
training or selection. Test scores are held-out reporting, not tuning criteria.

Protocol hashes isolate different grids, seeds, teacher weights, data, and
student settings. Partitions are cached independently of student parameters.
Per-trial and per-final-seed results resume after interruption. Outputs include
summary.csv, final_seeds.csv, per-mode trials.csv/best.json, partition tensors
(assignments, metric centers, realized features, labels, counts), and accuracy.png.

Run tests/test_teacher_metric_grip.py on Colab before the sweep. No local
numerical experiment or smoke test is required.

## Full parameter gradient distance

Set modes=('full_gradient',) to sweep the trained teacher's output-Jacobian
distance under the same condensation/evaluation protocol. This is a logit
Jacobian, not a CE-loss gradient. All parameter groups contribute, with native
Euclidean parameter weighting and output-channel averaging.

The readout contribution is computed exactly as the hidden-feature Gram plus
the constant bias Gram. Internal parameters use Rademacher parameter-direction
JVPs, matching the prior teacher-kernel comparison. With C output channels,
R projections per seed and L seeds, concatenate hidden features, the constant
bias feature, and each seed's JVP block divided by sqrt(R*C*L). Its inner
products equal the earlier exact-readout-plus-sketched-internal kernel up to
floating-point error. Default R=512 and L=2 give 7425 features for Cora's
256-wide, seven-output teacher. No explicit all-node/full-parameter Jacobian
or N-by-N distance matrix is required. This is a sketch approximation of the
full gradient distance, not an exact Jacobian calculation.

Sketch seeds and dimension are fixed across hyperparameters. Features are
cached per teacher/data/library configuration independently of the grid and
density, with one atomic file per sketch seed. Existing s2x/hidden/logits runs
retain their protocol hashes and resume behavior. Use a separate output root
for concurrent gradient and earlier sweeps. Run tests/test_teacher_gradient_features.py
in Colab to verify equivalence with the previous kernel and cache reuse.

## Arxiv and reusable teachers

`prepare_metric_teacher(dataset, output_dir, **settings)` trains and caches a
two-layer GCN on original training labels, selecting its checkpoint by validation
accuracy then CE. It supports cora, citeseer, and arxiv. Teacher artifacts include
all-node probabilities/logits, checkpoint, and validation history. Training and
checkpoint selection never receive test labels. Changing teacher settings or
training/validation supervision creates a new cache directory.

Pass the returned folder as teacher_run and dataset='arxiv', ratio=0.0025 to
run_teacher_metric_grip for the repository's 454-node budget. The Arxiv loader
retains existing preprocessing (undirected edges and feature standardization
fit on training nodes). Each distance uses the same fixed teacher. Existing
Cora teacher folders and sweep hashes remain compatible. Partitions use all
nodes and teacher probabilities; student supervision remains uniform CE on
the representatives, with original-graph validation/test evaluation. No dense
all-pairs distance matrix is formed.

## Full grid with raw convex representatives

Set representative='raw_convex', modes=['hidden'], reconstruction_steps=1000,
and reconstruction_lr=0.05. Every distinct (T, kl_weight) case recomputes teacher
soft labels and the GRIP partition, then optimizes a within-cluster convex
combination of original input X toward the teacher hidden centers. It starts
from raw-space geometric medians. The representative labels remain the mean
teacher probabilities of the resulting clusters. The best Adam iterate is
selected only by hidden reconstruction error.

These reconstructed representatives are used inside the validation objective,
not merely after selection. Student dropout/lr/weight-decay combinations share
the reconstruction cache when T and KL weight match. Final student seeds remain
disjoint from search seeds and test is evaluated only after configuration
selection. Per-partition histories and coefficients are saved; summary includes
initial/final reconstruction error and selected step. Old median experiment
hashes are unchanged. Expanded search ranges make comparisons to older winners
descriptive; strict search-budget comparisons require matching grids.

Set representative_labels='mixture_mean' with representative='raw_convex' to
use each partition's learned convex coefficients to mix its teacher probabilities
before every student validation trial. Reconstruction still targets the frozen
hidden centers; changing labels does not change that optimizer. Temperature is
applied before mixing. Mean labels are retained as cluster_mean_y in the saved
partition; y stores the actual student targets. Uniform student CE is unchanged.
The label rule has a separate protocol hash, preserving all earlier caches.
