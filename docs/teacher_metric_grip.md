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
