# Within-cluster convex reconstruction

The experiment reads the saved hidden-distance winner and its cached partition
from a teacher_metric_grip sweep. It never repartitions nodes or changes the
mean teacher labels, teacher weights, student hyperparameters, or graph
preprocessing. Both arms train the existing two-layer GCN on identity adjacency
with the source experiment's uniform CE settings and paired student seeds.

For each cluster j, the representative is the convex combination of its member
S²X features. One coefficient per original node is optimized using a grouped
softmax; weights are nonnegative and sum to one separately in each cluster.
Initial coefficients reproduce the existing 30-step Weiszfeld geometric median.
This is a restriction to each cluster's S²X convex hull, not the raw X convex hull.

Targets are the saved geometric medians in the teacher's readout-input space
S ReLU(SXW₁+b₁). For an identity condensed graph, that same readout input is
ReLU(XcW₁+b₁). The implementation evaluates this expression directly so a GCN's
cached original adjacency cannot accidentally be reused. Gradients flow to
mixture coefficients while teacher parameters remain frozen.

Adam minimizes uniform mean squared hidden distance to these fixed targets.
The initial iterate is eligible; the best reconstruction-loss iterate is saved
without student validation or test feedback. This is nonconvex optimization
despite convex constraints on inputs, with no exact preimage guarantee. A
near-degenerate median mixture can also make softmax optimization slow.

The baseline and optimized representatives are evaluated with identical student
seeds and validation checkpoint rules; test is reporting only. Outputs include
per-cluster errors, reconstruction history, paired validation/test differences,
summary accuracy and student-seed SD, coefficients, features, and a figure.
Student-seed repetitions do not measure teacher/partition uncertainty. The source
partition and teacher hashes are checked; cached original artifacts are read-only.

Run tests/test_convex_representatives.py in Colab. Tests cover original adjacency
cache isolation, equivalence of the convex initialization to GRIP medians,
within-cluster support, teacher freezing, and reconstruction optimization.

## Raw-input mixtures

Set feature_source='raw' to use the dataset loader's original input X instead
of S²X. Arxiv X retains the existing training-set-fitted standardization; raw
here means before graph propagation, not before dataset preprocessing. The
partition, hidden targets, mean teacher labels, and student settings remain
the same saved hidden-distance experiment. Raw mixtures initialize from the
geometric median computed in X for each of those fixed clusters. Thus both
feature spaces use their own median initialization rule, not identical mixture
coefficients. Each includes its corresponding median control.

Both modes use the existing Adam solver for 1000 steps and select the lowest
reconstruction-error iterate, without accuracy feedback. Raw and S²X runs have
separate cache identities; existing S²X caches remain usable. Run each mode
with the same student seeds and compare both raw-minus-S²X paired accuracies
and the per-space convex-minus-median differences.
