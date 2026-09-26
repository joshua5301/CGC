# Initial versus trained teacher kernels

This comparison uses the ORIGINAL GCN teacher that generated the cached student
targets. Its hidden width, output classes, seed and dropout are recovered from
the saved protocol. The trained checkpoint is verified against ALL saved teacher
probabilities (maximum absolute error <=1e-5). Original student predictions,
teacher targets and CE calibration/evaluation split remain unchanged.

Pass an explicit checkpoint or automatically search teacher-named `.pt`/`.pth`
files directly in the student run folder and its parent. Only safe weights-only
loading is used. If no matching checkpoint exists, the existing teacher fitter
is rerun in Colab with the original configuration and original train/validation
split. The selected epoch AND probabilities must agree, or execution stops.
No new hyperparameter search or teacher replacement is performed. Verified
replayed weights are saved for subsequent reuse. Test labels are not used.

The initial network is instantiated at that teacher's seed, not an unrelated
initialization ensemble. The teacher's own two-layer architecture is retained;
this is not a student-architecture-matched trained model for each student.
GCN/SAGE/GIN frozen students all evaluate the same teacher-based distances.

At both states compare pre-readout hidden features, exact readout trace NTK,
internal-parameter trace NTK, their sum, and final logits. Internal Jacobians
use the same projection seeds at both states. Readout is exact; internal uses
512 projections and two replicas by default. The trained-state full kernel is
a finite empirical NTK at the learned parameters, not an infinite-width kernel
or a claimed description of the full Adam training trajectory. Eval mode
disables dropout in both states. This experiment has one teacher seed, not an
ensemble over training runs.

Primary contrasts are trained hidden versus initial hidden and trained full
versus trained hidden. The latter tests whether Jacobians add value beyond
learned representations. The trained-logit comparator is particularly coupled
to the teacher-defined CE targets; improvements concern this teacher-based
surrogate and do not establish ground-truth accuracy or condensation gains.
No test accuracy is used for selection. The original teacher's validation-based
checkpoint selection is preserved and recorded.

Files: phase-wise Gram caches under `teacher_kernel_study/<hash>`, recovery
provenance, CE tables, and initial/trained comparison plots. Local checks are
syntax-only; checkpoint validation and numerical tests run in Colab.
