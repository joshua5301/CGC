# Student-matched empirical kernels

The runner reads hidden width, dropout and layer count from the ORIGINAL saved
student protocol. Class/output width is read from teacher probabilities and
checked by reproducing all saved students' initial logits. It uses the same
ProbeGNN constructor, native PyG initialization, biases, GIN epsilon, aggregation,
self-loop convention and original graph. Width/output-size overrides are not
accepted. For each architecture and original model seed, initial logits must
match every corresponding saved run; otherwise the experiment stops.

Kernel networks use independent seeds, not evaluated students' seeds. Hidden
features remain pre-readout: h dimensions for GCN/GIN, 2h for SAGE's neighbor/root
branches. Their dimension is NOT compressed to the number of classes. The
classifier has the original number of classes so the full channel-averaged
empirical NTK differentiates the correctly shaped student architecture.

Compare exact hidden/readout kernels, internal-parameter NTK and their sum,
plus the random final-logit kernel, as in the preceding readout study. There is
no per-architecture winner selection or fitted group weighting. All original
student architectures are evaluated, and previous distances are preserved.

Alignment is to the student's deterministic EVAL forward map at initialization.
Dropout probability is taken from the student, but dropout is disabled in eval,
exactly as when the cached outputs were evaluated. This is not the expected
stochastic training kernel, an Adam-preconditioned kernel, or the trained NTK.
The original Adam/weight-decay training dynamics are NOT claimed to equal this
fixed Euclidean parameter kernel. These distinctions are recorded in metadata.

The preceding CE calibration/evaluation split, thresholds, teacher targets and
student outputs remain unchanged. No student retraining or test-label use occurs.
Tests run in Colab; local verification is limited to parsing and diff checks.
