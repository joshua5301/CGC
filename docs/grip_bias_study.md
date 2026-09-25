# Class-0 suppression diagnostic

The trained two-layer GCN and uniform soft CE are unchanged. After restoring the
unadjusted validation-best checkpoint, predict argmax(log_softmax - delta e_0).
Delta=0 is the baseline; positive delta makes class 0 harder to predict without
changing the relative scores of any other classes. No model parameter is updated.

For each density and partition seed, one common delta is selected from a supplied
grid by mean validation accuracy across calibration student seeds. Ties choose
the smallest delta. The delta is then frozen for disjoint final student seeds;
their checkpoints are also selected on unadjusted validation accuracy. Test scores
never select delta or checkpoints. The validation node set remains the same across
the two student-seed groups; these are not independent validation datasets.

Both partition seeds 0 and 1234 receive the same calibration procedure. Outputs
include raw/corrected results, paired differences, macro recall, class-0 precision
and recall, and 1/3-to-0 errors. A positive boundary flag means the selected delta
is at the supplied grid maximum, not that larger deltas necessarily help.

Cached confusion runs now retain log-probabilities. Old runs saved only predictions,
so a new code revision retrains in a new protocol-hashed directory. Subsequent runs
with the same model settings reuse those files, including when only the delta grid
changes. This diagnostic quantifies how much a single output offset can reproduce;
it does not prove that representation quality or class allocation caused the gap.
