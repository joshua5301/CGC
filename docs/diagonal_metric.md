# Global diagonal sensitivity scaling

Run `diagonal_grip.py` after the existing Colab setup. It consumes the completed
`GRIP_cora_fisher_diagnostic` artifacts without refitting the teacher or metric
students. No local training or smoke tests were run, as requested.

For propagated inputs Z=P^2 X, estimate each feature's sensitivity by averaging
the squared column norm of the centered-logit Jacobian over all original nodes
and frozen graphless students 23,24,25. Class-mean logits are removed before
computing Jacobians. Dropout is disabled; derivatives at ReLU zero use zero.
Only batches of node-by-class-by-feature Jacobians are materialized. No dense
feature-by-feature Hessian or Fisher matrix is needed.

Set w_k=(1-alpha)+alpha*s_k/mean(s). If all sensitivities are zero, use w=1.
This is an empirical sensitivity heuristic, not a certified Lipschitz bound.
Alpha is 0,.1,.3,.5,.8, so every dimension has a positive floor. Average weight
is one. Sensitivities and the weights actually used are saved as tensors.

For positive alpha, call the unchanged original GRIP partition routine on
Z*sqrt(w), then divide its geometric-median representatives by sqrt(w).
Cell labels are the teacher-probability means. Original GRIP normalization of
feature distance and teacher KL is retained, in the transformed space.
Geometric medians use the existing finite Weiszfeld iterations. The transformed
k-means initialization uses the same seed 0 across alphas, but its assignments
can differ because its geometry differs. Alpha zero reuses the exact previous
GRIP artifact. Empty-cell budget loss raises an error rather than silently
changing the student budget. No claim of globally optimal partitioning is made.

Fixed Cora .052, source gamma=.001, T=10, mu=1. Five variants and independent
seeds 46--55 give 50 fits. Students use hidden width 256, two layers, dropout .9,
uniform soft CE, 1000 epochs, the existing learning-rate schedule, and identity
edges during training. The original feature dimension is preserved. Separate
validation checkpoints evaluate graphless Z inputs and raw-graph GCN transfer.
The latter is an empirical transfer check, not part of a graphless proof.

All alpha results are reported; no test-based winner is selected. Pointwise
paired confidence intervals are printed; summary.json also stores Bonferroni
intervals for the eight positive-alpha vs GRIP test contrasts across two domains.
These intervals condition on the graph, split, teacher, and sensitivity models.
A future selected alpha needs fresh confirmation; this grid is exploratory.
Completed runs are cached. Detailed logs and checkpoints go to Drive.

```python
import os, subprocess
os.chdir('/content/GRIP')
r = subprocess.run(['git', 'pull', '--ff-only',
    'https://github.com/joshua5301/GRIP.git', 'main'], capture_output=True, text=True)
if r.returncode:
    raise RuntimeError(r.stdout + r.stderr)
```

```python
%run diagonal_grip.py --diagnostic-dir /content/drive/MyDrive/GRIP_cora_fisher_diagnostic --output /content/drive/MyDrive/GRIP_cora_diagonal_metric
```
