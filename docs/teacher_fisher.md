# Teacher diagonal Fisher alpha sweep

Run teacher_fisher_grip.py after the existing Colab setup. Reuses the completed
GRIP_cora_fisher_diagnostic source, labels, baseline condensation and frozen
students. The teacher Fisher method itself does not need those students; they
are reused for the student-metric control and frozen prediction diagnostics.

The old diagnostic did not save its teacher parameters. Rebuild the same
full-landmark ReLU kernel feature transform and logistic teacher on train labels
only, with the original gamma. Check reconstructed probabilities against saved
teacher probabilities (maximum absolute error <=1e-7). Fold the transform into
the classifier and verify its probabilities too (<=1e-9). Abort on mismatch.
This reconstruction deliberately supports only the full-landmark ReLU source.

For z=P^2 X, compute p=softmax(t(z)/T) at the original label temperature. Keep
landmarks, kernel normalization, transform and classifier fixed. Differentiate
each class log probability with respect to query z, square each gradient,
weight by that class probability, sum classes, and average original nodes:

    s_k = mean_v sum_c p_c(z_v) * (d log p_c(z_v) / d z_k)^2
    w_k = (1-alpha) + alpha * s_k / mean(s)
    D_kk = sqrt(w_k)

The classwise derivatives are taken before summing, avoiding the zero expected
score. Teacher fitting is not differentiated. No test or validation labels enter
the metric. Query batches of 64 limit autograd memory. The score is a global
diagonal Fisher heuristic, not a certified KL approximation or risk bound after
averaging, diagonalization and mixing. Temperature affects the metric.

Sweep alpha=0,.05,.1,.3,.5,.8,.95. Alpha zero reuses exact original GRIP.
Include the previous student-centered-logit metric at alpha=.5 as a control.
For positive alpha, reuse unchanged GRIP on D z and map representatives back
with D inverse. Teacher labels remain fixed and condensed labels are cell means.
Transformed k-means initialization can change with alpha; seed is always zero.
Fail if GRIP drops nodes and violates the fixed condensed-node budget.

Fixed Cora .052, gamma=.001, T=10, mu=1, dropout=.9. Eight variants and independent
evaluation seeds 56--65 give 80 fits. Two-layer width-256 students train with
identity edges, uniform soft CE and the existing 1000-epoch optimizer schedule.
Independent validation-best checkpoints measure graphless prediction and raw
graph GCN transfer. Display every alpha, with no test-based winner selection.
Summary includes pointwise and Bonferroni paired intervals for 14 nonbaseline
versus GRIP comparisons across the two domains. These are exploratory,
conditional on one source condensation and one teacher. A later selected alpha
needs independent confirmation. Teacher Fisher versus student centered logits
changes both the source model and sensitivity definition.

Artifacts, sensitivities, teacher reconstruction parameters, checks, checkpoints
and detailed logs are saved to Drive; completed work is cached. Local smoke
tests and training were skipped at the user's explicit request.

```python
import os, subprocess
os.chdir('/content/GRIP')
r = subprocess.run(['git', 'pull', '--ff-only',
    'https://github.com/joshua5301/GRIP.git', 'main'], capture_output=True, text=True)
if r.returncode:
    raise RuntimeError(r.stdout + r.stderr)
```

```python
%run teacher_fisher_grip.py --diagnostic-dir /content/drive/MyDrive/GRIP_cora_fisher_diagnostic --output /content/drive/MyDrive/GRIP_cora_teacher_fisher
```
