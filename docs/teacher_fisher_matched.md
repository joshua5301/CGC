# Matched alpha ablation

Run teacher_fisher_matched.py after teacher_fisher_full.py has completed. The
fixed configuration is Cora .052, gamma=.001, label T=1, mu=10, Fisher tau=1,
dropout=.9. Requires both source evaluation domains to have selected the same
alpha=.05 configuration. Reuses its frozen teacher and confirmation scores,
and constructs alpha=0 using the unchanged scaled-GRIP routine with all weights
one. Teacher labels, seed-zero initialization rule, budget and student protocol
are matched. Alpha=0 is not independently tuned GRIP.

Checks source code/helper hashes, runtime, data, teacher, positive artifact,
configuration digests and source run configurations. The separate output folder
preserves all full-sweep artifacts. Missing source confirmation runs produce an
error instead of silently mixing experiments. Reduced baseline node budget is
also an error.

Only ten additional baseline fits on source confirmation seeds 69--78 are
needed. Both graphless and raw-graph GCN domains use independently selected
validation checkpoints on the same training trajectory. Prints mean +/- sample
SD and paired alpha=.05 minus alpha=0 changes. summary.json stores paired val
and test statistics, wins/ties/losses and two-domain Bonferroni test intervals.
CSV contains both variants' per-seed scores. Completed baseline fits are cached.

This is an ablation using already reported confirmation seeds, not a new
independent confirmation or a cross-split generalization experiment. Local
training and smoke tests are intentionally skipped at the user's request.

```python
import os, subprocess
os.chdir('/content/GRIP')
r = subprocess.run(['git', 'pull', '--ff-only',
    'https://github.com/joshua5301/GRIP.git', 'main'], capture_output=True, text=True)
if r.returncode:
    raise RuntimeError(r.stdout + r.stderr)
```

```python
%run teacher_fisher_matched.py --source /content/drive/MyDrive/GRIP_cora_teacher_fisher_full --output /content/drive/MyDrive/GRIP_cora_teacher_fisher_matched
```
