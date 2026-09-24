# Proposed-method full sweep

`teacher_fisher_full.py` independently fits Cora .052 teachers from train labels;
no previous diagnostic files or student ensemble are required. Uses propagated
Z=P^2 X, ReLU kernel teacher, global diagonal input Fisher at fixed tau=1, scaled
GRIP partitioning and inverse-scaled geometric-median representatives. The
teacher is fit once per gamma, and its logits and Fisher are cached. Label
temperature T is applied only to logits for soft labels; D is reused across T,
mu and student seeds. The kernel and Fisher derivatives follow the previous
implementation, with fixed full-node landmarks and fixed teacher parameters.

Defaults:

* gamma: .001,.01,.1,1
* label T: .1,.2,.5,1,2,5,10
* mu: 0,.1,.2,.5,1,2,5,10
* alpha: .05,.1,.3,.5,.8,.95

Positive alpha only: no original-GRIP or student-sensitivity baselines. This is
1344 settings and 4032 selection fits on seeds 66,67,68. Fixed tau=1 and dropout
.9. Each fit trains a two-layer width-256 GCN with identity edges and uniform
soft CE for 1000 epochs using the existing optimizer schedule. Evaluation
independently selects validation checkpoints for graphless Z inputs and raw
graph GCN transfer. One trajectory supplies both evaluations.

Select the largest mean validation score separately for each domain, breaking
ties by ascending gamma,T,mu,alpha. Test scores never enter selection. Confirm
the one or two unique selected configurations with seeds 69--78 (10 or 20 more
fits). Report each domain's own selected configuration only. These are fresh
student seeds, not new data splits, teachers or condensation initializations.
There is no baseline superiority claim in this proposed-method-only sweep.

Reuses the original GRIP distance/KL normalization in the scaled geometry and
seed-zero transformed-space k-means initialization. Settings where empty cells
reduce the requested budget are excluded and recorded in invalid_settings.json.
Other failures stop execution. Finite Weiszfeld iterations are inherited from
GRIP; no exact optimum or certified risk bound is claimed.

Code, runtime, data, teacher and artifact hashes protect resume caches. Every
finished fit is saved immediately. Selection summary and student_runs.csv are
updated after each setting; confirmation rows after each fit. Checkpoints are
saved for confirmation fits only. Final output is two concise rows; epoch logs
go to a timestamped Drive log. No local training or smoke testing was performed
at the user's request; verification was static code review and diff checks.

After the existing Colab dependency and Drive setup:

```python
import os, subprocess
os.chdir('/content/GRIP')
r = subprocess.run(['git', 'pull', '--ff-only',
    'https://github.com/joshua5301/GRIP.git', 'main'], capture_output=True, text=True)
if r.returncode:
    raise RuntimeError(r.stdout + r.stderr)
```

```python
%run teacher_fisher_full.py --output /content/drive/MyDrive/GRIP_cora_teacher_fisher_full
```
