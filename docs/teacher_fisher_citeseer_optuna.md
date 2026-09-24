# Citeseer bounded Optuna search

40 sequential TPE trials for Citeseer .036 (120 condensed nodes), fixed tau=1
and dropout=.9. Uses the Citeseer Nyström teacher helper with 3000 seed-zero
landmarks and train-label-only fitting. Fisher query derivatives hold teacher
and landmarks fixed. No Cora optimum or full-sweep result is required.

Because no Citeseer tuned winner was supplied, retain the earlier broad ranges:
gamma categorical .001,.01,.1,1; label T categorical .1,.2,.5,1,2,5,10;
mu categorical 0,.1,.2,.5,1,2,5,10; alpha continuous log .05--.95.
First trial is gamma=.01,T=.2,mu=.1,alpha=.05, an initial candidate rather than
an asserted optimum. At most four teacher/Fisher computations are needed.

Same protocol as the Cora Optuna runner: selection seeds 79--81, mean GCN
validation objective, 10 startup trials, fixed TPE seed, no epoch pruning.
Graphless scores do not guide selection. Confirm the GCN-selected setting on
seeds 82--91 in both domains. Up to 120 selection fits plus 10 confirmation
fits. Uniform soft CE, two-layer width-256 student, 1000 epochs. Positive alpha
only; this is a proposed-method search, not a baseline comparison.

Replay/resume uses deterministic ask/tell and saved trial JSON, validated against
source/runtime/configuration hashes. Per-seed fits are cached. Invalid budget
trials count toward the limit and are recorded. Rerun with a larger --trials to
extend, using the same software/environment and output directory. Forty trials
are a limited exploration, not exhaustive coverage or a global optimum guarantee.
Results are conditional on the fixed data split and condensation initialization.
No local training or smoke tests were run, following the user's instruction.

```python
%pip -q install optuna==4.5.0
import os, subprocess
os.chdir('/content/GRIP')
r = subprocess.run(['git', 'pull', '--ff-only',
    'https://github.com/joshua5301/GRIP.git', 'main'], capture_output=True, text=True)
if r.returncode:
    raise RuntimeError(r.stdout + r.stderr)
```

```python
%run teacher_fisher_citeseer_optuna.py --trials 40 --output /content/drive/MyDrive/GRIP_citeseer_teacher_fisher_optuna
```
