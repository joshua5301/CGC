# Budgeted Cora boundary search

40 sequential Optuna TPE trials replace the expanded Cartesian grid. Fixed
Cora .052, Fisher tau=1, dropout=.9, uniform soft CE, two layers and the existing
1000-epoch protocol. Proposals use mean GCN-transfer validation on seeds 79--81.
Graphless results are diagnostic only; neither test scores nor graphless
validation guide TPE or final selection. Confirm the single selected setting
on seeds 82--91 in both domains. At most 120 selection fits plus 10 confirmation
fits, fewer if invalid budgets or duplicate settings occur. No epoch pruning.

Search space:

* gamma categorical: 1e-5,3e-5,1e-4,3e-4,.001 (at most five teacher/Fisher fits)
* label T categorical: .5,1,2
* mu continuous log-uniform: 3 to 100
* alpha continuous log-uniform: .002 to .1

Enqueue the old gamma=.001,T=1,mu=10,alpha=.05 winner as the first trial, evaluated
on the same new selection seeds as every other candidate. This remains proposed
method only: alpha zero is not in this positive log search. The smaller budget
does not guarantee discovery of the optimum. Future runs should start with a
bounded trial budget rather than automatically expanding a large grid.

Use fixed sampler seed 20260924, 10 startup trials, single-worker execution.
Every trial saves its parameters and objective as JSON. On resume, recreate
the seeded ask/tell sequence from trial zero and use cached fits to recover the
same objectives, verifying proposals against saved trials. No live SQLite
database is needed on Drive. Source, Optuna/runtime and experiment settings are
validated before replay. An interrupted trial resumes from finished seed fits.
Budget-loss trials are marked failed, count toward the 40-trial budget and are
reported. Other errors stop execution. --trials can increase but cannot shrink
an existing study. Changes to settings require a new output directory.

summary.csv contains both domains; selection.json identifies the GCN-selected
setting, also reported in the graphless row. confirmation.json describes the
fixed-data/condensation limitation. Old source scripts and artifacts are left
unchanged. No local smoke tests or training were run, at the user's request.

Optuna references:
https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
https://optuna.readthedocs.io/en/stable/faq.html

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
%run teacher_fisher_optuna.py --trials 40 --output /content/drive/MyDrive/GRIP_cora_teacher_fisher_optuna
```
