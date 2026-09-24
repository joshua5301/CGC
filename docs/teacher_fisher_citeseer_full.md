# Citeseer 3.6% teacher Fisher tuning

Run teacher_fisher_citeseer_full.py after the existing Colab dependencies and
Drive setup. Uses Citeseer with its standard repository split and 120 condensed
nodes. Cora scripts and helpers remain unchanged so existing Cora source caches
remain verifiable.

Same proposed-method-only grid as Cora: gamma .001,.01,.1,1; label T
.1,.2,.5,1,2,5,10; mu 0,.1,.2,.5,1,2,5,10; alpha .05,.1,.3,.5,.8,.95.
Fisher tau=1 and student dropout=.9 are fixed. 1344 settings times three
selection seeds (66--68) gives 4032 fits before any invalid-budget exclusions.
Select by validation alone, separately for graphless and original-graph GCN
evaluation, then confirm the one or two selected settings on seeds 69--78.
Teacher and metric are fit once per gamma and reused across label temperatures.

Citeseer has more than 3000 nodes. The ReLU kernel teacher uses 3000 landmarks
chosen without replacement with CPU random seed zero, matching the repository's
teacher landmark count. Save those indices and hold landmarks and bandwidth
fixed when differentiating query inputs. Fit on train labels only. The
Nyström classifier and its folded kernel form must agree within 1e-9 in
probability at temperature one before using its Fisher. No dense input Fisher
matrix is constructed; only diagonal score averages are retained.

Student: two layers, hidden width 256, identity edges during training, uniform
soft CE, 1000 epochs, existing Adam schedule and independent validation-best
checkpoint per evaluation domain. No old Cora or Citeseer diagnostic artifacts
are needed. Existing data are reused or fetched by the repository loader.

Results and resumable checkpoints follow teacher_fisher_full.py. Settings losing
condensed nodes are excluded with reasons recorded. No baseline is trained or
claimed beaten. Intervals across student seeds do not capture graph/split or
condensation uncertainty. Local smoke tests and training are skipped at the
user's request; only static review and diff checks were performed.

```python
import os, subprocess
os.chdir('/content/GRIP')
r = subprocess.run(['git', 'pull', '--ff-only',
    'https://github.com/joshua5301/GRIP.git', 'main'], capture_output=True, text=True)
if r.returncode:
    raise RuntimeError(r.stdout + r.stderr)
```

```python
%run teacher_fisher_citeseer_full.py --output /content/drive/MyDrive/GRIP_citeseer_teacher_fisher_full
```
