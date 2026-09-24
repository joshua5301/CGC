# Explicit ReLU-kernel medoid experiment

Fixed Cora 5.2% comparison: original GRIP, Euclidean medoids, and ReLU-kernel
medoids. Reuses the completed `GRIP_cora_fisher_diagnostic` source artifact and
frozen models. Frozen models measure prediction changes only; they do not choose
the kernel, assignments or representatives. Source integrity and runtime checks
are inherited from `fisher_grip.load_source`.

Inputs are Z=P^2 X. Both medoid variants start from the exact same GRIP cells and
the same within-cell source nodes nearest the GRIP representatives in Z space.
Their objective is the average normalized feature distance plus mu times average
normalized forward teacher KL. Feature distances are unsquared Euclidean norms
in Z or the explicit ReLU feature map. Each geometry is scaled once by its mean
distance to its global feature mean; teacher KL is scaled by mean KL to global
mean teacher labels. Coefficient mu is inherited from the source experiment.

Assignments minimize fixed-center costs, except that each medoid remains in its
own cell. Labels become exact cell means. Medoid updates minimize the within-cell
sum of distances over actual members. These constrained updates preserve every
cell and monotonically decrease this clustering objective (checked at runtime).
This is a local alternating algorithm, not globally optimal k-medoids.

The map uses the existing arc-cosine ReLU kernel and regularized Nyström/Cholesky
features. Default 3000 landmarks means all 2708 Cora nodes are landmarks: a
regularized full-landmark representation, not low-rank compression. Smaller
`--basis-num` samples landmarks with seed 0. Pairwise distances require O(N^2)
memory, suitable for this Cora diagnostic, not a large-graph algorithm.

Condensed features are original Z rows at medoid indices, with cell-mean teacher
labels. No inverse mapping or mean-in-kernel-space preimage is needed. Node-feature
norm information is retained. Kernel objectives have different fixed scales, so
their numerical objective values are not directly comparable across variants.

Three variants, seeds 36--45: 30 fits. Training uses a 2-layer GCN with identity
edges, hidden width 256, dropout .9, uniform soft CE, 1000 epochs and the existing
optimizer schedule. Evaluation separately chooses the first best validation
checkpoint for graphless Z inputs and raw-graph GCN transfer. The latter is an
empirical transfer test, not covered by a fixed-kernel linear-head risk bound.
The source provides gamma=.001, T=10, mu=1; this is a fixed-configuration test,
not a hyperparameter sweep. Primary contrast: ReLU vs Euclidean medoids.

Output contains paired student-seed intervals, and Bonferroni intervals for four
planned ReLU contrasts in summary.json. Intervals condition on the fixed split,
teacher and condensation. There is no guarantee that the retrained GCN belongs
to this fixed-feature linear-head class or that clustering descent lowers risk.
Completed fits are cached. No local smoke test or training was run for this
implementation, at the user's request.

Colab, after the earlier environment and Drive setup:

```python
import os, subprocess
os.chdir('/content/GRIP')
r = subprocess.run(['git', 'pull', '--ff-only',
    'https://github.com/joshua5301/GRIP.git', 'main'], capture_output=True, text=True)
if r.returncode:
    raise RuntimeError(r.stdout + r.stderr)
```

```python
%run relu_medoid.py --diagnostic-dir /content/drive/MyDrive/GRIP_cora_fisher_diagnostic --output /content/drive/MyDrive/GRIP_cora_relu_medoid
```
