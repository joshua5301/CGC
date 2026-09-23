# Distance propagation with Q = I, two-layer GCN

The student is the existing **two-layer GCN**, hidden width 256, ReLU, bias,
dropout, and **uniform mean soft-label cross entropy** in `model_training`.
No cell-size weighting, architecture change, norm constraint, or extra training
loss is introduced. Condensation uses raw input features. The teacher still
uses the existing kernel model on A^2 X with training labels only.

## Objective and the actual GCN operator

Use the existing symmetrically normalized adjacency A (including self loops),
not a row-normalized random walk. GCN's A generally has row sums different
from one. With representatives C, root assignment a, and fixed teacher F:

```
d[v,j] = ||X[v] - C[j]||_2
delta  = |A 1 - 1|
q      = A delta + delta
cost   = A (A d) + q[:,None] ||C||[None,:]
J      = mean_v { cost[v,a[v]] + mu KL(F[v] || Yc[a[v]]) }
Yc[j]  = mean_{v:a[v]=j} F[v]
```

Only sparse matrix products are used; A^2 is never materialized. There is no
rho parameter: the two-layer GCN fixes the propagation rule and depth.
No per-term normalization is applied. The extra q term is essential, even
for identical constant features, because A does not preserve constants.

For the evaluation-mode student

```
H1  = ReLU(A X W1 + b1)       H1c = ReLU(C W1 + b1)
Z   = A H1 W2 + b2           Zc  = H1c W2 + b2
L1  = ||W1||_2               L2  = ||W2||_2
```

triangle inequalities and ReLU's Lipschitz property give, for every v,j:

```
||Z[v]-Zc[j]|| <= L1 L2 cost[v,j] + L2 ||b1|| delta[v].
```

Proof: layer one is bounded by L1 (A d + delta ||C||).
At layer two add L2 delta ||H1c||, and use
||H1c[j]|| <= L1 ||C[j]|| + ||b1||. The output bias cancels.
This handles nonlinear intermediate representations; it is not SGC feature
matching. Dropout is disabled during evaluation; this is not a bound on
individual stochastic training passes.

With Yc equal to cell means, CE's sqrt(2) Lipschitz constant gives:

```
R_F(g) <= mean H(F)
        + sqrt(2) L1 L2 mean_v cost[v,a[v]]
        + mean_v KL(F[v] || Yc[a[v]])
        + sqrt(2) L2 ||b1|| mean delta
        + sum_j (n_j/N) KL(Yc[j] || softmax(Zc[j])).
```

Here R_F is teacher-target risk, not true-label or population risk. Optimizing
J with freely chosen mu is a bound-derived surrogate. Common finite weight
and bias bounds across trainings would be needed to turn its coefficients
into fixed certified constants; the default unconstrained GCN does not impose
these. The bias term is independent of condensation only after such a common
bound is fixed. No claim that decreasing J guarantees increased test accuracy.

**Uniform student training is retained for baseline fairness.** The weighted
residual in the theorem is not replaced by uniform loss. If epsilon_uniform
is the mean of per-representative KL losses, the weighted residual is at most
`(m max_j n_j / N) epsilon_uniform`, so imbalance can weaken the connection.
We log cell sizes and retain assignments but do not change the training loss.

## Optimization

Raw-X deterministic k-means, repaired to exactly m nonempty cells, initializes
the assignment. Centers start at cell means. Repeat:

1. Score every destination in representative batches. Protect one current
   member per cell (lowest current cost), and move other nodes only when their
   fixed-center cost strictly decreases. This is a nonempty descent step,
   not an assertion of an exact constrained assignment optimum.
2. Set Yc to cell teacher means (the forward-KL minimizer).
3. Update C using weighted geometric medians. For S one-hot membership, the
   data weights are `(A.T)^2 S`, with an additional atom at the origin of
   weight `q.T S` for each representative. Use Vardi-Zhang coincident-point
   handling and retain the best iterate, including the starting point.

Recompute the full objective after assignment and center/label updates and
reject numerical objective increases. Stop on relative objective improvement
or the iteration cap. Keep float64 condensation, float32 student training.
Working distance/weight matrices are N by batch_size, not N by all m.
Feature storage is dense, as in the existing pipeline. This batching does not
make the full teacher/GCN pipeline guaranteed to fit arbitrary large graphs.

Default batch size 32, median iterations 30, outer iterations 20, tolerance
1e-6. These are numerical controls, not part of the accuracy sweep. A single
standalone run is also available:

```bash
python main.py --dataset_name cora --ratio 0.052 --edges distance_identity \
  --gamma 0.01 --T 1 --label_weight 1 --outer_iters 20 --gpu 0
```

## Colab protocol

After cloning the repo and preparing data, execute `colab_distance.py` (or paste
its contents into a notebook cell). The default starts the **pilot** on
Cora 5.2% and Citeseer 3.6%. Set PRESET to `full` for the joint sweep.

| Setting | Pilot | Full |
|---|---|---|
| gamma | Cora .01, Citeseer .1 | .01, .1, 1 |
| T | Cora 1, Citeseer .2 | .2, .5, 1, 2 |
| distance mu | .1, .3, 1, 3, 10 | same |
| GRIP kl_weight | .1, .2, .5, 1, 2 | same |
| dropout | .1, .5, .9 | same |
| student seeds | 0, 1, 2 | same |

Teacher kernel is fixed at the repository's dataset setting (Cora ReLU,
Citeseer erf). Student settings are identical across methods: 1000 epochs,
evaluation every 10, Adam lr .01, weight decay 5e-4, existing halfway learning
rate change, existing best-validation checkpoint metric. GRIP's original
partition and normalized-GCN route are retained. The new method passes the
precomputed GCN operator without a second normalization. Both use identity
edges on the condensed graph and the existing uniform loss.

The pilot has 20 condensations / 180 student fits across both datasets;
full has 240 condensations / 2160 student fits. These are substantial sweeps,
not a short smoke check. No A100 runtime estimate is claimed.

Reinitialize every student with the same seed for corresponding runs of both
methods, independently of condensation RNG consumption. Report mean test and
sample standard deviation at the configuration/dropout selected by **mean
validation** across the three student seeds. These are student-training seeds;
condensation uses one fixed seed. Vary `--seed` explicitly for a separate
condensation-seed robustness experiment. Test accuracy is logged but never
used for selection. This remains ordinary validation tuning, not nested CV.

Teacher logits are cached per gamma; changing T needs no teacher retraining.
Condensed graphs and each student run are saved atomically. Rerunning the same
command resumes completed runs. Source/data/config fingerprints prevent stale
results from being reused; changing code or epoch/solver settings triggers new
entries. Cache artifacts are loaded only from the user's own output directory.

Outputs on Drive: `teachers/*.pt`, `condensed/*.pt` (features, labels,
assignments, per-stage objective, timing), `runs/*.json`, `manifest.json`,
`summary.csv`, `best.json`, and timestamped console logs. Summary files describe
the current invocation and include only configurations with all requested
student repeats completed. Existing individual results are retained.

Examples of additional controls:

```bash
# One setting, integration smoke test (not a meaningful accuracy comparison)
python sweep_distance.py --cases cora:0.052 --mus 1 --baseline-kl .5 \
  --dropouts .5 --repeat 1 --epoch 20 --eval-every 10 \
  --outer-iters 2 --median-iters 5 --raw-data-dir ./data

# All three citation-graph densities; expand only after the first comparison
python sweep_distance.py --preset full \
  --cases cora:0.013,cora:0.026,cora:0.052,citeseer:0.009,citeseer:0.018,citeseer:0.036
```

Tests: `python -m pytest tests/test_partition_distance.py -q`. These cover
the actual nonlinear biased PyG GCN inequality and CE risk decomposition,
the row-mass counterexample, independent dense/adjoint objective checks,
monotone updates, nonempty cells, coincident weighted medians, representative
batching, and CPU/CUDA agreement when CUDA is available.
