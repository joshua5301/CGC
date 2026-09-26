# Model-output preservation by graph distances

`src.gnn_distance_probe.run_gnn_distance_probe` asks whether nearby nodes have
similar model outputs, without using their labels. It reuses exact tree caches.

## Training protocol

The default trains two-layer GCN, mean GraphSAGE and GIN on five uniform random
subsets of 70 original training nodes, with two model initialization seeds each:
30 trained models. Every architecture receives the same subsets. Sampling is
without replacement and is not class balanced. All models use original features,
the full original graph, uniform CE, and a fixed final epoch. No validation/test
labels or validation-selected checkpoints are used. This is a small-supervision
diagnostic, not training on a small induced or condensed graph.

A fixed random pool of 512 nodes outside the entire original training pool is
used for evaluation. All distances and models use the same nodes and pairs.
Initial outputs are also evaluated, once per model initialization and budget;
they are not duplicated across supervision-subset seeds. Predictions, selected
training IDs and class counts are cached per run. Protocol hashes include the
repository revision, versions, data hashes and settings. Changing these creates
a new output directory. Standard deviations describe crossed subset/model-seed
runs; they are not confidence intervals from independent observations.

## Measurements

Each distance is divided by its median over the common unordered node pairs.
If that median is zero, use the positive-pair median; all-zero distances use one.
Multiplying all distances by a positive constant cannot improve any metric.

Output gaps are measured both by Euclidean distance between class-centered
logits and by total variation between softmax predictions. Centering removes
the common logit offset, which cannot change predictions. Each output gap is
also normalized by its positive-pair median within that model and phase.

- `relative_gap`: mean output gap among each node's k nearest neighbors divided
  by the mean across all pairs. Below one means closer outputs than the random
  pair reference. Lower is better; this is invariant to output rescaling.
- `normalized_gap_p95`: 95th percentile of normalized neighbor output gaps.
  This checks whether the mean hides neighbors with very different outputs.
- `ratio_p99`: 99th percentile of normalized output gap / normalized distance
  over positive-distance pairs. This is an empirical tail diagnostic, not a
  certified Lipschitz constant. Zero-distance pairs with different outputs are
  counted separately and make `ratio_max` infinite.
- `health`: raw output dispersion and a degeneracy flag. Constant-output models
  get NaN normalized scores, not perfect preservation scores. Train accuracy
  and CE help identify failed training without evaluating held-out labels.

Neighbor ranks supply a complementary comparison unaffected by distance units.
Global normalization does not eliminate all geometric distortions. Interpret
local gaps together with tails and health, rather than selecting by one number.

## Comparators and limits

Comparators include raw and row-L2 feature distances, GRIP S²X distance, raw
exact sum tree distances at depths 1/2/3, and raw/L2 depth-one mean distances.
Raw and L2 tree caches must share weight, self-loop policy and dtype. The L2
metric does not change the inputs used to train the students.

Depth counts neighborhood expansions. Depth one omits some information used
by two-layer models; depths two and three should be examined separately.
GCN degree normalization, GraphSAGE mean aggregation and GIN sum aggregation
have different dependencies on graph structure. Success across these probes
is empirical evidence, not a proof that every architecture satisfies a TMD
bound or that condensed-graph training will preserve its outputs. Prediction
similarity also does not by itself establish loss preservation when targets
differ, or preservation of training gradients.

The tests in `tests/test_gnn_distance_probe.py` cover distance-scale invariance,
common-logit-offset invariance, zero-distance violations, collapsed outputs,
and the three model interfaces. Run them in Colab before the experiment.

## Saved-output diagnosis

`src.gnn_distance_diagnostics.analyze_saved_probe` reads existing prediction
files and verified distance caches, without training or accessing labels.
It measures centered-logit direction (one minus cosine similarity), magnitude
differences, and probability TV, each relative to the all-pair mean. Direction
is undefined for zero centered logits; those pairs are excluded and their
coverage is reported. Positive per-node scaling cannot change direction.

For centered logits z_i = r_i u_i, the exact decomposition is

    ||z_i - z_j||² = (r_i - r_j)² + 2 r_i r_j (1 - <u_i, u_j>).

The two contributions are divided by the same all-pair mean squared logit
gap. Their sum is the neighbor mean squared gap relative to the all-pair
mean squared gap; it is not the original unsquared relative-gap metric.
The angular contribution still depends on magnitudes. Use the separate
direction metric to remove that dependence.

Initial/trained changes are paired on architecture and model initialization.
Initial predictions are shared across subset seeds, so delta standard
deviations are descriptive, not independent-sample confidence intervals.
Negative delta means that neighborhoods became more selective for similar
outputs relative to all pairs; it need not mean absolute gaps decreased.

The report also includes per-run norms, confidence, entropy, train CE,
degree correlations, and all pairs below a normalized distance tolerance
(default 1e-8), distinguishing exact zeros. Raw output gaps, probability
gaps and prediction disagreement allow inspection of numerical near-zeros.
A nonzero gap by itself is not a certified counterexample: cache precision,
graph conventions and applicable model assumptions still matter.

Degree correlations and direction/magnitude decomposition are diagnostic,
not causal evidence. No held-out labels or model selection are introduced.

## Full-node sampling with GRIP teacher targets

`src.probe_teacher.tune_probe_teacher` fits the existing GRIP kernel logistic
teacher on S²X using only original training labels. Kernel features are shared
across the gamma grid. Select gamma by validation accuracy only; ties choose
the smallest gamma. Kernel, basis and seed are fixed. No test labels are read
and no training+validation refit is performed. The selected model supplies
soft labels at fixed T=1 for every node, including original training nodes;
there is no ground-truth override.

Pass these probabilities as `pseudo_labels` and the returned configuration as
`teacher_config` to `run_gnn_distance_probe`. Students sample without replacement
from ALL graph nodes, with the same selections across architectures. Loss is
uniform soft-target CE. Reported train accuracy now means agreement with the
teacher argmax, not ground-truth accuracy. The target array enters the cache
hash, and teacher selection provenance is saved in the protocol.

The fixed probe pool remains outside the original training pool, matching the
previous experiment. It may overlap student supervision drawn from all nodes;
`probe_training_overlap` reports that count. This is output preservation on
the graph, not a held-out student-generalization score. Unlike the original
hard-label protocol, validation labels ARE used for teacher gamma selection.
Saved-output diagnostics work unchanged and do not retrain any model.
