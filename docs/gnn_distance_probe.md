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
