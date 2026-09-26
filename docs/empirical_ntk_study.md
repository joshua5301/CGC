# Paired empirical NTK depth study

The question is whether poor distance preservation comes from the Jacobian
sketch, initialization sampling, or the geometry of the initialization NTK.
No student is retrained. The previous local CE protocol and original GCN,
GraphSAGE and GIN student outputs are reused, with hashes verified.

## Paired construction

For each architecture, hidden width and network seed, instantiate the same
two-layer `ProbeGNN` used by prior experiments, with dropout disabled and C=16
output channels. One frozen network supplies BOTH output features and Jacobians:

    K_RF(i,j)  = mean_m <f_m(i), f_m(j)> / C
    K_NTK(i,j) = mean_m sum_c <grad f_mc(i), grad f_mc(j)> / C.

The NTK is the scalar trace of the vector-output NTK, divided by channel count;
it is not the full matrix-valued NTK training operator. All native trainable
parameters participate, including biases and GIN epsilon. No group reweighting
or parameter normalization is introduced. Default PyG initialization is retained.
Output features are the logits/channels, not softmax probabilities.

This differs from the earlier unpaired comparison: it used output width equal
to hidden width for RF and scalar output for NTK, with different network seeds.
Old comparators remain in the tables, but causal comparisons should use the
paired `deep_rf_*` / `deep_ntk_*` methods at equal width and ensemble count.
Output width is fixed across hidden-width experiments. Width changes under
native PyG parameterization are NOT a convergence test toward the analytic
standard-parameterized bias-free GCN kernel.

## Defaults and what each axis measures

- Hidden widths: 64 and 256.
- Network seeds: 5000 through 5007; nested ensembles of 1, 4 and 8 networks.
- Jacobian projections: 64, 256 and 512 independent Rademacher directions.
- Two independent sketch seeds: 6000 and 7000. For network seed s the actual
  direction seed is sketch_seed + 100003*s. Network seeds are separate from
  evaluated student seeds.
- Fixed output width: 16. All output channels are differentiated, not projected
  to one random scalar. Directions are shared across channels, but channels
  occupy separate coordinates in the sketch, so no cross-channel terms enter.

Compute JVPs by automatic differentiation. Concatenate the N-by-C directional
derivatives over R directions and divide by sqrt(R*C). The resulting Gram is an
unbiased estimate of the channel-averaged trace NTK. Prefixes share directions
within a replica. Gram calculations and distance subtraction use float64;
network operations and derivatives use float32, with TF32 disabled.

The larger sketch is not declared ground truth. Projection stability compares
each sketch to the largest sketch from the OTHER independent replica of the
same frozen networks, on evaluation nodes only. An additional exact audit uses
16 fixed evaluation nodes: explicitly form parameter gradients channel by
channel and accumulate their exact Gram, without random projection. This is
performed for the first network seed at every architecture and width. It tests
sketch fidelity for that finite network, not all seeds or the infinite-width limit.

`exact_audit.csv` reports relative distance error (without scalar recalibration),
distance Spearman correlation and nearest-5 overlap. `stability.csv` reports
nearest-10 overlap and the same distance diagnostics for independent sketches
and nested network ensembles. `ensemble_agreement.csv` additionally compares
two disjoint halves of the initialization seeds, if the maximum ensemble size
is even. NTK halves average both largest sketches to reduce projection noise;
residual sketch error still contributes. Nested-ensemble comparison includes
shared networks and is not an independent replication. The maximum ensemble
compared against itself gives the trivial identity, not evidence of convergence.

## CE evaluation and reporting

Every candidate is evaluated against ALL saved student architectures, not just
the matching architecture. Fixed disjoint calibration/evaluation node split,
quantiles, k values, per-student scale calibration and CE normalization are
inherited. No best setting is selected using these scores and no test accuracy
is consulted. Crossed student seeds are not independent statistical replicates.

The default grid has 18 paired RF distances and 108 NTK distances (including
two sketch replicas), plus the 15 preceding comparators. All tables are saved.
`detailed_summary.csv` joins architecture/width/ensemble/projection metadata to
the CE table. CE plots show matching architectures, mean and p95 at fixed k.
Bands are the range of two sketch replicas after averaging student runs, NOT
confidence intervals. The exact-audit plot separates approximation accuracy
from CE preservation. Student-run variation remains in the saved summary.
`paired_rf_runs.csv` and `paired_rf_summary.csv` compare NTK minus RF within the
same frozen student, width and initialization ensemble; negative deltas favor
NTK. Each sketch replica is kept separate in the summary.

Interpret improvements in neighbor stability separately from improvements in
CE preservation. A stable NTK with worse CE preservation is evidence against
that initialization geometry under the current setup, not against all NTKs.
No condensed graph is trained by this experiment.

## Compute, caching and verification

With defaults, 48 frozen networks each require 2*512 JVPs, plus six small-node
exact audits. This is substantially heavier than prior distance comparisons;
it is intended for Colab A100. Local numerical tests are not run. Each network's
outputs, each requested sketch prefix Gram and exact audit are atomically
cached. Completed networks are reused on rerun. An incomplete replica restarts
its projection sequence to preserve prefixes. Changing the study configuration
uses a separate cache. Input data and numerical-library versions are hashed.

Tests verify vector-output trace normalization, explicit projection directions,
nested sketch prefixes, RNG preservation, distance/stability identities, unchanged
baseline CE scores, full cache reuse and plotting. Execute them in Colab.
