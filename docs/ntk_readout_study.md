# Readout decomposition

`run_readout_study` consumes the preceding empirical NTK local result. It keeps
all architectures, widths and output dimensions in that result, using its
largest ensemble and projection count and both independent sketch seeds.
The frozen distance networks are reconstructed with the SAME seeds and their
logits checked against the previous cache. No student is retrained. Original
student hashes, graph and probe IDs are checked. Tests run in Colab only.

Five distances are compared on the same networks:

- `logits`: the old random final-output feature kernel, averaged over channels.
- `hidden`: the inputs to the final linear readout, with no random projection.
- `readout`: exact trace NTK from the final linear weights and bias only.
- `internal`: sketched trace NTK from every other trainable parameter.
- `full`: exact readout Gram plus sketched internal Gram.

GCN hidden features are S ReLU(GCN1(X)), width h; the final GCN weight matrix
commutes with graph propagation, while its output bias remains outside S.
GIN features are the inputs to the last linear layer of the second GIN MLP,
width h. SAGE has two separate trainable final linear maps, for neighbor mean
and root features. Its faithful readout feature concatenates both inputs,
width 2h; reducing this to h would change its parameter geometry.

Under the native parameter metric, the channel-averaged readout NTK equals
Z Z^T plus a constant bias kernel. Its distance therefore equals the hidden
feature distance, up to numerical roundoff. Hidden/readout are a consistency
check rather than two distinct hypotheses. No independent normalization of
the readout and internal components is applied before summing. Final distance
normalization and CE calibration follow the previous evaluation unchanged.

The full NTK decomposes exactly by disjoint parameter groups. Internal sketches
set readout directions to zero while using the same Rademacher RNG stream as
the preceding study. Summing group Grams avoids finite-sketch cross terms;
consequently the new `full` need not equal the previous finite projected full
NTK exactly, but estimates the same finite-network trace kernel. The readout
component is now exact. Maximum-ensemble matrices for the five families are
cached per network. The preceding 15 non-deep comparators are retained.

All three saved student architectures remain evaluated. Tables include all
pair quantiles and k-NN thresholds from the previous run. Plots show matching
student/feature architectures; CSV tables retain cross-architecture results.
Primary comparison is `full` versus `hidden`, not versus random logits.

Colab tests explicitly differentiate all parameters of tiny GCN/SAGE/GIN
networks, check the exact readout-feature identity and full=readout+internal,
and verify that excluding all parameters produces a zero sketch.
