# Original Repo

https://github.com/XYGaoG/CGC/tree/main

# Q=I MPNN surrogate (evaluated with 2-layer GCN)

See [the evaluation protocol](docs/mpnn_evaluation.md) and
[the conditional MPNN bound design](docs/mpnn_identity_design.md).
`colab_distance.py` is a ready-to-paste Colab cell with Drive checkpointing and
resume support. `sweep_distance.py --preset pilot` compares the new method,
the raw-feature ablation and GRIP using the same 2-layer GCN and unchanged
uniform student loss. The default objective is `((I+P_closed)/2)^2 D + mu KL`,
independent of the evaluation GCN's normalization matrix. The older GCN-specific
`distance` objective remains available only as an explicit legacy option.


### Robust teacher-label compression (experimental)

See [the objective, assumptions, and Colab preset](docs/robust_labels.md) for
exact L1-ball worst-label costs. Distance-based radii are not certified; the
student remains the original two-layer GCN with uniform CE.
