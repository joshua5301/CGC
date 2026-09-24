# Original Repo

https://github.com/XYGaoG/CGC/tree/main

`--sgc_refine` runs GRIP until assignments stabilize, then refines its cells with
an additional centered-SGC-logit squared distance. Candidates preserve the
initial cell count and are accepted only when a freshly fitted SGC lowers the
original teacher-label CE by more than `--refine_tolerance`. A rejected candidate
is discarded. This mode evaluates the accepted float64 SGC, not the default GCN.

`--refine_beta 0.1 --refine_rounds 5 --sgc_ridge 0.001` controls the added term,
outer iteration cap and uniform-soft-CE SGC regularization. `--sgc_steps 1000`
and `--grip_steps 1000` cap the solvers; failure of the SGC gradient check or
GRIP assignment stabilization stops execution. Geometric medians retain GRIP's
finite iterations; candidate center descent is also approximate. Accepted
measured teacher risk decreases, without a test-risk or global-optimum guarantee.

