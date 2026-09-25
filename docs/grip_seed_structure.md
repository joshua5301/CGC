# GRIP seed 1234 structure analysis

`analyze_grip_seed_structure` reads cached partitions and the exact teacher tensor
from `measure_grip_seed_costs`. It does not rerun initialization, train students,
or read validation/test ground truth for its statistics. If multiple source cases
exist for a dataset/ratio, pass `case_dirs` explicitly rather than silently mixing
hyperparameters or code revisions. Reconstructed objectives are checked against
saved values with 1e-5 tolerance because representative features were saved float32.

Outputs compare seed 1234 with all other converged full-budget saved seeds:

- `runs`: size Gini, effective cluster count, mass/uniform TV, small-cell frequency,
  largest 10% of cells' original mass and cost, teacher purity and label entropy,
  soft class-mass shift induced by uniform representative weighting, nearest
  representative distance and teacher-class geometric margin.
- `reference_comparison`: each reference metric versus other seeds' mean, standard
  deviation, z score and empirical midrank percentile. These are descriptive, not
  p-values or independent significance tests across the many metrics.
- `clusters`: original-node-weighted cost contributions sum to J. Per-node cost is
  shown separately. Cluster IDs must not be matched across seeds.
- `classes`: teacher argmax groups' node/representative counts and node-attributed
  costs, plus soft teacher mass versus uniform representative-label mass. These
  are predicted classes, not ground-truth class preservation measurements.
- `size_groups`: cells divided by their size relative to N/m; original-node mass
  and cost attribution reveal whether excess cost is concentrated in large cells.

Representative distances use the saved global GRIP feature scale. The teacher
margin is (nearest different-class distance - nearest same-class distance) divided
by their sum, excluding self and representatives lacking either neighbor type.
Its valid count is reported. This is a geometric proxy, not a trained GCN margin.
Small cells have at most half the average cell size. Effective cluster count is
1/sum(mass^2); mass_uniform_tv compares n_k/N with 1/m. label_mass_tv compares mean
teacher probabilities over original nodes with mean probabilities over representatives.

Unusual allocation, cost concentration or separation can support candidate
mechanisms but does not demonstrate why accuracy improves. A later controlled
intervention is needed for causality. No new accuracy measurements are made here.
