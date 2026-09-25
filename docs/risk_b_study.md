# B-only grid search with new-seed confirmation

`run_b_study` expands each baseline B by factors 0.25, 0.5, 1, 2, 4, 8.
Teacher, dropout, optimization settings, and all other parameters remain fixed.
Both surrogate and split initializations receive the same grid, five partition
seeds, and three GCN seeds per partition. Selection averages GCN runs within
each partition and then partitions equally. Ties use the smaller B factor.

Each method independently selects one B per density using search validation.
The choices are written before confirmation. Only selected settings are then
evaluated using ten new partition seeds and five new GCN seeds; seed sets must
be disjoint from search in both categories. B is never reselected on confirmation.
Teacher tensors are cached once and shared across B, initializers, and stages.
The original two-layer GCN, uniform soft-label CE, and node-order pairing from
the initialization comparison are retained. Test accuracy is never evaluated.

Default three-density study: 540 search student fits and 300 confirmation fits.
The paired comparison subtracts surrogate from split within each new partition
seed after averaging its five GCN runs. The descriptive Student-t 95% interval
uses these ten partition differences, not fifty independent model runs. This
measures seed robustness on the same validation nodes, not unseen-data accuracy;
multiple density comparisons are not corrected for multiple testing.

J is reported for diagnostics but cannot be compared numerically across B.
V_final and E_final are reported separately. grid_boundary flags an endpoint
winner, and converged_fraction shows capped solver runs rather than concealing
them. Flags do not automatically trigger adaptive retuning of confirmation.

All combinations and student seeds are persisted using the existing runner.
Same-protocol reruns resume; changed grid, seeds, settings, or Git revision
create another protocol folder. Selected/final tables and paired differences
are saved there; latest.json points to the last completed report.
