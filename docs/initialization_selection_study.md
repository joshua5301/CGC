# Initialization selection study

`run_initialization_study` keeps each density's stored teacher and student
hyperparameters fixed. Optional `configs={(dataset, ratio): {...}}` overrides
individual settings. The evaluation path is the same as the preceding
K-means/K-means++ comparison: two-layer GCN and uniform soft-label CE.
No test accuracy is computed and no new evaluator is introduced.

Defaults: candidate partition seeds 10 through 29 for both initializers,
reference K-means seed 1234, discovery student seeds 0 through 4, confirmation
student seeds 200 through 209. Confirmation seeds are disjoint from discovery,
but are not claimed to be unused in previous experiments. Confirmation is
conditional on the same validation nodes and candidate pool, not an independent
generalization test of a learned selection policy.

Diagnostics record mean within-cluster squared feature distance about arithmetic
means after feature K-means, initial GRIP cost after median/label updates, final
feature and normalized KL terms, weighted KL, total GRIP cost, node count, and
convergence. Instrumentation does not change GRIP updates. Cost comparisons
are within a density and fixed teacher/KL setting only.

Before any student training, seeded nested candidate subsets of sizes 1, 5, 10,
20 are generated. Twenty permutations are used, shared across initializers.
For each subset, the first randomly ordered candidate represents random choice;
the other choices minimize initial SSE or final GRIP cost. Validation is never
consulted. Reference seed 1234 is excluded. Only converged candidates retaining
the requested budget qualify; insufficient eligible candidates stop the study
with costs.csv available for inspection rather than silently changing budgets.

Every candidate receives five discovery runs. Selected candidates and the
reference receive ten confirmation runs; repeated selections reuse the same
runs. At most 123 condensed graphs and 1,845 student fits occur for the default
three-density Citeseer study. This is an upper bound, not a runtime estimate.

summary.csv reports confirmation validation and gain over random selection.
subset_std describes overlapping subset-selection outcomes, not independent
replications or a confidence interval. At size 20 cost-based selections repeat
the same winner and their subset_std can be zero. correlations.csv uses
Spearman correlation within each density and initializer, excluding reference;
negative correlation means lower cost associates with higher discovery valid.

Each candidate artifact, teacher output, and per-student-seed result is saved.
Same-protocol reruns reuse completed work. Protocol changes, including revision,
produce a new case directory. Derived tables are rebuilt. A future independent
partition-seed pool is needed to validate the chosen policy before final test.
