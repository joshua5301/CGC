# Comparing GRIP initialization and refinement

`visualize_grip` captures the exact FAISS K-means assignment before GRIP iterations
and its final assignment for two partition seeds. The optional initial snapshot
does not change partition optimization. Initial J uses GRIP geometric medians and
mean teacher labels on the K-means partition, not the FAISS squared-error objective.

One PCA+t-SNE projection of all original H=A^2 X features is fitted per dataset,
shared by every seed, stage, and budget. t-SNE coordinates do not enter clustering.
The function targets <=15000 nodes and trains no GCN. Teacher fitting/clustering
and projection are intended to run in Colab only. Ground-truth validation/test
labels are not used to select views or color them.

Cluster colors are aligned by maximum-overlap Hungarian matching: both initial
partitions to the first seed's initial partition, then each final partition to its
own aligned initial partition. Color matching is bookkeeping, not evidence that
clusters are equivalent. ARI and NMI are computed on full high-dimensional
clustering assignments, not on the projection. Reassignment percent minimizes
label permutation via maximum overlap. Unmatched clusters receive fresh IDs.

Representative plots show means of member-node t-SNE coordinates with marker
area 25+8*sqrt(cluster size). These are schematic locations, not t-SNE embeddings
of high-dimensional geometric medians. Marker colors use argmax of the original
teacher soft-label mean. Allocation bars count those teacher-derived classes and
do not assert ground-truth purity. Initial/final actual feature representatives
and soft labels are saved in partition artifacts alongside assignments.

PNG/PDF plots, a cluster table, projection coordinates, exact parameters, teacher
outputs and full partition artifacts are saved. Dense overlapping t-SNE colors
can conceal changes; interpret them together with ARI/NMI and reassignment rates.
Projection neighborhoods and global shapes are not proofs of separability.
