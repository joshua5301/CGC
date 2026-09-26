# Labels matched to frozen convex coefficients

run_mixture_label_study loads the hidden-mode validation winner from a
raw-convex full sweep. It freezes condensed inputs, teacher, partition, T,
student hyperparameters and convex coefficients. The two target choices are
the saved cluster mean and sum_i alpha_ji softmax(teacher_logits_i/T).
Teacher probabilities are computed on the original graph, not by querying
the teacher on synthetic inputs. Labels are not jointly optimized.

Graph, teacher weights, original mean labels and reconstruction from the saved
coefficients are checked. Both arms train the same two-layer GCN with uniform
CE and paired student seeds; validation chooses the student checkpoint and
test is reporting only. Alpha weights mix nodes inside a representative label;
they do not weight the student CE across representatives. Cluster-size-weighted
CE is a separate experiment and is not enabled here.

Outputs include both label matrices, mean L1 label change, argmax-change count,
accuracy summary, per-seed paired differences and a figure. This is a controlled
label change at the previous winner, not a new hyperparameter sweep for the
mixture-label method. Tests run in Colab, not locally.
