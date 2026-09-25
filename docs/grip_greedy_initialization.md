# GRIP-cost greedy initialization

`partition(..., init='greedy')` selects node representatives using the same
normalized Euclidean distance plus forward teacher KL cost as GRIP. The first
representative minimizes total cost. Each subsequent representative minimizes
the total nearest-representative cost with previously selected representatives
fixed. Every unselected node is evaluated; there is no candidate sampling.
Ties select the lowest node index. Selected nodes anchor their initial cells.

Costs use float64. Pairwise costs are cached for at most 16 million entries;
larger datasets recompute candidate blocks and can be expensive. The block size
controls memory, not the candidate set. Zero normalization scales are floored.
Subsequent geometric-median and mean-label updates are unchanged, including
GRIP's removal of empty cells. Report both requested and actual node counts.

For finite fixed costs d_ij, let M_i=max_j d_ij. The benefit
F(S)=mean_i[M_i-min_{j in S} d_ij], with F(empty)=0, is normalized, monotone,
and submodular. Exact greedy therefore obtains at least (1-1/e) of the optimal
benefit among node subsets of the same size. This is a benefit guarantee, not a
multiplicative approximation of GRIP loss, a guarantee for unrestricted centers,
or a prediction-accuracy guarantee. The finite-iteration median implementation
does not by itself certify exact block minimization.

Reference: Nemhauser, Wolsey and Fisher (1978),
https://doi.org/10.1007/BF01588971.

`run_experiments(..., method='grip', grip_init='greedy',
grip_init_block_size=256)` exposes this initialization. The default remains
`kmeans`. The protocol fingerprint includes initialization settings. Hyperparameter
selection uses validation only; optional final test evaluation uses the checkpoint
selected by validation for each final seed.

Focused checks in `tests/test_grip_greedy.py` cover a hand-computed greedy
trajectory, label-only separation, candidate block boundaries, and zero-cost
ties. Run them in Colab with `python -m pytest tests/test_grip_greedy.py`.
Local verification is limited to syntax and diff checks; no local model runs.
