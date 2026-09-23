# Propagated features + structure residual, identity condensed graph

`hybrid_grip.py` implements the proposed combination. Original GRIP and student
training are unchanged. The older `StructureBound` used raw-X medians and row-L2
structure residuals; this experiment uses GRIP H2 medians and row-L1 residuals.
Do not compare the numerical structure coefficients of these two objectives.

## Exact implemented objective

Z = the existing GRIP symmetric-normalized adjacency squared times X.
P = the existing `build_transition`: binarized directed incoming-neighbor mean,
self edges removed, isolated nodes use self transitions. This P is deliberately
identified separately from the symmetric operator used to compute Z.
S is a nonempty hard partition; Q=I; F is the full training-label teacher.

    J = mean_v ||Z_v - C[a(v)]|| / feature_scale
      + lambda * ||PS-S||_(1,1) / (2N)
      + mu * mean_v KL(F_v || Y[a(v)]) / kl_scale.

Feature/KL scales are the fixed global GRIP scales, floored at 1e-12. The structure
term is exactly the fraction of transition mass crossing cells, in [0,1]. No
initial-residual normalization or changing scales are used. Teacher probabilities
are floored at 1e-12 then normalized. Labels are per-cell teacher means.

For a node move a->b, the cut delta is the mass connected to cell a minus that
connected to cell b, counting BOTH outgoing and incoming transitions, excluding
self transitions. This gives an exact sequential descent step. The full objective
is checked after each assignment sweep and center update. No singleton can move.
Medians use the existing monotone Vardi-Zhang/Weiszfeld implementation.

All lambda settings independently start from the SAME saved original GRIP graph,
including its centers, labels and partition. `hybrid_0` runs the same refinement
without structure: it is a solver control, NOT a claim of bitwise identity with
original GRIP. `grip` is the original unrefined baseline. All retain its actual
number of nonempty cells. No structural N-by-N or N-by-m dense matrix is formed;
the feature assignment cost is N-by-m. Structural moves cost O(degree+m), and
feature costs/median updates remain a separate computational expense.

## Theoretical scope and missing certification

For a row-stochastic mean-aggregation MPNN

    h_next(v) = phi(h(v), sum_u P[v,u] psi(h(v),h(u))),

assume uniform bounds over the chosen model class:

    ||phi(x,m)-phi(x',m')|| <= a ||x-x'|| + b ||m-m'||;
    ||psi(x,y)-psi(x',y')|| <= c ||x-x'|| + d ||y-y'||;
    ||psi(hc_j,hc_k)|| <= M.

Using the same parameters on the original and condensed graphs gives

    e_next <= (a+bc)e + bd P e + b M r,
    r_v = ||(PS)[v,:] - Q[a(v),:]||_1.

This follows by replacing original endpoints with their cell representatives and
then subtracting the two cell-mass sums. It includes nonlinear messages; it does
NOT cover every aggregator or unconstrained attention model. Sum aggregation
requires degree-dependent treatment, and standard symmetric-normalized GCN uses
a different propagation operator. Hard S commutes with nodewise nonlinearities.

Let cP=max column sum(P). Summing rows yields

    E_next <= t E + b M R,  t=a+bc+bd*cP, R=sum_v r_v.
    E_L <= A E_0 + B R,
    A=product_l t_l,
    B=sum_l b_l M_l product_(s>l) t_s.

For ANY fixed propagated descriptor Z, including the existing GRIP H2,

    E_0 <= sum_v ||X_v-Z_v|| + sum_v ||Z_v-C[a(v)]||.

The first term is an explicit smoothing/descriptor remainder independent of the
condensation. The runner records its mean and maximum, not a certified final
risk constant. The remainder can be large; PS-S alone cannot recover information
already erased by Z. Uniform M_l needs a justified hidden-state/input bound over
the model class, not constants fitted retrospectively to favorable runs.

For logits and teacher F, CE is sqrt(2)-Lipschitz. Since Y is each cell's mean F,

    R_F(original logits) <= sqrt(2)(A E_0 + B R)/N
        + mean_v H(F_v) + mean_v KL(F_v||Y[a(v)])
        + sum_j (n_j/N) KL(Y_j||softmax(condensed_logits_j)).

This compares SAME-parameter networks and is not a guarantee that the trained
student's test risk improves. With the normalized implemented objective, literal
coefficients from this bound would require

    lambda = 2 B / (A * feature_scale),
    mu = kl_scale / (sqrt(2) * A * feature_scale).

The experiment fixes mu to the established GRIP value and validates lambda;
these are surrogate coefficients, not certified bound constants. The GCN protocol
does not constrain the required norms. Its symmetric operator is also different
from structural P. We do not certify a risk bound for this evaluated GCN.

Uniform condensed CE remains unchanged. The last display has cell-mass weights;
those cannot be silently replaced by uniform weights. If losses are bounded by
Lmax, the difference is at most Lmax/2 * ||n/N - 1/m||_1. Such a loss bound is not
enforced here. True-label risk additionally requires a teacher error term.

## Experiment and Colab

Cora ratio .052; gamma=.01, T=2, mu=.5, dropout=.9; GCN-2-256, Adam lr=.01,
weight decay=5e-4, 1000 epochs, evaluation every 10 epochs, uniform soft CE.
The existing strict-best-validation checkpoint rule is preserved.

Lambda grid: 0,.001,.003,.01,.03,.1,.3,1 plus original GRIP. Select using mean
validation on student seeds 0-2 (ties prefer GRIP then smaller lambda). Freeze
that choice before running fresh seeds 3-12 for GRIP, hybrid_0 and the selected
variant. If GRIP or hybrid_0 wins, only these two controls are confirmed. At most
57 student fits. No teacher/dropout/KL sweep is silently added. Confirmation
intervals are pointwise paired Student-t intervals conditional on this data and
fixed condensation, not variation across splits or condensation seeds.

Caches fingerprint source, data, settings and saved condensation bytes. Per-fit
JSON files support resume; manifest, selection.json, summary.json and
student_runs.csv retain diagnostics. Detailed output goes to the log; the cell
shows one updating progress line, selection validation values and confirmation.

After the user's Drive/data setup:

```python
import os, subprocess
subprocess.run(["git", "-C", "/content/GRIP", "pull", "--ff-only"],
               check=True, stdout=subprocess.DEVNULL)
os.chdir("/content/GRIP")
%run hybrid_grip.py
```

Do not interpret a local short-epoch smoke run as a performance experiment.

## Full tuning with fixed gamma/dropout

`hybrid_full.py` keeps Cora .052, gamma=.01, dropout=.9, identity condensed
edges, 2-layer GCN and uniform soft CE fixed. Its default grid is:

- T: .1,.2,.5,1,2,5
- mu: 0,.1,.2,.5,1,2,5
- lambda: 0,.0001,.001,.003,.01,.03,.1,.3,1,3,10

Original GRIP is also tuned over exactly the same T/mu grid. Each T/mu pair has
one GRIP plus 11 hybrid settings, giving 504 settings and 1512 selection fits on
seeds 0-2. Lambda=0 is the refinement control; mu=0 is the feature/structure-only
assignment ablation (soft labels still come from the teacher).

Only one teacher is fitted: its logits are cached and temperature-scaled for
each T. Each T/mu GRIP initialization is shared by that pair's hybrid settings.
No sequential warm starts across lambda are used. Source/data/config/artifact
hashes protect resumed runs. This uses a separate output directory from the
earlier fixed-T/mu experiment and does not import that runner's cache.

Select the best GRIP, best lambda=0 hybrid and best positive-lambda hybrid using
mean validation only. Ties choose smaller lambda, then T, then mu. Commit those
choices to selection.json before fresh-seed evaluation. Independently confirm
these settings on seeds 3-12, plus lambda=0 at the positive winner's exact T/mu
to isolate the structure effect. Duplicate settings are evaluated only once.
This costs at most 40 additional fits, total at most 1552. A positive winner is
confirmed even if its selection validation is worse than a control: it is labeled
the best *positive* configuration, not the overall winner. Do not reselect using
fresh test scores. CIs remain pointwise and conditional on fixed data/condensation.

Only selection winners and confirmation tables are printed. All configurations,
including structural diagnostics, are in summary.csv; individual fits are in
student_runs.csv and runs/*.json. These are saved incrementally. Full logs go to
full_*.log. `--confirmation-seeds 0` runs only the full selection sweep.

```python
import os, subprocess
subprocess.run(["git", "-C", "/content/GRIP", "pull", "--ff-only"],
               check=True, stdout=subprocess.DEVNULL)
os.chdir("/content/GRIP")
%run hybrid_full.py
```
