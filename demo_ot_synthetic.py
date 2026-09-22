"""Synthetic demonstration of the 1-hop neighbourhood-OT objective (src/partition_ot.py).
Five variants, all scored with the SAME structural objective J (same alpha/beta/mu, F, H, P):
  A  root + KL descent (GRIP-like), isolated condensed graph (P_cond = I)
  B  A's partition / features / labels + mean cell-transition P_cond
  C  A's partition / features fixed, adjacency from the cell LPs
  D  A's partition fixed, adjacency and features alternated
  E  full alternating descent (assignment, labels, adjacency, features)
python demo_ot_synthetic.py [--N 60 --m 6 --alpha 1 --beta 1 --mu 1 --iters 5]"""
import argparse, time
import numpy as np
from src.partition_ot import NeighborhoodOT

ap = argparse.ArgumentParser()
ap.add_argument('--N', type=int, default=60); ap.add_argument('--d', type=int, default=4); ap.add_argument('--C', type=int, default=3)
ap.add_argument('--m', type=int, default=6); ap.add_argument('--alpha', type=float, default=1.0); ap.add_argument('--beta', type=float, default=1.0)
ap.add_argument('--mu', type=float, default=1.0); ap.add_argument('--iters', type=int, default=5); ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

# ---- synthetic attributed graph with supplied teacher probabilities
y = rng.integers(0, args.C, args.N)
H = rng.normal(size=(args.N, args.d)) + 2.0 * np.eye(args.C, args.d)[y]
prob = np.where(y[:, None] == y[None, :], 0.3, 0.03)
A = np.triu(rng.random((args.N, args.N)) < prob, 1); A = A | A.T
src, dst = np.nonzero(A)
from src.partition_ot import build_transition
P, meta = build_transition(np.vstack([src, dst]), args.N)
logits = 2.5 * np.eye(args.C)[y] + rng.normal(size=(args.N, args.C))
Fp = np.exp(logits); Fp /= Fp.sum(1, keepdims=True)
# k-means-style initial assignment (non-empty)
centers = H[rng.choice(args.N, args.m, replace=False)]
for _ in range(10):
    assign = np.argmin(((H[:, None] - centers[None]) ** 2).sum(-1), 1)
    for j in range(args.m):
        if (assign == j).any():
            centers[j] = H[assign == j].mean(0)
for j in range(args.m):
    if not (assign == j).any():
        assign[rng.integers(0, args.N)] = j
print(f'graph: N {args.N} d {args.d} C {args.C} m {args.m} nnz(P) {P.nnz} {meta}')
quiet = lambda *_: None
kw = dict(alpha=args.alpha, beta=args.beta, mu=args.mu, log=quiet)
rows = []


def report(name, obj, t):
    rec = obj.history[-1]
    dens = float((obj.Pc > 0).mean())
    rows.append((name, rec['root'], rec['w1'], rec['kl'], rec['J'], obj.counts.min(), obj.counts.max(), dens, t))


# A: root + KL descent with the isolated condensed graph
t0 = time.time()
A_obj = NeighborhoodOT(H, P, Fp, args.m, assign, alpha=args.alpha, beta=0.0, mu=args.mu, log=quiet)
A_obj.run(outer_iters=args.iters, final_polish=False)
A_obj.beta = args.beta                       # score with the full objective
A_obj.Pc = np.eye(args.m); A_obj.gammas = None
A_obj.evaluate('A: isolated')
report('A isolated', A_obj, time.time() - t0)

# B: same partition / features / labels, mean cell-transition adjacency
t0 = time.time()
A_obj.Pc = A_obj._mean_cell_transition(); A_obj.gammas = None
A_obj.evaluate('B: mean P')
report('B mean-transition', A_obj, time.time() - t0)

# C: fixed partition and features, LP adjacency
t0 = time.time()
C_obj = NeighborhoodOT(H, P, Fp, args.m, A_obj.assign, **kw)
C_obj.Z, C_obj.Y = A_obj.Z.copy(), A_obj.Y.copy(); C_obj.cost_all = None
C_obj.adjacency_block(); C_obj.evaluate('C: LP adjacency', w1_from='gammas')
report('C LP adjacency', C_obj, time.time() - t0)

# D: fixed partition, adjacency and features alternated
t0 = time.time()
D_obj = NeighborhoodOT(H, P, Fp, args.m, A_obj.assign, **kw)
D_obj.Z, D_obj.Y = A_obj.Z.copy(), A_obj.Y.copy(); D_obj.cost_all = None
prev = np.inf
for it in range(args.iters):
    D_obj.adjacency_block(); D_obj.evaluate('D: adjacency', w1_from='gammas')
    D_obj.feature_block(); cur = D_obj.evaluate('D: features')['J']
    if prev - cur < 1e-6 * abs(prev):
        break
    prev = cur
D_obj.adjacency_block(); D_obj.evaluate('D: polish', w1_from='gammas')
report('D LP adj + features', D_obj, time.time() - t0)

# E: full alternating descent from the same initial assignment
t0 = time.time()
E_obj = NeighborhoodOT(H, P, Fp, args.m, assign, **kw)
E_obj.run(outer_iters=args.iters)
report('E full alternating', E_obj, time.time() - t0)

print(f'\nobjective J = alpha {args.alpha:g} * root + beta {args.beta:g} * W1 + mu {args.mu:g} * KL  (per-node means)')
print(f'{"variant":<22s} {"root":>9s} {"W1":>9s} {"KL":>9s} {"J":>9s}  cells(min,max)  density  time')
for r in rows:
    print(f'{r[0]:<22s} {r[1]:9.5f} {r[2]:9.5f} {r[3]:9.5f} {r[4]:9.5f}   ({r[5]:3d},{r[6]:3d})     {r[7]:5.2f}  {r[8]:5.2f}s')
print('\nE: stage-wise objective history')
for h in E_obj.history:
    print(f"  {h['stage']:<11s} J {h['J']:.6f}  root {h['root']:.6f}  w1 {h['w1']:.6f}  kl {h['kl']:.6f}"
          + (f"  moved {h['moved']} pair_ot {h['pair_ot']}" if 'moved' in h else '')
          + (f"  lp {h['lp_time']}s nvar_max {h['nvar_max']} resid {h['residual_max']:.1e}" if 'lp_time' in h else ''))
J = [h['J'] for h in E_obj.history]
print('monotone:', all(J[i + 1] <= J[i] + 1e-8 for i in range(len(J) - 1)))
