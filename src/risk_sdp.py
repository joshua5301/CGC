import time

import numpy as np
from scipy.linalg import eigh


def normalize_features(H):
    X = np.asarray(H, dtype=np.float64)
    if X.ndim != 2 or not len(X) or not np.isfinite(X).all():
        raise ValueError('Require a finite nonempty feature matrix')
    X = X - X.mean(0)
    scale = np.sqrt(np.square(X).sum() / len(X))
    return X / scale if scale > 0 else X


def partition_value(X, Q, assignment, B):
    labels, assignment = np.unique(assignment, return_inverse=True)
    n = np.bincount(assignment)
    s, r = np.zeros((len(labels), X.shape[1])), np.zeros((len(labels), Q.shape[1]))
    np.add.at(s, assignment, X)
    np.add.at(r, assignment, Q)
    V = max(0., (np.square(X).sum() - (s * s / n[:, None]).sum()) / len(X))
    E = (X.T @ Q - (s / n[:, None]).T @ r) / len(X)
    return float(B * B / 4 * V + 2 * B * np.linalg.norm(E))


def dual_lower_bound(X, Q, m, B, U, row_dual, trace_dual, nonnegative_dual,
                     safety=1e-8):
    N = len(X)
    U = np.asarray(U, dtype=np.float64)
    y = np.asarray(row_dual, dtype=np.float64)
    M = np.asarray(nonnegative_dual, dtype=np.float64)
    if safety <= 0 or not all(np.isfinite(a).all() for a in (U, y, M, trace_dual)):
        raise ValueError('Require finite dual candidates and positive safety margin')
    U = U / max(1., np.linalg.norm(U)) / (1 + safety)
    M = np.maximum((M + M.T) / 2, 0)
    alpha, beta = B * B / 4, 2 * B
    cross = X @ U @ Q.T
    C = (alpha * (X @ X.T) + beta * (cross + cross.T) / 2) / N
    D = (y[:, None] + y[None, :]) / 2 + float(trace_dual) * np.eye(N) - C - M
    minimum = float(eigh(D, subset_by_index=[0, 0], eigvals_only=True)[0])
    margin = safety * max(1., np.linalg.norm(D, ord=np.inf), abs(float(trace_dual)))
    shift = max(0., -minimum) + margin
    repaired_trace = float(trace_dual) + shift
    constant = alpha * np.square(X).sum() / N + beta * np.sum(U * (X.T @ Q)) / N
    raw = float(constant - y.sum() - m * repaired_trace)
    return dict(lower_bound=max(0., raw), dual_raw_bound=raw,
                dual_min_eigenvalue=minimum, dual_trace_shift=shift,
                dual_repaired_min_eigenvalue=minimum + shift,
                dual_U=U, dual_y=y, dual_M=M, dual_t=repaired_trace)


def solve_risk_sdp(H, Q, m, B, eps=1e-5, max_iters=10000, max_nodes=512,
                   time_limit_secs=1800., safety=1e-8):
    import cvxpy as cp

    X, Q = normalize_features(H), np.asarray(Q, dtype=np.float64)
    N, d = X.shape
    if Q.ndim != 2 or len(Q) != N or not np.isfinite(Q).all():
        raise ValueError('Require finite labels with one row per feature')
    if not isinstance(m, (int, np.integer)) or not 1 <= m <= N or not np.isfinite(B) or B <= 0:
        raise ValueError('Require integer 1 <= m <= N and finite B > 0')
    if N > max_nodes:
        raise ValueError(f'Dense SDP has {N} nodes; explicitly increase max_nodes={max_nodes} or select a subset')
    if min(eps, max_iters, time_limit_secs, safety) <= 0:
        raise ValueError('Require positive solver limits and tolerances')
    started = time.perf_counter()
    alpha, beta = B * B / 4, 2 * B
    Z = cp.Variable((N, N), symmetric=True)
    Y, R = cp.Variable(Q.shape), cp.Variable((d, Q.shape[1]))
    nonnegative, rows, trace = Z >= 0, Z @ np.ones(N) == 1, cp.trace(Z) == m
    moment = R == X.T @ (Q - Y) / N
    constraints = [Z >> 0, nonnegative, rows, trace, Y == Z @ Q, moment]
    gram = X @ X.T
    objective = alpha * (np.square(X).sum() - cp.sum(cp.multiply(gram, Z))) / N + beta * cp.norm(R, 'fro')
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver='SCS', eps=eps, max_iters=max_iters,
                  time_limit_secs=time_limit_secs, verbose=False)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or Z.value is None:
        raise RuntimeError(f'SDP failed: {problem.status}')
    z = (Z.value + Z.value.T) / 2
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite SDP solution')
    certificate = dual_lower_bound(X, Q, m, B, -moment.dual_value / beta,
        rows.dual_value, trace.dual_value, nonnegative.dual_value, safety)
    E = X.T @ (Q - z @ Q) / N
    value = alpha * (np.square(X).sum() - np.sum(gram * z)) / N + beta * np.linalg.norm(E)
    result = dict(Z=z, status=problem.status, solver_value=float(problem.value),
        relaxed_value=float(value), row_residual=float(np.max(np.abs(z.sum(1) - 1))),
        trace_residual=float(abs(np.trace(z) - m)), nonnegative_violation=float(max(0., -z.min())),
        psd_violation=float(max(0., -eigh(z, subset_by_index=[0, 0], eigvals_only=True)[0])),
        moment_residual=float(np.max(np.abs(R.value - E))),
        iterations=int(problem.solver_stats.num_iters), seconds=time.perf_counter() - started,
        nodes=N, clusters=m, B=float(B), bound_kind='numerically_repaired_dual')
    result.update(certificate)
    return result


def round_risk_sdp(Z, m, seed=0):
    from sklearn.cluster import kmeans_plusplus

    N = len(Z)
    if m == 1:
        return np.zeros(N, dtype=np.int64)
    if m == N:
        return np.arange(N)
    values, vectors = eigh((Z + Z.T) / 2, subset_by_index=[N - m, N - 1])
    embedding = vectors * np.sqrt(np.maximum(values, 0))
    _, ids = kmeans_plusplus(embedding, m, random_state=seed, n_local_trials=1)
    chosen = []
    for idx in ids:
        if int(idx) not in chosen:
            chosen.append(int(idx))
    if len(chosen) < m:
        chosen.extend(i for i in range(N) if i not in chosen)
        chosen = chosen[:m]
    centers = embedding[chosen]
    costs = np.square(embedding).sum(1)[:, None] + np.square(centers).sum(1) - 2 * embedding @ centers.T
    assignment = costs.argmin(1)
    assignment[chosen] = np.arange(m)
    return assignment
