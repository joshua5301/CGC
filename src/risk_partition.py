import math
import time

import torch


def seed_partition(X, Q, m, B, generator, block_size):
    N = len(X)
    if m == N:
        return torch.arange(N, device=X.device)
    if m == 1 or float(X.square().sum(1).mean()) == 0:
        return torch.arange(N, device=X.device) % m

    centered_q = Q - Q.mean(0)
    vh = X.square().sum(1).mean()
    vq = centered_q.square().sum(1).mean()
    if float(vq) > 0:
        eta = (vq / vh).sqrt()
        wx, wq = B * B / 4 + B * eta, B / eta
    else:
        wx, wq = X.new_tensor(B * B / 4), X.new_tensor(0.0)

    Z = torch.cat((wx.sqrt() * X, wq.sqrt() * centered_q), dim=1)
    z2 = Z.square().sum(1)
    nearest = torch.full((N,), torch.inf, dtype=X.dtype, device=X.device)
    selected = torch.zeros(N, dtype=torch.bool, device=X.device)
    seeds = []
    current = int(torch.randint(N, (1,), generator=generator, device=X.device))
    for j in range(m):
        seeds.append(current)
        selected[current] = True
        distance = (z2 + z2[current] - 2 * (Z @ Z[current])).clamp_min(0)
        nearest = torch.minimum(nearest, distance)
        nearest[selected] = 0
        if j + 1 < m:
            if float(nearest.sum()) > 0:
                current = int(torch.multinomial(nearest, 1, generator=generator))
            else:
                current = int(torch.nonzero(~selected)[0, 0])

    seeds = torch.tensor(seeds, device=X.device)
    centers = Z[seeds]
    assignment = torch.empty(N, dtype=torch.long, device=X.device)
    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        cost = (z2[start:end, None] + z2[seeds][None, :]
                - 2 * Z[start:end] @ centers.T).clamp_min(0)
        assignment[start:end] = cost.argmin(1)
    assignment[seeds] = torch.arange(m, device=X.device)
    return assignment


@torch.no_grad()
def risk_partition(H, Q, m, B, seed=0, max_sweeps=30, block_size=1024,
                   atol=1e-12, rtol=1e-10):
    if not 1 <= m <= len(H) or not math.isfinite(B) or B <= 0:
        raise ValueError('Require 1 <= m <= N and finite B > 0')
    if max_sweeps < 0 or block_size < 1 or min(atol, rtol) < 0:
        raise ValueError('Invalid partition solver settings')

    started = time.perf_counter()
    X, Q = H.detach().double(), Q.detach().double()
    offset = X.mean(0)
    X = X - offset
    raw_scale = X.square().sum(1).mean().sqrt()
    scale = raw_scale if float(raw_scale) > 0 else X.new_tensor(1.0)
    X = X / scale

    N, d = X.shape
    K = Q.shape[1]
    alpha, beta = B * B / 4, 2 * B
    generator = torch.Generator(device=X.device).manual_seed(seed)
    assignment = seed_partition(X, Q, m, B, generator, block_size)
    x2, q2 = X.square().sum(1), Q.square().sum(1)
    energy, moment = x2.mean(), X.T @ Q / N

    def aggregate():
        n = torch.bincount(assignment, minlength=m).double()
        s = X.new_zeros(m, d).index_add_(0, assignment, X)
        r = Q.new_zeros(m, K).index_add_(0, assignment, Q)
        return n, s, r

    def score(n, s, r):
        c = s / n[:, None]
        error = moment - c.T @ r / N
        variance = (energy - (s * c).sum() / N).clamp_min(0)
        value = float(alpha * variance + beta * error.norm())
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite partition objective')
        return error, variance, value

    counts, sums, label_sums = aggregate()
    error, variance, objective = score(counts, sums, label_sums)
    history, moves_history = [objective], []

    def apply_moves(ids, destinations):
        nonlocal counts, sums, label_sums, error, variance, objective
        sources = assignment[ids]
        next_counts = (counts + torch.bincount(destinations, minlength=m)
                       - torch.bincount(sources, minlength=m))
        if bool((next_counts > 0).all()):
            xb, qb = X[ids], Q[ids]
            next_sums, next_labels = sums.clone(), label_sums.clone()
            next_sums.index_add_(0, sources, -xb)
            next_sums.index_add_(0, destinations, xb)
            next_labels.index_add_(0, sources, -qb)
            next_labels.index_add_(0, destinations, qb)
            next_error, next_variance, next_objective = score(
                next_counts, next_sums, next_labels)
            tolerance = atol + rtol * max(1.0, abs(objective))
            if next_objective < objective - tolerance:
                counts, sums, label_sums = next_counts, next_sums, next_labels
                error, variance, objective = next_error, next_variance, next_objective
                assignment[ids] = destinations
                return len(ids)
        if len(ids) == 1:
            return 0
        middle = len(ids) // 2
        left = apply_moves(ids[:middle], destinations[:middle])
        right = apply_moves(ids[middle:], destinations[middle:])
        return left + right

    converged = m in (1, N) or float(raw_scale) == 0
    for _ in range(0 if converged else max_sweeps):
        moved = 0
        order = torch.randperm(N, generator=generator, device=X.device)
        for ids in order.split(block_size):
            c, y = sums / counts[:, None], label_sums / counts[:, None]
            xb, qb = X[ids], Q[ids]
            sources = assignment[ids]
            ua, va = xb - c[sources], qb - y[sources]
            ua2, va2 = ua.square().sum(1), va.square().sum(1)
            ka = counts[sources] / (N * (counts[sources] - 1).clamp_min(1))
            kb = counts / (N * (counts + 1))
            ub2 = (x2[ids, None] + c.square().sum(1)[None, :]
                   - 2 * xb @ c.T).clamp_min(0)
            vb2 = (q2[ids, None] + y.square().sum(1)[None, :]
                   - 2 * qb @ y.T).clamp_min(0)
            xe, ce = xb @ error, c @ error
            ta = ((xe - ce[sources]) * va).sum(1)
            tb = ((xe * qb).sum(1)[:, None] - xe @ y.T - qb @ ce.T
                  + (ce * y).sum(1)[None, :])
            cross_x = (ua * xb).sum(1)[:, None] - ua @ c.T
            cross_q = (va * qb).sum(1)[:, None] - va @ y.T
            next_norm2 = (error.square().sum() - 2 * (ka * ta)[:, None]
                          + (ka.square() * ua2 * va2)[:, None]
                          + 2 * kb[None, :] * tb
                          + kb.square()[None, :] * ub2 * vb2
                          - 2 * ka[:, None] * kb[None, :] * cross_x * cross_q)
            delta_v = -ka[:, None] * ua2[:, None] + kb[None, :] * ub2
            delta = alpha * delta_v + beta * (next_norm2.clamp_min(0).sqrt()
                                              - error.norm())
            delta.scatter_(1, sources[:, None], torch.inf)
            delta[counts[sources] <= 1] = torch.inf
            best_delta, destinations = delta.min(1)
            tolerance = atol + rtol * max(1.0, abs(objective))
            take = best_delta < -tolerance
            if bool(take.any()):
                moved += apply_moves(ids[take], destinations[take])

        counts, sums, label_sums = aggregate()
        error, variance, objective = score(counts, sums, label_sums)
        if objective > history[-1] + 1e-8 * max(1.0, abs(history[-1])):
            raise FloatingPointError('Partition objective increased')
        history.append(objective)
        moves_history.append(moved)
        if moved == 0:
            converged = True
            break

    result = dict(x=((sums / counts[:, None]) * scale + offset).float().cpu(),
                  y=(label_sums / counts[:, None]).float().cpu(),
                  counts=counts.long().cpu(), J=objective, V=float(variance),
                  moment_error=float(error.norm()), history=history,
                  moves=moves_history, sweeps=len(moves_history),
                  converged=converged, B=float(B), seed=int(seed))
    result['seconds'] = time.perf_counter() - started
    return result
