import math
import time

import torch

from src.risk_partition import seed_partition


def _transport(cost, epsilon, steps, tolerance):
    n, m = cost.shape
    cost = cost - cost.mean(1, keepdim=True)
    cost.sub_(cost.mean(0, keepdim=True))
    scale = cost.square().mean().sqrt().clamp_min(1e-15)
    log_kernel = cost.div_(scale).div_(-epsilon)
    u, v = cost.new_zeros(n), cost.new_zeros(m)
    for iteration in range(steps):
        u = -math.log(n) - torch.logsumexp(log_kernel + v[None, :], dim=1)
        v = -math.log(m) - torch.logsumexp(log_kernel + u[:, None], dim=0)
        if (iteration + 1) % 10 == 0 or iteration + 1 == steps:
            row_mass = (u + torch.logsumexp(log_kernel + v[None, :], dim=1)).exp()
            if float((row_mass * n - 1).abs().max()) < tolerance:
                break
    plan = (log_kernel + u[:, None] + v[None, :]).exp()
    before = max(float((plan.sum(1) * n - 1).abs().max()),
                 float((plan.sum(0) * m - 1).abs().max()))
    plan.mul_(((1 / n) / plan.sum(1).clamp_min(1e-300)).clamp_max(1)[:, None])
    plan.mul_(((1 / m) / plan.sum(0).clamp_min(1e-300)).clamp_max(1)[None, :])
    row_gap = (1 / n - plan.sum(1)).clamp_min(0)
    col_gap = (1 / m - plan.sum(0)).clamp_min(0)
    missing = float(row_gap.sum())
    if missing > 1e-16:
        plan.add_(row_gap[:, None] * (col_gap / col_gap.sum())[None, :])
    residual = max(float((plan.sum(1) * n - 1).abs().max()),
                   float((plan.sum(0) * m - 1).abs().max()))
    if not bool(torch.isfinite(plan).all()) or residual > 1e-7:
        raise FloatingPointError('Transport marginal rounding failed')
    return plan, dict(sinkhorn_steps=iteration + 1, sinkhorn_residual=before,
                      rounding_mass=missing, marginal_residual=residual)


@torch.no_grad()
def uniform_transport(H, Q, m, B, seed=0, max_sweeps=30, block_size=1024,
                      atol=1e-12, rtol=1e-10, epsilon=0.05, sinkhorn_steps=200,
                      sinkhorn_tolerance=1e-6, oracle_retries=4, line_steps=20):
    if not 1 <= m <= len(H) or not math.isfinite(B) or B <= 0:
        raise ValueError('Require 1 <= m <= N and finite B > 0')
    if min(epsilon, sinkhorn_tolerance) <= 0 or min(sinkhorn_steps, oracle_retries, line_steps, block_size) < 1:
        raise ValueError('Invalid transport settings')
    if max_sweeps < 0 or min(atol, rtol) < 0:
        raise ValueError('Invalid objective tolerances')
    if H.is_cuda:
        torch.cuda.synchronize(H.device)
    started = time.perf_counter()
    X, Q = H.detach().double(), Q.detach().double()
    offset = X.mean(0)
    X = X - offset
    scale = X.square().sum(1).mean().sqrt().clamp_min(1e-15)
    X = X / scale
    n, d = X.shape
    alpha, beta = B * B / 4, 2 * B
    energy, moment = X.square().sum(1).mean(), X.T @ Q / n
    generator = torch.Generator(device=X.device).manual_seed(seed)
    assignment = seed_partition(X, Q, m, B, generator, block_size)
    counts = torch.bincount(assignment, minlength=m).double()
    c = X.new_zeros(m, d).index_add_(0, assignment, X) / counts[:, None]
    y = Q.new_zeros(m, Q.shape[1]).index_add_(0, assignment, Q) / counts[:, None]
    vq = (Q - Q.mean(0)).square().sum(1).mean()
    eta = (vq / energy.clamp_min(1e-15)).sqrt()
    wx = alpha + B * eta
    wq = B / eta if float(eta) > 0 else 0.0
    cost = -2 * wx * (X @ c.T) - 2 * wq * (Q @ y.T)
    plan, diagnostic = _transport(cost, epsilon, sinkhorn_steps, sinkhorn_tolerance)
    del cost, assignment, counts
    oracle_history = [diagnostic]

    def score(s, r):
        variance = (energy - m * s.square().sum()).clamp_min(0)
        error = moment - m * s.T @ r
        objective = float(alpha * variance + beta * error.norm())
        if not math.isfinite(objective):
            raise FloatingPointError('Nonfinite transport objective')
        return variance, error, objective

    s, r = plan.T @ X, plan.T @ Q
    variance, error, objective = score(s, r)
    history, step_history = [objective], []
    status = 'iteration_limit'
    for sweep in range(max_sweeps):
        c, y = m * s, m * r
        direction = error / error.norm().clamp_min(1e-15)
        cost = -2 * alpha * (X @ c.T) - beta * ((X @ direction) @ y.T + (Q @ direction.T) @ c.T)
        accepted = False
        for retry in range(oracle_retries):
            candidate, diagnostic = _transport(cost, epsilon * 0.25 ** retry,
                                               sinkhorn_steps, sinkhorn_tolerance)
            diagnostic.update(sweep=sweep + 1, epsilon=epsilon * 0.25 ** retry)
            oracle_history.append(diagnostic)
            ds, dr = candidate.T @ X - s, candidate.T @ Q - r
            tolerance = atol + rtol * max(1.0, abs(objective))
            for backtrack in range(line_steps):
                step = 0.5 ** backtrack
                next_variance, next_error, value = score(s + step * ds, r + step * dr)
                if value < objective - tolerance:
                    plan.lerp_(candidate, step)
                    s, r = plan.T @ X, plan.T @ Q
                    variance, error, value = score(s, r)
                    if value > objective + tolerance:
                        raise FloatingPointError('Transport objective increased')
                    objective = value
                    history.append(value)
                    step_history.append(step)
                    accepted = True
                    break
            del candidate
            if accepted:
                break
        del cost
        if not accepted:
            status = 'oracle_stalled'
            break
    row_residual = float((plan.sum(1) * n - 1).abs().max())
    column_residual = float((plan.sum(0) * m - 1).abs().max())
    if max(row_residual, column_residual) > 1e-7:
        raise FloatingPointError('Final transport marginals violated')
    result = dict(x=(m * s * scale + offset).float().cpu(), y=(m * r).float().cpu(),
                  counts=(plan.sum(0) * n).cpu(), J=objective, V=float(variance),
                  moment_error=float(error.norm()), history=history, steps=step_history,
                  sweeps=len(step_history), converged=False, status=status, B=float(B), seed=int(seed),
                  row_residual=row_residual, column_residual=column_residual,
                  oracle_history=oracle_history, mass_tv=float((plan.sum(0) - 1 / m).abs().sum() / 2))
    result['seconds'] = time.perf_counter() - started
    return result
