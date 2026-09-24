import torch
from src.partition import partition, geometric_medians, cell_means

EPS = 1e-12


def fit_sgc(X, labels, ridge=1e-3, steps=1000):
    X, labels = X.detach().double(), labels.detach().double()
    W = torch.zeros(X.shape[1], labels.shape[1], dtype=X.dtype, device=X.device, requires_grad=True)
    b = torch.zeros(labels.shape[1], dtype=X.dtype, device=X.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([W, b], max_iter=steps, tolerance_grad=1e-8,
                                 tolerance_change=1e-12, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = -(labels * (X @ W + b).log_softmax(1)).sum(1).mean() + ridge * W.square().sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    closure()
    gradient = max(float(W.grad.abs().max()), float(b.grad.abs().max()))
    if not torch.isfinite(W).all() or not torch.isfinite(b).all() or not gradient <= 1e-5:
        raise RuntimeError(f'SGC solver did not converge: gradient={gradient:g}; increase sgc_steps')
    return W.detach(), b.detach()


@torch.no_grad()
def sgc_risk(X, labels, W, b):
    return float(-(labels * (X @ W + b).log_softmax(1)).sum(1).mean())


@torch.no_grad()
def refine_candidate(X, labels, centers, assignment, W, beta, mu, scales, rounds=10, steps=50):
    if not 0 <= beta <= 1:
        raise ValueError('refine_beta must lie in [0, 1]')
    centers, assignment = centers.clone(), assignment.clone()
    W = W - W.mean(1, keepdim=True)
    logits = X @ W
    sx, sy = scales
    sz = (logits - logits.mean(0)).square().sum(1).mean().clamp(min=EPS)
    entropy = (labels * labels.log()).sum(1)
    m = len(centers)
    def feature_cost(C):
        delta = C[assignment] - X
        return (1 - beta) * delta.norm(dim=1).mean() / sx + beta * (delta @ W).square().sum(1).mean() / sz
    def objective(C, Y):
        kl = (entropy - (labels * Y[assignment].clamp(min=EPS).log()).sum(1)).mean()
        return float(feature_cost(C) + mu * kl / sy)
    Y = cell_means(labels, assignment, m)
    previous = objective(centers, Y)
    for _ in range(rounds):
        costs = []
        for x, f, h, ent in zip(X.split(2048), labels.split(2048), logits.split(2048), entropy.split(2048)):
            cost = (1 - beta) * torch.cdist(x, centers) / sx + beta * torch.cdist(h, centers @ W).square() / sz
            costs.append(cost + mu * (ent[:, None] - f @ Y.clamp(min=EPS).log().T) / sy)
        costs = torch.cat(costs).cpu().numpy()
        a = assignment.cpu().numpy().copy()
        counts = torch.bincount(assignment, minlength=m).cpu().numpy()
        for v, j in enumerate(costs.argmin(1)):
            old = a[v]
            if counts[old] > 1 and costs[v, j] < costs[v, old] - 1e-12:
                a[v] = j
                counts[old] -= 1
                counts[j] += 1
        assignment = torch.as_tensor(a, device=X.device)
        Y = cell_means(labels, assignment, m)
        counts = torch.bincount(assignment, minlength=m).to(X.dtype)[:, None]
        for _ in range(steps):
            delta = centers[assignment] - X
            rows = (1 - beta) * delta / delta.norm(dim=1, keepdim=True).clamp(min=EPS) / sx
            rows += 2 * beta * (delta @ W) @ W.T / sz
            gradient = torch.zeros_like(centers).index_add_(0, assignment, rows) / len(X)
            direction = gradient * len(X) / counts
            decrease = (gradient * direction).sum()
            if decrease <= EPS:
                break
            before, rate = feature_cost(centers), 1.
            for _ in range(40):
                proposal = centers - rate * direction
                after = feature_cost(proposal)
                if after <= before - 1e-4 * rate * decrease:
                    centers = proposal
                    break
                rate *= .5
            else:
                break
        current = objective(centers, Y)
        if current > previous + 1e-8 * max(1., abs(previous)):
            raise RuntimeError('Fixed-SGC objective increased')
        if previous - current <= 1e-7 * max(1., abs(previous)):
            break
        previous = current
    return centers, Y, assignment


def refine_sgc(X, labels, cluster_num, mu=.5, beta=.1, rounds=5,
               ridge=1e-3, sgc_steps=1000, grip_steps=1000, tolerance=1e-6):
    if not 0 <= beta <= 1 or mu < 0 or ridge <= 0 or tolerance < 0 or min(rounds, sgc_steps, grip_steps) < 1:
        raise ValueError('invalid SGC refinement parameters')
    X, labels = X.detach().double(), labels.detach().double().clamp(min=EPS)
    labels = labels / labels.sum(1, keepdim=True)
    C, Y, a, converged = partition(X, labels, cluster_num, mu, grip_steps, return_state=True)
    if not converged:
        raise RuntimeError('GRIP assignment did not converge; increase grip_steps')
    W, b = fit_sgc(C, Y, ridge, sgc_steps)
    risk = sgc_risk(X, labels, W, b)
    history = [dict(step=0, risk=risk, accepted=True)]
    print(f'SGC initial risk={risk:.8f}')
    reference = geometric_medians(X, torch.zeros(len(X), dtype=torch.long, device=X.device), 1)
    sx = (X - reference).norm(dim=1).mean().clamp(min=EPS)
    sy = (labels * (labels.log() - labels.mean(0).log())).sum(1).mean().clamp(min=EPS)
    for step in range(1, rounds + 1):
        candidate, targets, assignment = refine_candidate(X, labels, C, a, W, beta, mu, (sx, sy))
        fitted_W, fitted_b = fit_sgc(candidate, targets, ridge, sgc_steps)
        candidate_risk = sgc_risk(X, labels, fitted_W, fitted_b)
        accepted = candidate_risk < risk - tolerance
        history.append(dict(step=step, risk=candidate_risk, accepted=accepted))
        print(f'SGC step {step}: risk={candidate_risk:.8f} {"accepted" if accepted else "rejected; stop"}')
        if not accepted:
            break
        C, Y, a, W, b, risk = candidate, targets, assignment, fitted_W, fitted_b, candidate_risk
    return dict(x=C, y=Y, assignment=a, W=W, b=b, risk=risk, history=history)
