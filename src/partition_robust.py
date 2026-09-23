"""GRIP with exact L1-ball worst-label costs; radius validity is an assumption.

The distance radius below is explicitly a heuristic, NOT a coverage certificate.
The label block uses finite projected subgradient steps, retaining its best
iterate. Neither an exact outer optimum nor a general-MPNN risk bound is claimed.
"""
import math
import numpy as np
import torch
import faiss
from src.partition import partition, geometric_medians, EPS


def worst_labels(prob, log_label, radius):
    """Exact argmax -p.log_label over simplex intersect L1(prob, radius).

    Inputs are validated by the caller. Leading dimensions may broadcast.
    Move <= radius/2 mass from cheapest classes to a most expensive class.
    """
    prob, log_label = torch.broadcast_tensors(prob, log_label)
    sink = log_label.argmin(-1, keepdim=True)
    order = log_label.argsort(dim=-1, descending=True, stable=True)
    donor = prob.gather(-1, order) * (order != sink)
    mass = torch.minimum(radius / 2, 1 - prob.gather(-1, sink).squeeze(-1)).clamp(min=0)
    before = donor.cumsum(-1) - donor
    removed = torch.minimum(donor, (mass.unsqueeze(-1) - before).clamp(min=0))
    out = prob - torch.zeros_like(prob).scatter(-1, order, removed)
    return out.scatter_add(-1, sink, removed.sum(-1, keepdim=True))


def distance_radii(features, train_mask, cap=.2, floor=0., mode='distance'):
    """Training-mask-only distances in H2; no validation/test labels accessed."""
    if not 0 <= floor <= cap <= 2 or mode not in ('distance', 'constant'):
        raise ValueError('Require 0 <= floor <= cap <= 2 and distance/constant mode')
    x = features.detach().cpu().numpy().astype('float32')
    mask = train_mask.detach().cpu().numpy().astype(bool)
    if mask.shape != (len(x),) or mask.sum() < 2 or not np.isfinite(x).all():
        raise ValueError('Need finite features and at least two training anchors')
    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x[mask])
    loo = np.sqrt(np.maximum(index.search(x[mask], 2)[0][:, 1], 0))
    positive = loo[loo > 0]
    scale = float(np.median(positive)) if len(positive) else 1.
    radii = []
    for block in np.array_split(x, max(1, math.ceil(len(x) / 4096))):
        r = np.sqrt(np.maximum(index.search(block, 1)[0][:, 0], 0))
        radii.append(r)
    distances = np.concatenate(radii)
    distances[mask] = 0.
    epsilon = np.full(len(x), cap) if mode == 'constant' else floor + (cap-floor) * distances / (distances+scale)
    return torch.as_tensor(epsilon, dtype=torch.float64, device=features.device), dict(
        radius_mode=mode, radius_cap=cap, radius_floor=floor, distance_scale=scale,
        radius_quantiles=np.quantile(epsilon, [0, .25, .5, .75, 1]).tolist(),
        radius_certificate=False, distance_space='A_hat^2 X', anchors=int(mask.sum()))


def fit_labels(prob, radius, assignment, initial, steps=150, step_size=2., box=32.):
    """Convex logit objective, box-constrained, best finite subgradient iterate.

    Reports a subgradient lower-bound gap for the box problem, not convergence
    to the unrestricted optimum. Exact ties may have nonzero selected gradient.
    """
    m = len(initial)
    counts = torch.bincount(assignment, minlength=m).to(prob.dtype)
    if (counts == 0).any():
        raise ValueError('Label solver requires nonempty cells')

    def evaluate(z):
        logq = z.log_softmax(1)
        p = worst_labels(prob, logq[assignment], radius)
        losses = -(p * logq[assignment]).sum(1)
        value = torch.zeros(m, device=prob.device, dtype=prob.dtype).index_add_(0, assignment, losses) / counts
        avg_p = torch.zeros_like(z).index_add_(0, assignment, p) / counts[:, None]
        return value, logq.exp() - avg_p

    z = initial.log()
    z = (z-(z.max(1, keepdim=True).values+z.min(1, keepdim=True).values)/2).clamp(-box, box)
    best, grad = evaluate(z)
    best_z = z.clone()
    lower = best - (grad*z).sum(1) - box*grad.abs().sum(1)
    uniform = torch.zeros_like(z)
    value, g = evaluate(uniform)
    better = value < best
    best[better], best_z[better] = value[better], uniform[better]
    lower = torch.maximum(lower, value-box*g.abs().sum(1))
    for t in range(1, steps+1):
        z = (z - step_size/math.sqrt(t) * grad).clamp(-box, box)
        value, grad = evaluate(z)
        lower = torch.maximum(lower, value-(grad*z).sum(1)-box*grad.abs().sum(1))
        better = value < best
        best[better], best_z[better] = value[better], z[better]
    return best_z.softmax(1), float((best-lower).clamp_min(0).max())


@torch.no_grad()
def robust_partition(features, teacher, m, radius, coefficient=.5, outer_iters=20,
                     label_steps=150, batch_size=32, log=print):
    x, f = features.double(), teacher.double()
    radius = torch.as_tensor(radius, device=x.device, dtype=x.dtype)
    if (radius.shape != (len(x),) or not torch.isfinite(radius).all()
            or (radius < 0).any() or (radius > 2).any()):
        raise ValueError('Each L1 radius must be finite and in [0, 2]')
    if (not torch.isfinite(x).all() or not torch.isfinite(f).all() or (f < 0).any()
            or not torch.allclose(f.sum(1), torch.ones(len(f), device=x.device, dtype=x.dtype))):
        raise ValueError('Require finite features and teacher simplex probabilities')
    if not math.isfinite(coefficient) or coefficient < 0 or min(outer_iters, label_steps, batch_size) < 1:
        raise ValueError('Invalid solver configuration')
    # Exact baseline dispatch, including its initialization and empty-cell policy.
    c, y, a = partition(features, teacher, m, coefficient)
    if not radius.any():
        return dict(H_cond=c, Y_cond=y, assign=a, history=[], diagnostics=dict(
            radius_certificate=False, risk_certificate=False, zero_radius_baseline=True))
    f = f.clamp_min(EPS)
    f = f / f.sum(1, keepdim=True)
    y = y / y.sum(1, keepdim=True)
    m = len(c)
    global_center = geometric_medians(x, torch.zeros(len(x), dtype=torch.long, device=x.device), 1)
    dist_scale = (x-global_center).norm(dim=1).mean().clamp_min(EPS)
    entropy = (f*f.log()).sum(1)
    kl_scale = (entropy-f @ f.mean(0).log()).mean().clamp_min(EPS)

    def assigned_cost(assign, centers, labels):
        logy = labels.log()[assign]
        worst = worst_labels(f, logy, radius)
        return (x-centers[assign]).norm(dim=1)/dist_scale + coefficient*((-worst*logy).sum(1)+entropy)/kl_scale

    history = []
    def record(stage):
        value = float(assigned_cost(a, c, y).mean())
        if history and value > history[-1]['objective'] + 1e-8*max(1., abs(value)):
            raise RuntimeError('Robust objective increased')
        history.append(dict(stage=stage, objective=value))
        log(f'Robust {stage}: J={value:.8g}')
        return value

    record('initial')
    gap = None
    for iteration in range(outer_iters):
        previous = history[-1]['objective']
        # Label block is evaluated before committing; no exact-minimizer claim.
        candidate, gap = fit_labels(f, radius, a, y, steps=label_steps)
        if assigned_cost(a, c, candidate).sum() <= assigned_cost(a, c, y).sum():
            y = candidate
        record(f'{iteration}:labels')
        current = assigned_cost(a, c, y)
        best, destination = current.clone(), a.clone()
        for j in range(0, m, batch_size):
            labels = y[j:j+batch_size].log()
            for start in range(0, len(x), 512):
                end = min(start+512, len(x))
                p = worst_labels(f[start:end, None, :], labels[None, :, :], radius[start:end, None])
                cost = torch.cdist(x[start:end], c[j:j+batch_size])/dist_scale
                cost += coefficient*((-p*labels[None]).sum(-1)+entropy[start:end, None])/kl_scale
                value, index = cost.min(1)
                better = value < best[start:end]
                best[start:end][better] = value[better]
                destination[start:end][better] = j+index[better]
        # Keep one current member in each cell so fixed labels remain meaningful.
        for j in range(m):
            members = (a == j).nonzero().flatten()
            anchor = members[current[members].argmin()]
            destination[anchor] = j
        a = destination
        record(f'{iteration}:assign')
        candidate = geometric_medians(x, a, m)
        old = torch.zeros(m, device=x.device, dtype=x.dtype).index_add_(0, a, (x-c[a]).norm(dim=1))
        new = torch.zeros_like(old).index_add_(0, a, (x-candidate[a]).norm(dim=1))
        improve = new <= old
        c[improve] = candidate[improve]
        value = record(f'{iteration}:centers')
        if previous-value <= 1e-7*max(1., abs(previous)):
            break
    # Final labels correspond to the returned partition.
    candidate, gap = fit_labels(f, radius, a, y, steps=label_steps)
    if assigned_cost(a, c, candidate).sum() <= assigned_cost(a, c, y).sum():
        y = candidate
    record('final_labels')
    return dict(H_cond=c, Y_cond=y, assign=a, history=history, diagnostics=dict(
        radius_certificate=False, risk_certificate=False, zero_radius_baseline=False,
        label_box_gap_max=gap, label_solver='best-iterate projected subgradient',
        label_steps=label_steps, dist_scale=float(dist_scale), kl_scale=float(kl_scale)))
