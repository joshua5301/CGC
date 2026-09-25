import time

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from src.partition import EPS, cell_means, geometric_medians, kmeans_init


def quantize_labels(q, observations):
    if int(observations) != observations or observations < 1:
        raise ValueError('Require a positive integer observation count')
    q = q.double()
    q = q / q.sum(1, keepdim=True)
    scaled = q * observations
    counts = scaled.floor().long()
    remaining = observations - counts.sum(1)
    order = (scaled - counts).argsort(1, descending=True, stable=True)
    additions = torch.arange(q.shape[1], device=q.device)[None, :] < remaining[:, None]
    counts.scatter_add_(1, order, additions.long())
    return counts.double() / observations


def fit_contamination(q, calibration_ids, labels, distance, observations,
                      prior_clip=(.05, .95), smoothing=5.):
    if not 0 < prior_clip[0] <= prior_clip[1] < 1 or smoothing <= 0:
        raise ValueError('Require interior prior limits and positive smoothing')
    obs = quantize_labels(q, observations)
    correct = q[calibration_ids].argmax(1).eq(labels[calibration_ids]).cpu().numpy()
    model = IsotonicRegression(increasing=False, out_of_bounds='clip')
    model.fit(distance[calibration_ids], correct.astype(float))
    rho = np.clip(model.predict(distance), *prior_clip)
    wrong = obs[calibration_ids][torch.as_tensor(~correct, device=obs.device)]
    background = (wrong.sum(0) + smoothing / q.shape[1]) / (len(wrong) + smoothing)
    diagnostics = dict(calibration_count=len(correct), calibration_errors=int((~correct).sum()),
        distance=model.X_thresholds_.tolist(), clean_prior=model.y_thresholds_.tolist(),
        background=background.cpu().tolist(), prior_min=float(rho.min()), prior_max=float(rho.max()))
    return obs, torch.as_tensor(rho, dtype=torch.float64, device=q.device), background, diagnostics


def mixture_terms(q, labels, prior, background, observations, pairwise=False):
    entropy = (q * q.clamp_min(EPS).log()).sum(1)
    if pairwise:
        kl = entropy[:, None] - q @ labels.clamp_min(EPS).log().T
    else:
        kl = entropy - (q * labels.clamp_min(EPS).log()).sum(1)
    noise_kl = entropy - q @ background.clamp_min(EPS).log()
    log_clean = prior.log()
    log_noise = torch.log1p(-prior) - observations * noise_kl
    if pairwise:
        log_clean, log_noise = log_clean[:, None], log_noise[:, None]
    log_clean = log_clean - observations * kl
    total = torch.logaddexp(log_clean, log_noise)
    return -total / observations, log_clean - total


def preserve_nonempty(old, proposed, gains, clusters):
    result = old.cpu().numpy().copy()
    target = proposed.cpu().numpy()
    counts = np.bincount(result, minlength=clusters)
    order = np.argsort(-gains.cpu().numpy(), kind='stable')
    for i in order:
        source, destination = result[i], target[i]
        if source != destination and counts[source] > 1:
            counts[source] -= 1
            counts[destination] += 1
            result[i] = destination
    return torch.as_tensor(result, device=old.device)


@torch.no_grad()
def mixture_partition(x, q, clusters, prior, background, observations,
                      kl_weight=.2, seed=0, steps=1000, tolerance=1e-7,
                      block_size=2048, scale_labels=None):
    started = time.perf_counter()
    x, q = x.double(), q.double()
    prior, background = prior.to(x).double(), background.to(x).double()
    if not 1 <= clusters <= len(x) or steps < 1 or tolerance <= 0 or kl_weight < 0:
        raise ValueError('Invalid mixture solver settings')
    if prior.shape != (len(x),) or not bool(((prior > 0) & (prior < 1)).all()):
        raise ValueError('Require one interior prior per node')
    if background.shape != (q.shape[1],) or not bool((background > 0).all()):
        raise ValueError('Require positive background probabilities')
    background = background / background.sum()
    zeros = torch.zeros(len(x), dtype=torch.long, device=x.device)
    global_center = geometric_medians(x, zeros, 1)
    dist_scale = (x - global_center).norm(dim=1).mean().clamp_min(EPS)
    source = q if scale_labels is None else scale_labels.to(q)
    kl_scale = (source * (source.clamp_min(EPS).log() - source.mean(0).clamp_min(EPS).log())).sum(1).mean().clamp_min(EPS)
    assignment = kmeans_init(x, clusters, seed=seed)
    counts = torch.bincount(assignment, minlength=clusters)
    for empty in (counts == 0).nonzero().flatten().tolist():
        donor = int(torch.where(counts[assignment] > 1)[0][0])
        counts[assignment[donor]] -= 1
        assignment[donor] = empty
        counts[empty] = 1
    centers = geometric_medians(x, assignment, clusters)
    labels = cell_means(q, assignment, clusters)

    def objective(ids, c, s):
        term, _ = mixture_terms(q, s[ids], prior, background, observations)
        return ((x - c[ids]).norm(dim=1) / dist_scale + kl_weight * term / kl_scale).mean()

    value = objective(assignment, centers, labels)
    history = [float(value)]
    converged, status = False, 'iteration_limit'
    for iteration in range(steps):
        proposals, gains = [], []
        for start in range(0, len(x), block_size):
            end = min(start + block_size, len(x))
            term, _ = mixture_terms(q[start:end], labels, prior[start:end], background, observations, True)
            cost = torch.cdist(x[start:end], centers) / dist_scale + kl_weight * term / kl_scale
            best, ids = cost.min(1)
            proposals.append(ids)
            gains.append(cost.gather(1, assignment[start:end, None]).flatten() - best)
        proposed = preserve_nonempty(assignment, torch.cat(proposals), torch.cat(gains), clusters)
        next_centers = geometric_medians(x, proposed, clusters)
        old_dist = x.new_zeros(clusters).index_add_(0, proposed, (x - centers[proposed]).norm(dim=1))
        new_dist = x.new_zeros(clusters).index_add_(0, proposed, (x - next_centers[proposed]).norm(dim=1))
        next_centers = torch.where((new_dist <= old_dist)[:, None], next_centers, centers)
        _, log_w = mixture_terms(q, labels[proposed], prior, background, observations)
        maximum = x.new_full((clusters,), -torch.inf).scatter_reduce_(0, proposed, log_w, reduce='amax')
        weight = (log_w - maximum[proposed]).exp()
        mass = x.new_zeros(clusters).index_add_(0, proposed, weight)
        sums = q.new_zeros(clusters, q.shape[1]).index_add_(0, proposed, weight[:, None] * q)
        next_labels = sums / mass[:, None]
        next_value = objective(proposed, next_centers, next_labels)
        if not bool(torch.isfinite(next_value)) or float(next_value) > float(value) + 1e-10:
            status = 'objective_rejected'
            break
        stable = torch.equal(proposed, assignment)
        change = float((next_labels - labels).abs().max())
        improvement = abs(float(value - next_value))
        assignment, centers, labels, value = proposed, next_centers, next_labels, next_value
        history.append(float(value))
        if stable and change <= tolerance and improvement <= tolerance * (1 + abs(float(value))):
            converged, status = True, 'converged'
            break
    _, log_w = mixture_terms(q, labels[assignment], prior, background, observations)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    return dict(x=centers.float().cpu(), y=labels.float().cpu(), assignment=assignment.cpu(),
        counts=torch.bincount(assignment, minlength=clusters).cpu(), nodes=clusters,
        initial_J=history[0], final_J=history[-1], history=history, converged=converged,
        status=status, sweeps=iteration + 1, posterior=log_w.exp().cpu(),
        posterior_mean=float(log_w.exp().mean()), posterior_low_share=float((log_w.exp() < .1).double().mean()),
        partition_seconds=time.perf_counter() - started)
