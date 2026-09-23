"""Held-out label residual correction with a conditional Lipschitz error bound.

The residual Lipschitz constant is an assumption, not estimated/certified here.
No validation/test labels are accepted by this module.
"""
import math
import torch


def split_training_indices(train_mask, fraction=.5, seed=0):
    """Split by RNG independent of labels; no label-based stratification."""
    indices = train_mask.detach().cpu().nonzero().flatten()
    if not 0 < fraction < 1 or len(indices) < 4:
        raise ValueError('Need >=4 training nodes and 0 < calibration fraction < 1')
    n_cal = min(len(indices)-2, max(2, round(len(indices)*fraction)))
    order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))
    return indices[order[n_cal:]], indices[order[:n_cal]]


def project_simplex(values):
    """Euclidean projection, not elementwise clipping plus renormalization."""
    ordered = values.sort(dim=-1, descending=True).values
    sums = ordered.cumsum(-1)-1
    rank = torch.arange(1, values.shape[-1]+1, device=values.device, dtype=values.dtype)
    active = ordered > sums/rank
    rho = active.sum(-1, keepdim=True).clamp_min(1)-1
    threshold = sums.gather(-1, rho)/(rho.to(values.dtype)+1)
    return (values-threshold).clamp_min(0)


@torch.no_grad()
def residual_correction(features, probabilities, calibration_indices, calibration_labels,
                        slope=2., delta=.05, candidate_k=(1, 2, 4, 8, 16, 32, 64), batch_size=512):
    """Select k using bias + Hoeffding radius; then add the mean label residual.

    Assumes ||(p-F)_u-(p-F)_v||_2 <= slope * ||H_u-H_v||_2 / scale.
    Teacher must not have trained on calibration labels. Conditional on fixed
    features/teacher and independent calibration labels, union-bound coverage
    holds over all N targets and candidate k, IF this slope assumption holds.
    Returns the raw bound before clipping at the simplex diameter sqrt(2).
    """
    h, f = features.double(), probabilities.double()
    indices = torch.as_tensor(calibration_indices, device=h.device, dtype=torch.long)
    labels = torch.as_tensor(calibration_labels, device=h.device, dtype=torch.long)
    n, classes = f.shape
    if (h.ndim != 2 or len(h) != n or indices.ndim != 1 or labels.shape != indices.shape
            or len(indices) < 2 or len(indices.unique()) != len(indices)
            or indices.min() < 0 or indices.max() >= n or labels.min() < 0 or labels.max() >= classes):
        raise ValueError('Invalid features/calibration indices/labels')
    if (not torch.isfinite(h).all() or not torch.isfinite(f).all() or (f < 0).any()
            or not torch.allclose(f.sum(1), torch.ones(n, device=f.device, dtype=f.dtype))):
        raise ValueError('Require finite features and simplex probabilities')
    if not math.isfinite(slope) or slope < 0 or not 0 < delta < 1 or batch_size < 1:
        raise ValueError('Invalid slope/delta/batch_size')
    if not candidate_k or any(int(k) != k or k < 1 for k in candidate_k):
        raise ValueError('candidate_k must be positive integers')
    ks = sorted({min(int(k), len(indices)) for k in candidate_k} | {len(indices)})
    k_tensor = torch.tensor(ks, device=h.device, dtype=torch.long)
    anchors = h[indices]
    # Scale depends only on features, so it does not select using label noise.
    nearest = []
    for start in range(0, len(indices), batch_size):
        d = torch.cdist(anchors[start:start+batch_size], anchors)
        row = torch.arange(len(d), device=h.device)
        d[row, row+start] = torch.inf
        nearest.append(d.min(1).values)
    nearest = torch.cat(nearest)
    positive = nearest[nearest > 0]
    scale = float(positive.median()) if len(positive) else 1.
    residual = torch.nn.functional.one_hot(labels, classes).double()-f[indices]
    noise = torch.sqrt(classes*math.log(2*n*classes*len(ks)/delta)/(2*k_tensor.double()))
    corrected, chosen, bounds = [], [], []
    for start in range(0, n, batch_size):
        distances, order = torch.cdist(h[start:start+batch_size], anchors).sort(dim=1, stable=True)
        bias = slope * distances.cumsum(1)[:, k_tensor-1]/k_tensor/scale
        bound = bias+noise[None]
        choice = bound.argmin(1)
        k = k_tensor[choice]
        sums = residual[order].cumsum(1)
        rows = torch.arange(len(order), device=h.device)
        correction = sums[rows, k-1]/k[:, None]
        corrected.append(project_simplex(f[start:start+len(order)]+correction))
        chosen.append(k)
        bounds.append(bound[rows, choice])
    corrected, chosen, bounds = torch.cat(corrected), torch.cat(chosen), torch.cat(bounds)
    values, counts = chosen.unique(return_counts=True)
    return dict(probabilities=corrected, chosen_k=chosen, conditional_bound=bounds,
                diagnostics=dict(slope=slope, delta=delta, distance_scale=scale, candidate_k=ks,
                    k_counts={str(int(k)): int(c) for k, c in zip(values, counts)},
                    median_bound=float(bounds.median()),
                    nonvacuous_fraction=float((bounds < math.sqrt(2)).double().mean()),
                    mean_probability_shift=float((corrected-f).norm(dim=1).mean()),
                    coverage_certified=False, residual_smoothness_verified=False))
