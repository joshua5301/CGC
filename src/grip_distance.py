import math

import torch


@torch.no_grad()
def distance_weights(distance, power):
    if not math.isfinite(power) or power < 0:
        raise ValueError('Require finite nonnegative distance power')
    if not bool(torch.isfinite(distance).all()) or bool((distance < 0).any()):
        raise ValueError('Require finite nonnegative distances')
    if power == 0:
        return torch.ones_like(distance, dtype=torch.float64)
    d = distance.double()
    positive = d[d > 0]
    scale = positive.median() if len(positive) else d.new_tensor(1.)
    log_weight = -power * torch.log1p(d / scale)
    weight = (log_weight - log_weight.max()).exp().clamp_min(1e-12)
    return weight / weight.mean()


@torch.no_grad()
def training_support_distance(features, train_mask, k, block_size=512):
    ids = train_mask.nonzero().flatten()
    if int(k) != k or not 1 <= k < len(ids):
        raise ValueError('Require integer 1 <= distance_k < labeled training count')
    x = features.double()
    result = []
    for start in range(0, len(x), block_size):
        rows = torch.arange(start, min(start + block_size, len(x)), device=x.device)
        distances = torch.cdist(x[rows], x[ids]).square()
        distances.masked_fill_(rows[:, None].eq(ids[None, :]), torch.inf)
        result.append(distances.topk(int(k), largest=False).values.mean(1))
    return torch.cat(result)
