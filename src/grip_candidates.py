import torch

from src.partition import kmeans_init


def candidate_pool(train_mask, source, seed=0):
    ids = train_mask.nonzero(as_tuple=True)[0].cpu()
    if source == 'random':
        ids = torch.randperm(len(train_mask), generator=torch.Generator().manual_seed(seed))[:len(ids)]
    elif source != 'train':
        raise ValueError('Candidate source must be train or random')
    return ids.sort().values


def coverage_seeds(x, candidates, count, block_size=256):
    ids = candidates.to(device=x.device, dtype=torch.long).sort().values
    if not 1 <= count <= len(ids) or len(ids.unique()) != len(ids):
        raise ValueError('Require distinct candidates and a feasible center count')
    x = x.double()
    distances = torch.cdist(x, x[ids]).square()
    nearest = x.new_full((len(x),), float('inf'))
    available = torch.ones(len(ids), dtype=torch.bool, device=x.device)
    chosen, history = [], []
    for _ in range(count):
        costs = torch.cat([
            torch.minimum(nearest[:, None], chunk).sum(0)
            for chunk in distances.split(block_size, dim=1)
        ])
        costs[~available] = float('inf')
        index = int(costs.argmin())
        chosen.append(index)
        available[index] = False
        nearest = torch.minimum(nearest, distances[:, index])
        history.append(float(nearest.mean()))
    return ids[chosen].cpu(), history


def candidate_initialization(x, train_mask, count, source, candidate_seed=0, kmeans_seed=1234):
    candidates = candidate_pool(train_mask, source, candidate_seed)
    ids, history = coverage_seeds(x, candidates, count)
    assignment = kmeans_init(x, count, seed=kmeans_seed, initial_centers=x[ids.to(x.device)])
    return dict(candidate_ids=candidates, initial_centroid_ids=ids,
                coverage_history=history, assignment=assignment.cpu())
