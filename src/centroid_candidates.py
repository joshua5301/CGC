import torch


@torch.no_grad()
def nearest_candidates(features, centers, sizes, block_size=16):
    features, centers = features.double(), centers.double()
    sizes = torch.as_tensor(sizes, device=features.device, dtype=torch.long)
    if sizes.shape != (len(centers),) or bool((sizes < 1).any()) or bool((sizes > len(features)).any()):
        raise ValueError('Candidate counts must be between one and the number of nodes')
    norms = features.square().sum(1)
    nodes, groups = [], []
    for start in range(0, len(centers), block_size):
        query = centers[start:start + block_size]
        distances = (query.square().sum(1)[:, None] + norms[None, :] - 2 * query @ features.T).clamp_min(0)
        order = distances.argsort(dim=1, stable=True)
        for offset, row in enumerate(order):
            cluster = start + offset
            selected = row[:int(sizes[cluster])]
            nodes.append(selected)
            groups.append(torch.full_like(selected, cluster))
    return torch.cat(nodes), torch.cat(groups)


def candidate_statistics(nodes, groups, original_assignment):
    reuse = torch.bincount(nodes, minlength=len(original_assignment))
    counts = torch.bincount(groups)
    return dict(candidate_slots=len(nodes), unique_candidates=int((reuse > 0).sum()),
                candidate_count_min=int(counts.min()), candidate_count_max=int(counts.max()),
                candidate_count_mean=float(counts.float().mean()), max_candidate_reuse=int(reuse.max()),
                outside_cluster_fraction=float((original_assignment[nodes] != groups).float().mean()))
