"""Matched, anchored medoid refinement in fixed explicit feature spaces."""
import torch
from fisher_partition import cell_labels
from src.teacher import get_kernel_features


@torch.no_grad()
def refine_medoid(z, teacher, centers, assignment, *, kernel=False,
                  coefficient=1., outer_steps=30, basis_num=3000):
    z, teacher = z.double(), teacher.double()
    a = assignment.clone().long()
    m = len(centers)
    # Identical starting source nodes for both geometries.
    selected = []
    for j in range(m):
        ids = torch.where(a == j)[0]
        if not len(ids):
            raise ValueError('Empty source cell')
        selected.append(ids[(z[ids] - centers[j]).square().sum(1).argmin()])
    medoids = torch.stack(selected)
    initial_medoids = medoids.clone()
    features = get_kernel_features(z, 'relu', basis_num) if kernel else z
    distances = torch.cdist(features, features).clamp_min_(0)
    distances.fill_diagonal_(0)
    scale = float((features - features.mean(0)).norm(dim=1).mean())
    if not scale > 1e-15:
        raise ValueError('Degenerate feature geometry')
    distances /= scale
    entropy = (teacher * teacher.clamp_min(1e-12).log()).sum(1)
    label_scale = max(float((entropy - teacher @ teacher.mean(0).clamp_min(1e-12).log()).mean()), 1e-12)
    rows = torch.arange(len(z), device=z.device)
    cells = torch.arange(m, device=z.device)

    def objective():
        labels = cell_labels(teacher, a, m)
        geometry = distances[rows, medoids[a]].mean()
        kl = (entropy - (teacher * labels[a].clamp_min(1e-12).log()).sum(1)).mean() / label_scale
        return labels, dict(objective=float(geometry + coefficient * kl),
                            geometry=float(geometry), label_kl_normalized=float(kl))

    labels, initial = objective()
    history = [dict(step=0, **initial)]
    status = 'max_steps'
    for step in range(1, outer_steps + 1):
        old_a, old_medoids = a.clone(), medoids.clone()
        kl_costs = (entropy[:, None] - teacher @ labels.clamp_min(1e-12).log().T) / label_scale
        costs = distances[:, medoids] + coefficient * kl_costs
        proposal = costs.argmin(1)
        improve = costs[rows, proposal] < costs[rows, a] - 1e-12
        a = torch.where(improve, proposal, a)
        # Anchors guarantee distinct, nonempty cells and feasible medoid updates.
        a[medoids] = cells
        for j in range(m):
            ids = torch.where(a == j)[0]
            sums = distances[ids[:, None], ids[None, :]].sum(0)
            best = sums.argmin()
            incumbent = torch.where(ids == medoids[j])[0][0]
            if sums[best] < sums[incumbent] - 1e-12:
                medoids[j] = ids[best]
        labels, value = objective()
        if value['objective'] > history[-1]['objective'] + 1e-9 * max(1., abs(history[-1]['objective'])):
            raise RuntimeError('Medoid objective increased')
        history.append(dict(step=step, moved=int((old_a != a).sum()), **value))
        if torch.equal(a, old_a) and torch.equal(medoids, old_medoids):
            status = 'converged'
            break
    if len(medoids.unique()) != m or not torch.equal(a[medoids], cells):
        raise RuntimeError('Invalid medoid partition')
    return dict(x=z[medoids].cpu(), y=labels.cpu(), assign=a.cpu(),
                medoids=medoids.cpu(), initial_medoids=initial_medoids.cpu(),
                history=history, geometry_scale=scale, label_scale=label_scale,
                feature_dimension=features.shape[1], status=status,
                algorithm='anchored within-cell medoids; fixed norm distances plus teacher KL',
                caveat='Descent of clustering objective, not a guarantee on retrained student risk.')
