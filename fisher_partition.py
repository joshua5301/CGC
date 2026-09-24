"""Fixed input-Fisher clustering with nonempty cells and bounded input centers."""
import numpy as np
import torch
import torch.nn.functional as F

from fisher_diagnostic import logits_and_hidden
from src.coarsening_features import project_rows


@torch.no_grad()
def build_factors(x, models, batch_size=128):
    """B_v with M_v=B_v.T B_v, averaged over frozen models; no d-by-d matrices."""
    if not models:
        raise ValueError('At least one frozen model is required')
    factors = []
    for start in range(0, len(x), batch_size):
        nodes = x[start:start+batch_size]
        parts = []
        for weights in models:
            w0, _, w1, _ = weights
            logits, hidden = logits_and_hidden(nodes, weights)
            p = logits.softmax(1)
            jacobian = torch.einsum('ch,nh,hd->ncd', w1, (hidden > 0).double(), w0)
            centered = jacobian-(p[:, :, None]*jacobian).sum(1, keepdim=True)
            parts.append(p.sqrt()[:, :, None]*centered/len(models)**.5)
        factors.append(torch.cat(parts, dim=1))
    return torch.cat(factors).contiguous()


class Geometry:
    def __init__(self, x, factors=None, batch_size=128):
        self.x, self.factors, self.batch_size = x.double(), factors, batch_size
        if factors is not None and (factors.shape[0] != len(x) or factors.shape[2] != x.shape[1]):
            raise ValueError('Invalid Fisher factor dimensions')
        self.scale = 1.
        reference = x.mean(0, keepdim=True)
        assigned = torch.zeros(len(x), dtype=torch.long, device=x.device)
        self.scale = max(float(self.assigned(reference, assigned)), 1e-12)

    def assigned(self, centers, assignment, gradient=False):
        delta = centers[assignment]-self.x
        if self.factors is None:
            costs = .5*delta.square().sum(1)
            vectors = delta if gradient else None
        else:
            projected = torch.einsum('nkd,nd->nk', self.factors, delta)
            costs = .5*projected.square().sum(1)
            vectors = torch.einsum('nkd,nk->nd', self.factors, projected) if gradient else None
        value = costs.mean()/self.scale
        if not gradient:
            return value
        grad = torch.zeros_like(centers).index_add_(0, assignment, vectors)/(len(self.x)*self.scale)
        return value, grad

    @torch.no_grad()
    def all_costs(self, centers):
        parts = []
        for start in range(0, len(self.x), self.batch_size):
            x = self.x[start:start+self.batch_size]
            if self.factors is None:
                costs = .5*torch.cdist(x, centers).square()
            else:
                b = self.factors[start:start+len(x)]
                projected = (b.reshape(-1, x.shape[1])@centers.T).reshape(len(x), b.shape[1], len(centers))
                projected -= torch.einsum('nkd,nd->nk', b, x)[:, :, None]
                costs = .5*projected.square().sum(1)
            parts.append(costs/self.scale)
        return torch.cat(parts)


def cell_labels(teacher, assignment, m):
    counts = torch.bincount(assignment, minlength=m)
    if (counts == 0).any():
        raise ValueError('Empty cell')
    return torch.zeros((m, teacher.shape[1]), dtype=teacher.dtype, device=teacher.device).index_add_(
        0, assignment, teacher)/counts[:, None]


def label_costs(teacher, labels, scale):
    return ((teacher*teacher.log()).sum(1, keepdim=True)-teacher@labels.clamp_min(1e-12).log().T)/scale


def nonempty_assignment(costs, assignment, m):
    """Exact improving moves for frozen centers/labels; deterministic, tie-stable."""
    costs = costs.detach().cpu().numpy()
    result = assignment.detach().cpu().numpy().copy()
    counts = np.bincount(result, minlength=m)
    if (counts == 0).any():
        raise ValueError('Initial assignment contains empty cells')
    for node, old in enumerate(result):
        new = int(costs[node].argmin())
        if counts[old] > 1 and costs[node, new] < costs[node, old]-1e-12:
            counts[old] -= 1
            counts[new] += 1
            result[node] = new
    return torch.from_numpy(result).to(assignment.device)


@torch.no_grad()
def optimize_centers(geometry, initial, assignment, radius, steps=100, tolerance=1e-4):
    """Convex quadratic over row balls, accelerated projected gradient with restart.

There is no ridge term. The row-ball avoids unconstrained excursions in the
poorly identified directions of a singular Fisher metric. No optimum is claimed
unless the numerical first-order gap reaches the fixed tolerance.
"""
    centers = initial.clone()
    if (centers.norm(dim=1) > radius+1e-8).any():
        raise ValueError('Initial center violates the input-radius constraint')
    value, gradient = geometry.assigned(centers, assignment, True)
    initial_value = float(value)
    reference = max(initial_value, 1e-12)
    y, momentum, step_bound = centers.clone(), 1., 1.
    for step in range(steps+1):
        gap = max(0., float((gradient*centers).sum()+radius*gradient.norm(dim=1).sum()))
        if gap <= tolerance*reference or step == steps:
            break
        step_bound = max(step_bound*.8, 1e-16)
        for restart in range(2):
            fy, gy = geometry.assigned(y, assignment, True)
            for _ in range(80):
                candidate = project_rows(y-gy/step_bound, radius)
                fc, gc = geometry.assigned(candidate, assignment, True)
                delta = candidate-y
                majorant = fy+(gy*delta).sum()+.5*step_bound*delta.square().sum()
                if float(fc-majorant) <= 1e-12*max(1., abs(float(fy))):
                    break
                step_bound *= 2
            else:
                raise RuntimeError('Center line search failed')
            if float(fc-value) <= 1e-12*max(1., abs(float(value))):
                break
            y, momentum = centers.clone(), 1.
        else:
            raise RuntimeError('Center update increased objective')
        # Retain the old point on a roundoff-only increase.
        if float(fc) > float(value):
            break
        previous = centers
        centers, value, gradient = candidate, fc, gc
        next_momentum = (1+(1+4*momentum**2)**.5)/2
        y = centers+(momentum-1)/next_momentum*(centers-previous)
        momentum = next_momentum
    gap = max(0., float((gradient*centers).sum()+radius*gradient.norm(dim=1).sum()))
    return centers, dict(initial=initial_value, final=float(value), steps=step, gap=gap,
        status='gap_tolerance' if gap <= tolerance*reference else 'iteration_or_numerical_limit')


@torch.no_grad()
def refine(x, teacher, centers, assignment, factors=None, coefficient=1.,
           outer_steps=20, center_steps=100, batch_size=128, log=print):
    x, teacher, centers = x.double(), teacher.double(), centers.double().clone()
    assignment = assignment.clone().long()
    if (coefficient < 0 or outer_steps < 1 or center_steps < 1
            or not all(torch.isfinite(v).all() for v in (x, teacher, centers))):
        raise ValueError('Invalid refinement input')
    m = len(centers)
    geometry = Geometry(x, factors, batch_size)
    label_scale = max(float((teacher*(teacher.log()-teacher.mean(0).log())).sum(1).mean()), 1e-12)
    radius = float(x.norm(dim=1).max())
    labels = cell_labels(teacher, assignment, m)

    def objective(c, a, y):
        geom = float(geometry.assigned(c, a))
        kl = float((teacher*(teacher.log()-y[a].clamp_min(1e-12).log())).sum(1).mean())/label_scale
        return dict(objective=geom+coefficient*kl, geometry=geom, label_kl=kl)

    current = objective(centers, assignment, labels)
    history = [dict(iteration=0, moved=0, **current)]
    status = 'outer_limit'
    for iteration in range(1, outer_steps+1):
        costs = geometry.all_costs(centers)+coefficient*label_costs(teacher, labels, label_scale)
        proposed = nonempty_assignment(costs, assignment, m)
        moved = int((proposed != assignment).sum())
        next_labels = cell_labels(teacher, proposed, m)
        next_centers, solver = optimize_centers(geometry, centers, proposed, radius, center_steps)
        updated = objective(next_centers, proposed, next_labels)
        if updated['objective'] > current['objective']+1e-9*max(1., abs(current['objective'])):
            raise RuntimeError('Refinement objective increased')
        if updated['objective'] > current['objective']:
            status = 'numerical_stall'
            break
        improvement = current['objective']-updated['objective']
        centers, assignment, labels, current = next_centers, proposed, next_labels, updated
        history.append(dict(iteration=iteration, moved=moved, center_solver=solver, **current))
        log(history[-1])
        if moved == 0 and improvement <= 1e-6*max(1., abs(current['objective'])):
            status = 'objective_stall'
            break
    return dict(x=centers.cpu(), y=labels.cpu(), assign=assignment.cpu(), history=history,
        geometry_scale=geometry.scale, label_scale=label_scale, radius=radius, status=status,
        caveat='Fixed local quadratic metric; no global CE-risk or retraining certificate.')
