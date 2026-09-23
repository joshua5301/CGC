"""GCN-faithful fixed coarsening and convex message-residual feature fitting.

All operators use PyG's source_to_target GCN normalization. Coarse edge weights
already include aggregated self loops; they must not be normalized twice.
"""
import math
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_remaining_self_loops


def gcn_operator(edge_index, edge_weight, n):
    ei = edge_index.detach().cpu().long()
    ew = torch.ones(ei.shape[1], dtype=torch.float64) if edge_weight is None else edge_weight.detach().cpu().double()
    if not torch.isfinite(ew).all() or (ew < 0).any():
        raise ValueError('Expected finite nonnegative edge weights')
    ei, ew = gcn_norm(ei, ew, n, add_self_loops=True, dtype=torch.float64)
    return torch.sparse_coo_tensor(ei.flip(0), ew, (n, n)).coalesce()


def fixed_coarsening(edge_index, edge_weight, n, assignment):
    """Returns raw coarse weighted edges and actual normalized P,Q (CPU float64)."""
    a = assignment.detach().cpu().numpy().astype(np.int64)
    if a.shape != (n,) or n < 1 or a.min() < 0:
        raise ValueError('Invalid assignment')
    m = int(a.max())+1
    counts = np.bincount(a, minlength=m)
    if (counts == 0).any():
        raise ValueError('Assignment must use contiguous nonempty cells')
    ei = edge_index.detach().cpu().long()
    ew = torch.ones(ei.shape[1], dtype=torch.float64) if edge_weight is None else edge_weight.detach().cpu().double()
    P = gcn_operator(ei, ew, n)
    # Exactly the same loop insertion as gcn_norm, before pooling raw weights.
    loop_ei, loop_ew = add_remaining_self_loops(ei, ew, fill_value=1., num_nodes=n)
    A = sp.csr_matrix((loop_ew.numpy(), (loop_ei[1].numpy(), loop_ei[0].numpy())), shape=(n, n))
    S = sp.csr_matrix((np.ones(n), (np.arange(n), a)), shape=(n, m))
    Ac = (S.T @ A @ S).tocoo()
    coarse_ei = torch.from_numpy(np.vstack([Ac.col, Ac.row])).long()
    coarse_ew = torch.from_numpy(Ac.data.copy()).double()
    Q = gcn_operator(coarse_ei, coarse_ew, m).to_dense()
    return dict(P=P, Q=Q, edge_index=coarse_ei, edge_weight=coarse_ew,
                counts=torch.from_numpy(counts), assignment=torch.from_numpy(a))


def project_rows(C, radius):
    norms = C.norm(dim=1, keepdim=True)
    return C * (radius/norms.clamp_min(1e-300)).clamp(max=1.)


def residual_value_grad(C, Q, target, assignment, omega, epsilon):
    """Smooth upper approximation, analytic gradient, and unsmoothed D_prop."""
    residual = (Q@C)[assignment]-target
    squared = residual.square().sum(1)
    smooth = (squared+epsilon**2).sqrt()
    value = (omega*smooth).mean()
    raw = (omega*squared.sqrt()).mean()
    aggregated = torch.zeros_like(C).index_add_(0, assignment,
        (omega/smooth)[:, None]*residual/len(target))
    gradient = Q.T@aggregated
    return value, gradient, raw


@torch.no_grad()
def optimize_features(target, Q, assignment, omega, initial, radius,
                      steps=1000, tolerance=1e-3, smooth_relative=1e-4, log=print):
    """Monotone restarted accelerated projected gradient with backtracking.

Stops on a convex first-order gap certificate, not small iterate changes. The
returned raw-cost best iterate includes the initialization. The certificate is
for the float64, fixed-S/Q feature problem, NOT for GCN risk or student training.
"""
    target, Q, omega, C = [v.detach().double() for v in (target, Q, omega, initial)]
    assignment = assignment.long().to(target.device)
    if (not all(torch.isfinite(v).all() for v in (target, Q, omega, C))
            or not math.isfinite(radius) or radius < 0 or steps < 1
            or not math.isfinite(tolerance) or tolerance <= 0
            or not math.isfinite(smooth_relative) or smooth_relative <= 0
            or (omega < 0).any() or omega.sum() <= 0
            or target.ndim != 2 or Q.shape != (len(C), len(C))
            or C.shape[1] != target.shape[1] or omega.shape != (len(target),)
            or assignment.shape != (len(target),) or assignment.min() < 0 or assignment.max() >= len(C)):
        raise ValueError('Invalid feature fitting problem')
    C = project_rows(C.clone(), radius)
    initial_raw = float((omega*((Q@C)[assignment]-target).norm(dim=1)).mean())
    scale = max(initial_raw, 1e-12)
    epsilon = max(smooth_relative*scale/float(omega.mean()), 1e-12)
    smoothing_slack = epsilon*float(omega.mean())
    value, gradient, raw = residual_value_grad(C, Q, target, assignment, omega, epsilon)
    best, best_raw = C.clone(), float(raw)
    y, momentum, L = C.clone(), 1., 1.
    history = []
    status = 'max_steps'
    if radius == 0 or initial_raw == 0:
        return dict(x=C, history=[dict(step=0, raw=float(raw), smooth=float(value), gap=0.)],
            initial=initial_raw, final=float(raw), epsilon=epsilon, smoothing_slack=smoothing_slack,
            gap=0., raw_suboptimality_upper=0., status='exact_trivial_optimum', steps=0)
    for step in range(steps+1):
        if step % 10 == 0 or step == steps:
            # Convexity: f(C)-f* <= <g,C> + R*sum_j ||g_j||.
            gap = max(0., float((gradient*C).sum()+radius*gradient.norm(dim=1).sum()))
            record = dict(step=step, raw=float(raw), smooth=float(value), gap=gap)
            history.append(record)
            log(record)
            if gap+smoothing_slack <= tolerance*scale:
                status = 'gap_tolerance'
                break
        if step == steps:
            break
        L = max(L*.8, 1e-16)
        # Restart acceleration whenever extrapolation fails monotone descent.
        for restart in range(2):
            fy, gy, _ = residual_value_grad(y, Q, target, assignment, omega, epsilon)
            for _ in range(80):
                candidate = project_rows(y-gy/L, radius)
                fc, gc, rc = residual_value_grad(candidate, Q, target, assignment, omega, epsilon)
                delta = candidate-y
                majorant = fy+(gy*delta).sum()+.5*L*delta.square().sum()
                if float(fc-majorant) <= 1e-12*max(1., abs(float(fy))):
                    break
                L *= 2
            else:
                raise RuntimeError('Feature optimizer line search failed')
            if float(fc-value) <= 1e-12*max(1., abs(float(value))):
                break
            y, momentum = C.clone(), 1.
        else:
            raise RuntimeError('Projected step increased smooth objective')
        previous = C
        C, value, gradient, raw = candidate, fc, gc, rc
        if float(raw) < best_raw:
            best, best_raw = C.clone(), float(raw)
        next_momentum = (1+math.sqrt(1+4*momentum**2))/2
        y = C+(momentum-1)/next_momentum*(C-previous)
        momentum = next_momentum
    # Certificate must refer to the returned iterate, not a discarded last step.
    value, gradient, raw = residual_value_grad(best, Q, target, assignment, omega, epsilon)
    gap = max(0., float((gradient*best).sum()+radius*gradient.norm(dim=1).sum()))
    raw_upper = gap+smoothing_slack
    status = 'gap_tolerance' if raw_upper <= tolerance*scale else 'max_steps'
    return dict(x=best, history=history, initial=initial_raw, final=best_raw,
        epsilon=epsilon, smoothing_slack=smoothing_slack, gap=gap,
        raw_suboptimality_upper=raw_upper, status=status, steps=step)


@torch.no_grad()
def feature_diagnostics(P, Q, X, C, assignment):
    device = X.device
    P, Q, X, C = P.to(device).double(), Q.to(device).double(), X.double(), C.double()
    a = assignment.to(device).long()
    omega = torch.sparse.sum(P, dim=0).to_dense()
    S = torch.nn.functional.one_hot(a, len(C)).double()
    residual = torch.sparse.mm(P, S)-Q[a]
    return dict(D_prop=float((omega*(torch.sparse.mm(P, X)-(Q@C)[a]).norm(dim=1)).mean()),
        D_struct=float(residual.abs().sum()/len(X)),
        raw_error=float((X-C[a]).norm(dim=1).mean()),
        max_feature_norm=float(C.norm(dim=1).max()))
