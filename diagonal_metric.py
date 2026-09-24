"""Global diagonal sensitivity metric for propagated-input GRIP."""
import torch
from src.partition import partition


@torch.no_grad()
def sensitivity(z, models, batch_size=128):
    """Mean squared centered-logit Jacobian columns, with dropout disabled."""
    total = torch.zeros(z.shape[1], dtype=torch.float64, device=z.device)
    per_model = []
    for w0, b0, w1, _ in models:
        score = torch.zeros_like(total)
        centered_w1 = w1 - w1.mean(0, keepdim=True)
        for x in z.double().split(batch_size):
            gate = (x @ w0.T + b0 > 0).double()
            jac = torch.einsum('ch,nh,hd->ncd', centered_w1, gate, w0)
            score += jac.square().sum((0, 1))
        score /= len(z)
        per_model.append(score.cpu())
        total += score
    total /= len(models)
    if not torch.isfinite(total).all():
        raise ValueError('Nonfinite sensitivity')
    return dict(score=total.cpu(), per_model=torch.stack(per_model))


@torch.no_grad()
def condense(z, teacher, score, alpha, cluster_num, coefficient, steps=100):
    if not 0 <= alpha < 1:
        raise ValueError('alpha must lie in [0, 1)')
    score = score.to(z.device).double()
    relative = score / score.mean() if float(score.mean()) > 0 else torch.ones_like(score)
    weight = (1 - alpha) + alpha * relative
    diagonal = weight.sqrt()
    centers, labels, assignment = partition(z.double() * diagonal, teacher,
                                            cluster_num, coefficient, steps)
    if len(centers) != cluster_num:
        raise RuntimeError('GRIP dropped empty cells; refusing unequal-budget comparison')
    return dict(x=(centers / diagonal).cpu(), y=labels.cpu(), assign=assignment.cpu(),
                weight=weight.cpu(), alpha=alpha, status='scaled-original-GRIP',
                weight_min=float(weight.min()), weight_max=float(weight.max()),
                caveat='Sensitivity estimates do not certify a Lipschitz bound or student transfer.')
