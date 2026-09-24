"""Input diagonal Fisher of the fixed full-landmark GRIP kernel teacher."""
import torch
import torch.nn.functional as F
from src.teacher import get_kernel_values, fit_logistic


def teacher_fisher(z, labels, train_mask, reference_probabilities, config, batch_size=64):
    # The earlier diagnostic saved probabilities, but not the fitted teacher.
    # Reconstruct exactly its full-landmark model and verify against those labels.
    if config['kernel'] != 'relu' or config['basis'] < len(z):
        raise ValueError('This reconstruction requires the full-landmark ReLU source teacher')
    with torch.no_grad():
        basis = z.detach().double().clone()
        kernel = get_kernel_values(basis, basis, 'relu')
        gram = (kernel + kernel.T) / 2
        eye = torch.eye(len(basis), dtype=basis.dtype, device=basis.device)
        chol = torch.linalg.cholesky(gram + 1e-8 * gram.diagonal().mean() * eye)
        transform = torch.linalg.solve_triangular(chol, eye, upper=False).T
        features = kernel @ transform
        targets = F.one_hot(labels[train_mask], reference_probabilities.shape[1]).double()
    head = fit_logistic(features[train_mask], targets, config['gamma'])
    with torch.no_grad():
        beta = (transform @ head).detach()
        rebuilt = (features @ head / config['T']).softmax(1)
        folded = (kernel @ beta / config['T']).softmax(1)
        source_error = float((rebuilt - reference_probabilities).abs().max())
        fold_error = float((rebuilt - folded).abs().max())
        if source_error > 1e-7 or fold_error > 1e-9:
            raise ValueError(f'Teacher reconstruction mismatch: source={source_error}, folded={fold_error}')
        del features, transform, kernel, gram, chol, eye, head
    score = torch.zeros(z.shape[1], dtype=torch.float64, device=z.device)
    for chunk in z.split(batch_size):
        x = chunk.detach().double().clone().requires_grad_(True)
        # Basis, bandwidth and fitted parameters stay fixed. Differentiate query only.
        logp = (get_kernel_values(x, basis, 'relu') @ beta / config['T']).log_softmax(1)
        probabilities = logp.detach().exp()
        for c in range(logp.shape[1]):
            gradient, = torch.autograd.grad(logp[:, c].sum(), x,
                                            retain_graph=c + 1 < logp.shape[1])
            score += (probabilities[:, c, None] * gradient.square()).sum(0).detach()
    score /= len(z)
    if not torch.isfinite(score).all() or float(score.mean()) <= 0:
        raise ValueError('Invalid teacher Fisher sensitivities')
    return dict(score=score.cpu(), basis=basis.cpu(), beta=beta.cpu(),
                temperature=config['T'], source_probability_max_error=source_error,
                folded_probability_max_error=fold_error,
                definition='mean_nodes sum_classes p_c * (d log p_c / d z_k)^2')
