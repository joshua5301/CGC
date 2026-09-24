"""Input diagonal Fisher of the fixed Nyström GRIP kernel teacher."""
import torch
import torch.nn.functional as F
from src.teacher import get_kernel_values, fit_logistic

METRIC_TEMPERATURE = 1.0


def teacher_fisher(z, labels, train_mask, reference_probabilities, config, batch_size=64):
    # Fit a fixed-landmark kernel teacher; optionally verify supplied probabilities.
    if config['kernel'] != 'relu' or config['basis'] < 1:
        raise ValueError('Expected a ReLU teacher and positive landmark count')
    with torch.no_grad():
        # Match the existing 3000-landmark teacher convention with a fixed CPU draw.
        generator = torch.Generator(device='cpu').manual_seed(config.get('condensation_seed', 0))
        indices = (torch.arange(len(z)) if config['basis'] >= len(z) else
                   torch.randperm(len(z), generator=generator)[:config['basis']])
        basis = z.detach().double()[indices.to(z.device)].clone()
        kernel = get_kernel_values(basis, basis, 'relu')
        gram = (kernel + kernel.T) / 2
        eye = torch.eye(len(basis), dtype=basis.dtype, device=basis.device)
        chol = torch.linalg.cholesky(gram + 1e-8 * gram.diagonal().mean() * eye)
        transform = torch.linalg.solve_triangular(chol, eye, upper=False).T
        query_kernel = get_kernel_values(z.double(), basis, 'relu')
        features = query_kernel @ transform
        classes = reference_probabilities.shape[1] if reference_probabilities is not None else int(labels.max()) + 1
        targets = F.one_hot(labels[train_mask], classes).double()
    head = fit_logistic(features[train_mask], targets, config['gamma'])
    with torch.no_grad():
        beta = (transform @ head).detach()
        logits = features @ head
        rebuilt = (logits / config['T']).softmax(1)
        folded = (query_kernel @ beta / config['T']).softmax(1)
        source_error = float((rebuilt - reference_probabilities).abs().max()) if reference_probabilities is not None else 0.
        fold_error = float((rebuilt - folded).abs().max())
        if source_error > 1e-7 or fold_error > 1e-9:
            raise ValueError(f'Teacher reconstruction mismatch: source={source_error}, folded={fold_error}')
        del features, transform, kernel, query_kernel, gram, chol, eye, head
    score = torch.zeros(z.shape[1], dtype=torch.float64, device=z.device)
    for chunk in z.split(batch_size):
        x = chunk.detach().double().clone().requires_grad_(True)
        # Basis, bandwidth and fitted parameters stay fixed. Differentiate query only.
        logp = (get_kernel_values(x, basis, 'relu') @ beta / METRIC_TEMPERATURE).log_softmax(1)
        probabilities = logp.detach().exp()
        for c in range(logp.shape[1]):
            gradient, = torch.autograd.grad(logp[:, c].sum(), x,
                                            retain_graph=c + 1 < logp.shape[1])
            score += (probabilities[:, c, None] * gradient.square()).sum(0).detach()
    score /= len(z)
    if not torch.isfinite(score).all() or float(score.mean()) <= 0:
        raise ValueError('Invalid teacher Fisher sensitivities')
    return dict(score=score.cpu(), landmark_indices=indices.cpu(), basis=basis.cpu(), beta=beta.cpu(), logits=logits.cpu(),
                source_probability_checked=reference_probabilities is not None,
                metric_temperature=METRIC_TEMPERATURE, label_temperature=config['T'],
                source_probability_max_error=source_error,
                folded_probability_max_error=fold_error,
                definition='mean_nodes sum_classes p_c * (d log p_c / d z_k)^2')
