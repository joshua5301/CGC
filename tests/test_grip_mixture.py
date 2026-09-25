import numpy as np
import torch

import src.grip_mixture as module


def test_quantization_preserves_integer_budget_and_simplex():
    q = torch.tensor([[.5, .3, .2], [.333, .333, .334]], dtype=torch.float64)
    observed = module.quantize_labels(q, 7)
    torch.testing.assert_close(observed.sum(1), torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(observed * 7, (observed * 7).round())
    assert bool(((observed - q).abs() <= 1 / 7).all())


def test_posterior_matches_normalized_multinomial_likelihood():
    q = torch.tensor([[.75, .25]], dtype=torch.float64)
    s = torch.tensor([[.8, .2], [.2, .8]], dtype=torch.float64)
    b = torch.tensor([.4, .6], dtype=torch.float64)
    rho = torch.tensor([.65], dtype=torch.float64)
    cost, log_w = module.mixture_terms(q, s, rho, b, 4, True)
    clean = torch.distributions.Multinomial(4, probs=s).log_prob(q * 4) + rho.log()
    noise = torch.distributions.Multinomial(4, probs=b).log_prob(q * 4) + torch.log1p(-rho)
    log_mix = torch.logaddexp(clean, noise)
    torch.testing.assert_close(log_w[0], clean - log_mix)
    torch.testing.assert_close(cost[0, 0] - cost[0, 1], -(log_mix[0] - log_mix[1]) / 4)
    clean_cost, clean_w = module.mixture_terms(q, s, torch.ones_like(rho), b, 4, True)
    torch.testing.assert_close(clean_cost[0], (q * (q.log() - s.log())).sum(1))
    torch.testing.assert_close(clean_w, torch.zeros_like(clean_w))


def test_noise_fit_does_not_read_selection_labels():
    q = torch.tensor([[.8, .2], [.7, .3], [.2, .8], [.1, .9]], dtype=torch.float64)
    labels = torch.tensor([0, 1, 1, 0])
    changed = torch.tensor([0, 1, 0, 1])
    args = (q, np.array([0, 1]))
    first = module.fit_contamination(*args, labels, np.arange(4.), 8)
    second = module.fit_contamination(*args, changed, np.arange(4.), 8)
    for a, b in zip(first[:3], second[:3]):
        torch.testing.assert_close(a, b)


def test_em_preserves_budget_and_descends(monkeypatch):
    monkeypatch.setattr(module, 'kmeans_init', lambda x, n, seed: torch.tensor([0, 0, 1, 1]))
    x = torch.tensor([[0.], [.1], [3.], [3.1]], dtype=torch.float64)
    q = torch.tensor([[.9, .1], [.7, .3], [.2, .8], [.1, .9]], dtype=torch.float64)
    q = module.quantize_labels(q, 10)
    result = module.mixture_partition(x, q, 2, torch.full((4,), .8), torch.tensor([.5, .5]), 10, steps=100)
    assert np.all(np.diff(result['history']) <= 1e-10)
    assert result['counts'].sum() == 4
    assert bool((result['counts'] > 0).all())
    assert bool(torch.isfinite(result['posterior']).all())
    torch.testing.assert_close(result['y'].sum(1), torch.ones(2))
