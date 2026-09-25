import numpy as np
import torch

import src.partition as module
from src.grip_reliability import fit_reliability, support_distances, blend_reliability, reliability_choices


def test_support_excludes_self_and_unlabeled_nodes():
    x = torch.tensor([[0.], [2.], [5.], [1.]])
    mask = torch.tensor([True, True, True, False])
    result = support_distances(x, mask, [1, 2], block_size=2)
    np.testing.assert_allclose(result[1], [4., 4., 9., 1.])
    np.testing.assert_allclose(result[2], [14.5, 6.5, 17., 1.])


def test_isotonic_uses_calibration_only_and_clips_extrapolation():
    distance = np.array([0., 1., 2., 3., 100.])
    weights, predicted, _ = fit_reliability(distance, [0, 1, 2], np.array([.1, .7, .3]), (.25, 4.))
    np.testing.assert_allclose(predicted, [.1, .5, .5, .5, .5])
    np.testing.assert_allclose(weights.mean(), 1.)
    assert np.all(np.diff(weights) <= 0)


def test_unit_weights_reproduce_baseline(monkeypatch):
    monkeypatch.setattr(module, 'kmeans_init', lambda x, n, **kw: torch.tensor([0, 0, 1, 1]))
    x = torch.tensor([[0.], [.1], [2.], [2.1]])
    q = torch.tensor([[.9, .1], [.7, .3], [.2, .8], [.1, .9]])
    baseline = module.partition(x, q, 2, return_diagnostics=True)
    weighted = module.partition(x, q, 2, label_weights=torch.ones(4), return_diagnostics=True)
    for key in ('x', 'y', 'assignment'):
        torch.testing.assert_close(baseline[key], weighted[key])
    np.testing.assert_allclose(baseline['final_J'], weighted['final_J'])


def test_representative_label_uses_reliability_not_mass(monkeypatch):
    monkeypatch.setattr(module, 'kmeans_init', lambda x, n, **kw: torch.zeros(len(x), dtype=torch.long))
    x = torch.tensor([[0.], [1.], [2.]])
    q = torch.tensor([[.9, .1], [.3, .7], [.1, .9]])
    weights = torch.tensor([4., 1., 1.])
    result = module.partition(x, q, 1, label_weights=weights, return_diagnostics=True)
    torch.testing.assert_close(result['y'][0], (q * weights[:, None]).sum(0) / weights.sum())
    torch.testing.assert_close(result['counts'], torch.tensor([3]))


def test_alpha_controls_deviation_without_changing_estimator():
    base, _, _ = fit_reliability(np.array([0., 1., 2.]), [0, 1, 2], np.array([0., .5, 1.]), (.25, 4.))
    np.testing.assert_array_equal(blend_reliability(base, 0), np.ones(3))
    np.testing.assert_allclose(blend_reliability(base, 1), base)
    for alpha in [.1, .5]:
        w = blend_reliability(base, alpha)
        np.testing.assert_allclose(w.mean(), 1.)
        np.testing.assert_allclose(w - 1, alpha * (base - 1))
    assert np.isfinite(base).all()


def test_zero_alpha_is_single_unweighted_candidate():
    assert reliability_choices([3, 10], [1., 0., .5, 0.]) == [
        (None, 0.), (3, .5), (10, .5), (3, 1.), (10, 1.)]
