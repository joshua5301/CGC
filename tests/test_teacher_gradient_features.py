import numpy as np
import torch

from src.empirical_ntk_study import make_network, sketch_grams
from src.ntk_readout_study import readout_features
from src.teacher_gradient_features import full_gradient_features


def test_full_features_match_previous_ntk_gram_and_resume(tmp_path, monkeypatch):
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    model = make_network(x, edges, 'gcn', 4, 2, 10)
    hidden, _, names, biases = readout_features(model, x, edges, 'gcn')
    seeds, projections = (20, 30), 8
    expected = (hidden.double() @ hidden.double().T).numpy() + biases
    for seed in seeds:
        expected += sketch_grams(model, x, edges, torch.arange(3), [projections], seed,
                                  exclude_names=names)[projections] / len(seeds)
    features, config = full_gradient_features(model, x, edges, tmp_path, {'test': 1}, projections, seeds)
    np.testing.assert_allclose((features.double() @ features.double().T).numpy(), expected, rtol=2e-6, atol=2e-6)
    assert features.shape == (3, 4 + 1 + 2 * projections * 2)
    assert all(p.grad is None for p in model.parameters())

    def fail(*args, **kwargs):
        raise RuntimeError('Cache was not reused')

    monkeypatch.setattr('src.teacher_gradient_features.jacobian_features', fail)
    resumed, resumed_config = full_gradient_features(model, x, edges, tmp_path, {'test': 1}, projections, seeds)
    torch.testing.assert_close(resumed, features)
    assert resumed_config == config
