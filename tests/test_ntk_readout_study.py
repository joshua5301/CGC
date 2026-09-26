import numpy as np
import pytest
import torch

from src.empirical_ntk_study import make_network, sketch_grams
from src.ntk_readout_study import readout_features


@pytest.mark.parametrize('architecture', ['gcn', 'sage', 'gin'])
def test_readout_features_equal_exact_readout_jacobian(architecture):
    x = torch.tensor([[1., .2], [.3, 2.], [2., 1.], [.1, .5]])
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    model = make_network(x, edges, architecture, 4, 2, 5000)
    features, output, names, biases = readout_features(model, x, edges, architecture)
    assert features.shape == (4, 8 if architecture == 'sage' else 4)
    parameters = list(model.named_parameters())
    result = model(x, edges)
    readout, internal, full = [torch.zeros(4, 4, dtype=torch.float64) for _ in range(3)]
    for channel in range(2):
        rows = []
        for node in range(4):
            gradients = torch.autograd.grad(result[node, channel], [p for _, p in parameters],
                                            retain_graph=True, allow_unused=True)
            rows.append([torch.zeros_like(p).flatten() if g is None else g.flatten()
                         for g, (_, p) in zip(gradients, parameters)])
        for included, target in ((lambda name: name in names, readout),
                                 (lambda name: name not in names, internal),
                                 (lambda name: True, full)):
            jacobian = torch.stack([torch.cat([g for g, (name, _) in zip(row, parameters) if included(name)])
                                    for row in rows]).double()
            target += jacobian @ jacobian.T / 2
    expected = features.double() @ features.double().T + biases
    np.testing.assert_allclose(readout.numpy(), expected.numpy(), atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(full.numpy(), (readout + internal).numpy(), atol=1e-10)
    np.testing.assert_allclose(result.detach().numpy(), output.numpy(), atol=1e-7)
    masked = sketch_grams(model, x, edges, torch.arange(4), [2], 10,
                          exclude_names=[name for name, _ in parameters])
    np.testing.assert_allclose(masked[2], 0., atol=1e-12)
