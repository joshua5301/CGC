import numpy as np
import pytest
import torch

from src.empirical_ntk_study import make_network
from src.trained_teacher_kernel import recover_teacher


def test_teacher_checkpoint_is_verified_without_retraining(tmp_path, monkeypatch):
    x = torch.tensor([[1., .2], [.1, 2.], [1., 1.]])
    edges = torch.tensor([[0, 1], [1, 0]])
    config = dict(hidden=4, seed=0, dropout=.5)
    model = make_network(x, edges, 'gcn', 4, 2, 0, .5)
    with torch.no_grad():
        model.layers[-1].bias.add_(torch.tensor([.5, -.5]))
        expected = model(x, edges).softmax(1).numpy()
    path = tmp_path / 'teacher_state.pt'
    torch.save(model.state_dict(), path)

    def forbidden(*args, **kwargs):
        raise RuntimeError('Checkpoint must be reused')

    monkeypatch.setattr('src.trained_teacher_kernel.fit_gcn_probe_teacher', forbidden)
    recovered, info = recover_teacher(tmp_path, config, x, edges, np.array([0, 1, 0]),
                                      np.array([True, True, False]), np.array([False, False, True]),
                                      expected, device='cpu')
    assert info['source'] == 'checkpoint'
    assert info['probability_max_error'] < 1e-7
    np.testing.assert_allclose(recovered(x, edges).softmax(1).detach().numpy(), expected, atol=1e-7)
    with pytest.raises(ValueError, match='predictions differ'):
        recover_teacher(tmp_path, config, x, edges, np.array([0, 1, 0]),
                        np.array([True, True, False]), np.array([False, False, True]),
                        expected[:, ::-1].copy(), checkpoint=path, device='cpu')
