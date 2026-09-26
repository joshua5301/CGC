import numpy as np

from src.probe_teacher import fit_gcn_probe_teacher


def test_gcn_teacher_returns_validation_selected_soft_targets():
    x = np.array([[1., 0.], [0., 1.], [1., 1.], [.2, .4]], dtype=np.float32)
    edges = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    result = fit_gcn_probe_teacher(x, edges, [0, 1], [0, 1], [2, 3], [0, 1],
                                   hidden=4, epochs=3, eval_every=2, device='cpu')
    best = result['sweep'].sort_values(['val_accuracy', 'val_ce', 'epoch'], ascending=[False, True, True]).iloc[0]
    assert result['config']['best_epoch'] == best.epoch
    assert result['probabilities'].shape == (4, 2)
    np.testing.assert_allclose(result['probabilities'].sum(1), 1., atol=1e-6)
    assert result['state_dict']
