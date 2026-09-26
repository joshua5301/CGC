import numpy as np

from src.gnn_distance_diagnostics import decompose_logits


def test_squared_gap_decomposition():
    z = np.array([[1., 2., 3.], [3., 1., 2.], [2., 4., 6.], [0., 0., 0.]])
    gaps, _, _, radial, angular = decompose_logits(z)
    np.testing.assert_allclose(gaps['centered_logits'] ** 2, radial + angular, atol=1e-12)
    assert np.isnan(gaps['direction'][3]).all()


def test_direction_ignores_positive_node_scales_and_common_offsets():
    z = np.array([[1., 2., 4.], [3., 0., 2.], [2., 1., 3.]])
    a = decompose_logits(z)[0]['direction']
    b = decompose_logits(z * np.array([[2.], [4.], [.5]]) + np.array([[9.], [-3.], [2.]]))[0]['direction']
    np.testing.assert_allclose(a, b, atol=1e-12)
