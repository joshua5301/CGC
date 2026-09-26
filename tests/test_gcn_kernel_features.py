import numpy as np
import pytest
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.gcn_kernel_features import gcn_two_layer_kernels, kernel_features, relu_covariance


def test_relu_covariance_known_angles_and_zero():
    covariance = torch.tensor([[1., 0., -1., 0.], [0., 1., 0., 0.],
                               [-1., 0., 1., 0.], [0., 0., 0., 0.]], dtype=torch.float64)
    value, derivative = relu_covariance(covariance)
    np.testing.assert_allclose(value[0].numpy(), [1., 1 / np.pi, 0., 0.], atol=1e-12)
    np.testing.assert_allclose(derivative[0].numpy(), [1., .5, 0., 0.], atol=1e-12)
    np.testing.assert_array_equal(value[3].numpy(), np.zeros(4))


def test_analytic_kernel_matches_finite_network_gradient_quadrature():
    x = torch.tensor([[1., .2], [.1, 2.], [-2., 1.], [0., 0.]], dtype=torch.float64)
    edges = torch.tensor([[0, 1], [1, 0]])
    normalized, weights = gcn_norm(edges, num_nodes=4, dtype=torch.float64)
    s = torch.sparse_coo_tensor(normalized.flip(0), weights, (4, 4)).to_dense()
    width = 8192
    angle = (torch.arange(width, dtype=torch.float64) + .37) * (2 * torch.pi / width)
    w = (np.sqrt(2) * torch.stack([angle.cos(), angle.sin()])).requires_grad_()
    a = torch.ones(width, dtype=torch.float64, requires_grad=True)
    hidden = s @ (np.sqrt(2) * torch.relu((s @ x) @ w / np.sqrt(2)))
    output = hidden @ a / np.sqrt(width)
    jacobian = []
    for scalar in output:
        grads = torch.autograd.grad(scalar, (w, a), retain_graph=True)
        jacobian.append(torch.cat([g.flatten() for g in grads]))
    jacobian = torch.stack(jacobian)
    expected = dict(gcn2_nngp=hidden @ hidden.T / width, gcn2_ntk=jacobian @ jacobian.T)
    actual = gcn_two_layer_kernels(x.numpy(), edges.numpy(), device='cpu')
    for name in actual:
        np.testing.assert_allclose(actual[name].numpy(), expected[name].detach().numpy(), atol=2e-3, rtol=3e-3)


def test_probe_kernel_is_full_kernel_submatrix_and_equivariant():
    x = np.array([[1., 0.], [1., 2.], [0., 3.], [2., -1.]])
    edges = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    ids = np.array([3, 0, 2])
    full = gcn_two_layer_kernels(x, edges, device='cpu')
    subset = gcn_two_layer_kernels(x, edges, ids, device='cpu')
    permutation = np.array([2, 0, 3, 1])
    inverse = permutation.argsort()
    permuted = gcn_two_layer_kernels(x[permutation], inverse[edges], inverse[ids], device='cpu')
    for name in full:
        np.testing.assert_allclose(subset[name], full[name][ids][:, ids], atol=1e-12)
        np.testing.assert_allclose(subset[name], permuted[name], atol=1e-12)
        features, distance, details = kernel_features(subset[name])
        np.testing.assert_allclose(features @ features.T, subset[name], atol=1e-12)
        squared = subset[name].diag()[:, None] + subset[name].diag()[None, :] - 2 * subset[name]
        np.testing.assert_allclose(distance.square(), squared, atol=1e-12)
        assert details['relative_reconstruction_error'] < 1e-12


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Colab CUDA check')
def test_cuda_matches_cpu():
    x = np.array([[1., .3], [.4, 2.], [0., 0.]])
    edges = np.array([[0, 1], [1, 0]])
    cpu = gcn_two_layer_kernels(x, edges, device='cpu')
    gpu = gcn_two_layer_kernels(x, edges, device='cuda')
    for name in cpu:
        np.testing.assert_allclose(cpu[name], gpu[name].cpu(), atol=1e-8, rtol=1e-8)
        _, distance, _ = kernel_features(gpu[name])
        assert torch.isfinite(distance).all()
