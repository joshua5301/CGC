import numpy as np
import torch
from torch import nn

from src.empirical_ntk_study import distance_agreement, exact_trace_gram, gram_distance, sketch_grams


class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 3, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(3, dtype=torch.float64))

    def forward(self, x, edges):
        return x @ self.weight + self.bias


def test_exact_trace_kernel_averages_channels_without_cross_terms():
    x = torch.tensor([[1., 2.], [-1., 3.], [0., 0.]], dtype=torch.float64)
    edges, ids = torch.empty((2, 0), dtype=torch.long), torch.arange(3)
    actual = exact_trace_gram(LinearProbe(), x, edges, ids)
    np.testing.assert_allclose(actual, (x @ x.T + 1).numpy(), atol=1e-12)
    np.testing.assert_allclose(gram_distance(actual), torch.cdist(x, x).numpy(), atol=1e-12)


def test_sketch_matches_explicit_directions_and_nested_prefix():
    x = torch.tensor([[1., 2.], [-1., 3.], [0., 0.]], dtype=torch.float64)
    edges, ids = torch.empty((2, 0), dtype=torch.long), torch.arange(3)
    before = torch.get_rng_state().clone()
    actual = sketch_grams(LinearProbe(), x, edges, ids, [2, 5], 100)
    assert torch.equal(before, torch.get_rng_state())
    generator = torch.Generator().manual_seed(100)
    blocks = []
    for count in range(1, 6):
        weight = torch.randint(0, 2, (2, 3), generator=generator).double() * 2 - 1
        bias = torch.randint(0, 2, (3,), generator=generator).double() * 2 - 1
        blocks.append(x @ weight + bias)
        if count in actual:
            features = torch.cat(blocks, dim=1) / np.sqrt(count * 3)
            np.testing.assert_allclose(actual[count], (features @ features.T).numpy(), atol=1e-12)
    shorter = sketch_grams(LinearProbe(), x, edges, ids, [2], 100)
    np.testing.assert_array_equal(actual[2], shorter[2])


def test_stability_distinguishes_distance_scale_from_neighbor_order():
    reference = np.abs(np.arange(5.)[:, None] - np.arange(5.))
    result = distance_agreement(2 * reference, reference, k=2)
    np.testing.assert_allclose(result['distance_relative_error'], 1.)
    np.testing.assert_allclose(result['distance_spearman'], 1.)
    np.testing.assert_allclose(result['neighbor_overlap'], 1.)
