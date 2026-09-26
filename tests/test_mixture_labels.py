import pytest
import torch

from src.mixture_label_study import mixture_labels


def test_labels_follow_input_coefficients_within_each_cluster():
    q = torch.tensor([[1., 0.], [0., 1.], [.8, .2], [.2, .8]])
    assignment = torch.tensor([0, 0, 1, 1])
    weights = torch.tensor([.9, .1, .25, .75], dtype=torch.float64)
    result = mixture_labels(q, assignment, weights, 2)
    torch.testing.assert_close(result, torch.tensor([[.9, .1], [.35, .65]]))
    torch.testing.assert_close(result.sum(1), torch.ones(2))
    uniform = mixture_labels(q, assignment, torch.full((4,), .5), 2)
    torch.testing.assert_close(uniform, torch.full((2, 2), .5))


def test_invalid_coefficient_mass_is_rejected():
    with pytest.raises(ValueError):
        mixture_labels(torch.ones(2, 2) / 2, torch.tensor([0, 0]), torch.ones(2), 1)
