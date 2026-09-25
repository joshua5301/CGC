import torch

from src.grip_distance import distance_weights, training_support_distance


def test_inverse_distance_formula_and_zero_power():
    d = torch.tensor([1., 2., 4.], dtype=torch.float64)
    torch.testing.assert_close(distance_weights(d, 1.), torch.tensor([4 / 3, 1., 2 / 3], dtype=torch.float64))
    torch.testing.assert_close(distance_weights(d, 0.), torch.ones_like(d))
    torch.testing.assert_close(distance_weights(d * 100, 1.), distance_weights(d, 1.))


def test_zero_distances_are_finite_and_uniform():
    w = distance_weights(torch.zeros(4), 2.)
    torch.testing.assert_close(w, torch.ones(4, dtype=torch.float64))


def test_distances_exclude_self_and_unlabeled_references():
    x = torch.tensor([[0.], [2.], [5.], [1.]])
    mask = torch.tensor([True, True, True, False])
    distance = training_support_distance(x, mask, 2, block_size=2)
    torch.testing.assert_close(distance, torch.tensor([14.5, 6.5, 17., 1.], dtype=torch.float64))
