import torch

from src.partition import greedy_init


def test_exact_feature_cost_selection():
    x = torch.tensor([[0.], [1.], [4.], [10.]], dtype=torch.float64)
    q = torch.full((4, 2), 0.5, dtype=torch.float64)
    for block_size in (1, 3, 8):
        assignment = greedy_init(x, q, 3, 0., 1., 1., block_size)
        torch.testing.assert_close(assignment, torch.tensor([0, 0, 2, 1]))


def test_teacher_cost_separates_identical_features():
    x = torch.zeros(4, 2, dtype=torch.float64)
    q = torch.tensor([[.9, .1], [.9, .1], [.9, .1], [.1, .9]], dtype=torch.float64)
    assignment = greedy_init(x, q, 2, 1., 1., 1.)
    torch.testing.assert_close(assignment, torch.tensor([0, 0, 0, 1]))


def test_zero_cost_ties_keep_initial_cells_nonempty():
    x = torch.zeros(4, 2, dtype=torch.float64)
    q = torch.full((4, 2), .5, dtype=torch.float64)
    assignment = greedy_init(x, q, 3, 1., 1., 1.)
    torch.testing.assert_close(assignment, torch.tensor([0, 1, 2, 0]))
