import torch

from src.partition import paired_feature_init, seed_centers


def test_solvers_share_seed_centers():
    x = torch.tensor([[0.], [1.], [3.], [10.], [12.]], dtype=torch.float64)
    for scheme in ('uniform', 'distance'):
        a, first, _ = paired_feature_init(x, 2, 7, f'{scheme}_kmeans')
        b, second, _ = paired_feature_init(x, 2, 7, f'{scheme}_kmedians')
        torch.testing.assert_close(first, second)
        assert len(first.unique()) == 2
        assert len(a.unique()) == len(b.unique()) == 2


def test_duplicate_points_keep_cells_nonempty():
    x = torch.zeros(6, 2, dtype=torch.float64)
    for scheme in ('uniform', 'distance'):
        indices = seed_centers(x, 4, 0, scheme == 'distance')
        assert len(indices.unique()) == 4
        assignment, _, info = paired_feature_init(x, 4, 0, f'{scheme}_kmedians')
        assert len(assignment.unique()) == 4
        assert info['feature_converged']
