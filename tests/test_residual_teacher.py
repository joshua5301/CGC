import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from src.residual_teacher import split_training_indices, project_simplex, residual_correction


def test_training_split_disjoint_complete_and_reproducible():
    mask = torch.tensor([True]*20+[False]*10)
    fit, cal = split_training_indices(mask, .5, 7)
    assert len(fit) == len(cal) == 10
    assert set(fit.tolist()).isdisjoint(cal.tolist())
    assert set(fit.tolist()+cal.tolist()) == set(range(20))
    fit2, cal2 = split_training_indices(mask, .5, 7)
    assert torch.equal(fit, fit2) and torch.equal(cal, cal2)


def test_euclidean_projection_and_nonexpansiveness():
    x = torch.tensor([[.9, .3, -.2], [2., -1., 0.], [1/3]*3], dtype=torch.float64)
    expected = torch.tensor([[.8, .2, 0.], [1., 0., 0.], [1/3]*3], dtype=torch.float64)
    torch.testing.assert_close(project_simplex(x), expected)
    torch.manual_seed(4)
    x = torch.randn(100, 7, dtype=torch.float64)
    p = torch.randn(100, 7, dtype=torch.float64).softmax(1)
    projected = project_simplex(x)
    assert (projected >= 0).all()
    torch.testing.assert_close(projected.sum(1), torch.ones(100, dtype=torch.float64))
    assert ((projected-p).norm(dim=1) <= (x-p).norm(dim=1)+1e-12).all()


def test_constant_error_is_corrected_in_the_right_direction():
    h = torch.arange(10, dtype=torch.float64)[:, None]
    f = torch.tensor([[.2, .8]]*10, dtype=torch.float64)
    result = residual_correction(h, f, [0, 3, 7], [0, 0, 0], slope=0.)
    torch.testing.assert_close(result['probabilities'], torch.tensor([[1., 0.]]*10, dtype=torch.float64))
    assert (result['chosen_k'] == 3).all()
    assert result['diagnostics']['coverage_certified'] is False


def test_neighbor_choice_and_bound_do_not_use_labels():
    h = torch.arange(10, dtype=torch.float64)[:, None]
    f = torch.tensor([[.2, .8]]*10, dtype=torch.float64)
    a = residual_correction(h, f, [0, 3, 7], [0, 0, 0], slope=4., batch_size=2)
    b = residual_correction(h, f, [0, 3, 7], [1, 1, 1], slope=4., batch_size=7)
    assert torch.equal(a['chosen_k'], b['chosen_k'])
    torch.testing.assert_close(a['conditional_bound'], b['conditional_bound'])
    assert not torch.allclose(a['probabilities'], b['probabilities'])


@pytest.mark.parametrize('slope', [-1., float('nan')])
def test_invalid_slope(slope):
    with pytest.raises(ValueError):
        residual_correction(torch.arange(3.)[:, None], torch.ones(3, 2)/2, [0, 1], [0, 1], slope=slope)
