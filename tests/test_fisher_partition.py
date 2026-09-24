import numpy as np
import torch

from fisher_diagnostic import pair_costs
from fisher_partition import Geometry, build_factors, nonempty_assignment, refine


def example():
    torch.manual_seed(42)
    x = torch.randn(8, 3, dtype=torch.float64)
    models = [(torch.randn(5, 3, dtype=torch.float64), torch.randn(5, dtype=torch.float64),
               torch.randn(2, 5, dtype=torch.float64), torch.randn(2, dtype=torch.float64)) for _ in range(2)]
    return x, models


def test_low_rank_factors_match_original_diagnostic_quadratic():
    x, models = example()
    b = build_factors(x, models, batch_size=3)
    c = x+.2*torch.randn_like(x)
    actual = .5*torch.einsum('nkd,nd->nk', b, c-x).square().sum(1)
    teacher = torch.ones(8, 2, dtype=torch.float64)/2
    expected = torch.stack([pair_costs(x, c, teacher, w)['quadratic'] for w in models]).mean(0)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-10)


def test_geometry_gradient_and_assignment_cost_match_autograd():
    x, models = example()
    for factors in (None, build_factors(x, models)):
        geometry = Geometry(x, factors, batch_size=3)
        c = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        a = torch.arange(8) % 3
        value = geometry.assigned(c, a)
        value.backward()
        actual, grad = geometry.assigned(c.detach(), a, True)
        assert torch.allclose(c.grad, grad, atol=1e-12)
        assert torch.allclose(value.detach(), actual)
        costs = geometry.all_costs(c.detach())
        assert torch.allclose(costs[torch.arange(8), a].mean(), value.detach(), atol=1e-12)


def test_assignment_keeps_nonempty_cells_and_preserves_ties():
    a = torch.tensor([0, 1, 2, 2])
    costs = torch.tensor([[0., 1., 1.], [0., 1., 1.], [0., 1., 1.], [0., 1., 1.]])
    result = nonempty_assignment(costs, a, 3)
    assert torch.equal(result, torch.tensor([0, 1, 0, 2]))
    assert torch.equal(nonempty_assignment(torch.zeros_like(costs), a, 3), a)


def test_refinement_is_monotone_bounded_and_returns_cell_mean_labels():
    x, models = example()
    teacher = torch.randn(8, 2, dtype=torch.float64).softmax(1)
    a = torch.arange(8) % 3
    c = torch.stack([x[a == j].mean(0) for j in range(3)])
    for factors in (None, build_factors(x, models)):
        result = refine(x, teacher, c, a, factors, coefficient=.3,
                        outer_steps=4, center_steps=20, batch_size=3, log=lambda _: None)
        assert np.all(np.diff([row['objective'] for row in result['history']]) <= 1e-10)
        assert result['x'].norm(dim=1).max() <= x.norm(dim=1).max()+1e-10
        assert (torch.bincount(result['assign'], minlength=3) > 0).all()
        for j in range(3):
            assert torch.allclose(result['y'][j], teacher[result['assign'] == j].mean(0))
