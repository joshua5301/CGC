import pandas as pd
import torch

from src.initialization_study import selection_plan
from src.partition import partition_cost


def test_cost_components():
    x = torch.tensor([[0.], [2.]], dtype=torch.float64)
    q = torch.tensor([[.5, .5], [.5, .5]], dtype=torch.float64)
    cost = partition_cost(x, q, torch.tensor([0, 0]), x.mean(0, keepdim=True),
                          q.mean(0, keepdim=True), 2., 1., 3.)
    assert cost == dict(feature=.5, kl=0., weighted_kl=0., J=.5)


def test_selection_uses_costs_and_excludes_reference():
    frame = pd.DataFrame(dict(init=['kmeans'] * 4, candidate=['a', 'b', 'c', 'ref'],
        partition_seed=[10, 11, 12, 1234], reference=[False, False, False, True],
        converged=[True] * 4, nodes=[2] * 4, requested_nodes=[2] * 4,
        initial_sse=[3., 1., 2., 0.], final_J=[1., 3., 2., 0.],
        discovery_val=[100., 0., 50., 100.]))
    plan = selection_plan(frame, [1, 3], 3, 0)
    assert 'ref' not in set(plan.candidate)
    assert set(plan[(plan.candidates == 3) & (plan.rule == 'min_sse')].candidate) == {'b'}
    assert set(plan[(plan.candidates == 3) & (plan.rule == 'min_grip')].candidate) == {'a'}
    frame['discovery_val'] = [0., 100., 0., 0.]
    pd.testing.assert_frame_equal(plan, selection_plan(frame, [1, 3], 3, 0))
    assert plan[plan.candidates == 1].groupby('repeat').candidate.nunique().eq(1).all()
