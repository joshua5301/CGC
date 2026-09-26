import torch

from src.partition import partition
from src.teacher_metric_grip import realize_partition


def test_metric_partition_realizes_original_input_dimension():
    metric = torch.tensor([[0., 0.], [0., .1], [4., 4.], [4., 4.1]])
    h = torch.tensor([[1., 0., 2.], [1., 0., 2.], [0., 3., 1.], [0., 3., 1.]])
    q = torch.tensor([[.9, .1], [.7, .3], [.2, .8], [.1, .9]])
    state = partition(metric, q, 2, initial_assignment=torch.tensor([0, 0, 1, 1]),
                      return_diagnostics=True)
    realized = realize_partition(h, state)
    assert realized.shape == (2, 3)
    assert state['x'].shape == (2, 2)
    for j in range(state['nodes']):
        members = state['assignment'] == j
        torch.testing.assert_close(realized[j], h[members][0])
        torch.testing.assert_close(state['y'][j], q[members].mean(0))
        assert int(state['counts'][j]) == int(members.sum())


def test_baseline_realization_matches_grip_centers():
    h = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                      [8., 8., 8.], [8., 9., 8.], [8., 8., 9.]])
    q = torch.tensor([[.9, .1]] * 3 + [[.1, .9]] * 3)
    state = partition(h, q, 2, initial_assignment=torch.tensor([0, 0, 0, 1, 1, 1]),
                      return_diagnostics=True)
    torch.testing.assert_close(realize_partition(h, state), state['x'])
