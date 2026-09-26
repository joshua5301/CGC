import torch
import numpy as np
import pandas as pd
from types import SimpleNamespace

from src.partition import partition
from src.teacher_metric_grip import prepare_metric_teacher, realize_partition


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


def test_teacher_uses_train_and_validation_and_resumes(tmp_path, monkeypatch):
    graph = SimpleNamespace(x=torch.ones(4, 2), edge_index=torch.tensor([[0, 1], [1, 0]]),
                            y=torch.tensor([0, 1, 0, 99]),
                            train_mask=torch.tensor([True, True, False, False]),
                            val_mask=torch.tensor([False, False, True, False]))
    calls = []

    def fit(x, edges, train_ids, train_labels, val_ids, val_labels, **kwargs):
        np.testing.assert_array_equal(train_ids, [0, 1])
        np.testing.assert_array_equal(train_labels, [0, 1])
        np.testing.assert_array_equal(val_ids, [2])
        np.testing.assert_array_equal(val_labels, [0])
        calls.append(1)
        return dict(state_dict={'weight': torch.ones(2)}, probabilities=np.full((4, 2), .5),
                    logits=np.zeros((4, 2)), sweep=pd.DataFrame([{'val_accuracy': 50.}]),
                    config={'model': 'gcn'})

    monkeypatch.setattr('src.teacher_metric_grip.get_dataset', lambda args: graph)
    monkeypatch.setattr('src.probe_teacher.fit_gcn_probe_teacher', fit)
    first = prepare_metric_teacher('arxiv', tmp_path, device='cpu', epochs=2)
    second = prepare_metric_teacher('arxiv', tmp_path, device='cpu', epochs=2)
    assert first == second
    assert len(calls) == 1
    third = prepare_metric_teacher('arxiv', tmp_path, device='cpu', epochs=3)
    assert third['folder'] != first['folder']
    assert len(calls) == 2
