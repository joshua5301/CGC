import torch

from src.grip_bias_study import suppress_class0, select_delta


def test_only_class_zero_can_lose_predictions():
    logits = torch.tensor([[2.1, 2., 0.], [0., 2., 1.], [1., 0., 2.]])
    original = logits.clone()
    assert suppress_class0(logits, 0).tolist() == [0, 1, 2]
    assert suppress_class0(logits, .2).tolist() == [1, 1, 2]
    torch.testing.assert_close(logits, original)


def test_selection_uses_mean_validation_and_smallest_tied_delta():
    data = dict(logits=torch.tensor([[2.1, 2.], [3., 0.]]), target=torch.tensor([1, 0]))
    delta, curve = select_delta([data, data], [0., .2, .5])
    assert delta == .2
    assert curve[0]['validation'] == 50
    assert curve[1]['validation'] == 100
