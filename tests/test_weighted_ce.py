import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from src.utils import soft_label_ce


def test_cell_mass_matches_expanded_original_nodes_and_gradients():
    logits = torch.tensor([[2., -1.], [-.4, .8], [.2, .1]], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[.8, .2], [.1, .9], [.5, .5]], dtype=torch.float64)
    counts = torch.tensor([1, 7, 2])
    assignment = torch.repeat_interleave(torch.arange(3), counts)
    logp = logits.log_softmax(1)
    actual = soft_label_ce(logp, targets, counts)
    expanded = -(targets[assignment] * logp[assignment]).sum(1).mean()
    torch.testing.assert_close(actual, expanded)
    g_actual = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    g_expanded = torch.autograd.grad(expanded, logits, retain_graph=True)[0]
    torch.testing.assert_close(g_actual, g_expanded)
    torch.testing.assert_close(actual, soft_label_ce(logp, targets, 10 * counts))
    torch.testing.assert_close(soft_label_ce(logp, targets), soft_label_ce(logp, targets, torch.ones(3)))
    assert not torch.isclose(actual, soft_label_ce(logp, targets))


@pytest.mark.parametrize('weights', [[0., 0.], [-1., 2.], [float('nan'), 1.], [1.]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        soft_label_ce(torch.zeros(2, 2), torch.ones(2, 2) / 2, torch.tensor(weights))
