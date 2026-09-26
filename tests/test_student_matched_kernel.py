import numpy as np
import pytest
import torch

from src.empirical_ntk_study import make_network
from src.gnn_distance_probe import ProbeGNN
from src.student_matched_kernel import student_specification


def test_settings_follow_student_protocol_without_width_defaults():
    source = dict(source_protocol=dict(hidden=37, dropout=.4, layers=2))
    spec = student_specification(source, 5)
    assert (spec['hidden'], spec['outputs'], spec['dropout']) == (37, 5, .4)
    assert spec['forward_mode'] == 'eval'
    assert not spec['optimizer_equivalence']


@pytest.mark.parametrize('architecture', ['gcn', 'sage', 'gin'])
def test_factory_matches_original_student_initialization(architecture):
    x = torch.tensor([[1., .1], [.4, 2.], [1., 3.]])
    edges = torch.tensor([[0, 1], [1, 0]])
    with torch.random.fork_rng():
        torch.manual_seed(100)
        original = ProbeGNN(architecture, 2, 5, 3, dropout=.5).eval()
        expected = original(x, edges).detach().numpy()
    before = torch.get_rng_state().clone()
    matched = make_network(x, edges, architecture, 5, 3, 100, dropout=.5)
    assert torch.equal(before, torch.get_rng_state())
    assert matched.dropout == .5 and not matched.training
    np.testing.assert_allclose(matched(x, edges).detach().numpy(), expected, atol=1e-7)
