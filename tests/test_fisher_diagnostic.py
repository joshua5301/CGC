import json
import sys

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

import fisher_diagnostic as diagnostic


def weights():
    # Strictly positive preactivations around the test input: no ReLU crossing.
    return (torch.tensor([[.3, -.2], [.1, .5], [-.4, .2]], dtype=torch.float64),
            torch.tensor([1., 1., 1.], dtype=torch.float64),
            torch.tensor([[.5, -.3, .2], [-.2, .4, -.5]], dtype=torch.float64),
            torch.tensor([.1, -.1], dtype=torch.float64))


def test_quadratic_equals_autograd_input_kl_hessian():
    w = weights()
    x = torch.tensor([[.2, .1]], dtype=torch.float64)
    direction = torch.tensor([[.4, -.3]], dtype=torch.float64)
    p = diagnostic.logits_and_hidden(x, w)[0].softmax(1).detach()

    def kl(flat):
        lq = diagnostic.logits_and_hidden(flat.reshape(1, 2), w)[0].log_softmax(1)
        return (p*(p.log()-lq)).sum()

    hessian = torch.autograd.functional.hessian(kl, x.flatten())
    expected = .5*direction.flatten()@hessian@direction.flatten()
    result = diagnostic.pair_costs(x, x+direction, p, w)
    assert float(result['quadratic'][0]) == pytest.approx(float(expected), abs=1e-12)
    assert float(result['gate_any'][0]) == 0
    assert torch.allclose(result['quadratic'], result['logit_quadratic'], atol=1e-12)


def test_exact_ce_identity_with_nonzero_teacher_mismatch_and_relu_crossing():
    w = weights()
    x = torch.tensor([[.2, .1], [.3, .4]], dtype=torch.float64)
    c = torch.tensor([[-20., 10.], [10., -20.]], dtype=torch.float64)
    teacher = torch.tensor([[.1, .9], [.9, .1]], dtype=torch.float64)
    result = diagnostic.pair_costs(x, c, teacher, w)
    assert result['identity_error'].abs().max() < 1e-12
    assert result['correction'].abs().max() > .01
    assert result['gate_any'].sum() > 0
    assert not torch.allclose(result['quadratic'], result['logit_quadratic'])


def test_local_approximation_converges_and_self_teacher_correction_vanishes():
    w = weights()
    x = torch.tensor([[.2, .1]], dtype=torch.float64)
    p = diagnostic.logits_and_hidden(x, w)[0].softmax(1)
    errors = []
    for step in (.1, .01, .001):
        result = diagnostic.pair_costs(x, x+step*torch.tensor([[1., -1.]]), p, w)
        errors.append(float((result['quadratic']-result['kl']).abs()/result['kl']))
        assert result['correction'].abs().max() < 1e-15
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < .001


def test_ranking_respects_ties_and_zero_kl_summary():
    exact = np.array([[0., 0., 1.], [.1, .5, 2.]])
    approximate = np.array([[2., 1., 3.], [0., 1., 2.]])
    report = diagnostic.ranking_report(exact, approximate)
    assert report['top1_agreement'] == 1
    assert report['excess_kl_mean'] == 0
    x = torch.tensor([[.2, .1]], dtype=torch.float64)
    p = diagnostic.logits_and_hidden(x, weights())[0].softmax(1)
    result = diagnostic.pair_costs(x, x, p, weights())
    stats = diagnostic.summarize({k: v.numpy() for k, v in result.items()})
    assert stats['relative_mae'] is None
    assert stats['spearman'] is None
    json.dumps(stats, allow_nan=False)


def test_small_pipeline_and_cache_resume_without_real_dataset_training(monkeypatch, tmp_path):
    torch.manual_seed(2)
    x = torch.rand(6, 3)
    data = Data(x=x, y=torch.tensor([0, 1, 0, 1, 0, 1]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        train_mask=torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool),
        val_mask=torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.bool),
        test_mask=torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.bool))
    monkeypatch.setattr(diagnostic, 'get_dataset', lambda args: data)
    monkeypatch.setattr(diagnostic, 'conv_graph_multi', lambda args, data: (x, x, x))
    monkeypatch.setattr(diagnostic, 'get_kernel_features', lambda *args: x.double())
    monkeypatch.setattr(diagnostic, 'fit_logistic', lambda *args: torch.tensor(
        [[.3, -.3], [.2, -.2], [-.1, .1]], dtype=torch.float64))

    def partition(z, teacher, *args):
        a = torch.tensor([0, 0, 1, 1, 2, 2])
        return z.reshape(3, 2, 3).mean(1).double(), teacher.reshape(3, 2, 2).mean(1), a

    monkeypatch.setattr(diagnostic, 'partition', partition)
    monkeypatch.setattr(sys, 'argv', ['fisher_diagnostic.py', '--device', 'cpu', '--output', str(tmp_path),
        '--epoch', '2', '--seeds', '23', '--steps', '.01,1', '--anchors', '2', '--batch-size', '3'])
    diagnostic.main()
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert len(summary['probes']) == 4
    assert len(summary['rankings']) == 2
    assert summary['identity_max_error'] < 1e-10
    with np.load(next((tmp_path/'diagnostics').glob('*.npz'))) as values:
        assert values['ranking_kl'].shape == (2, 3)
        assert values['assignment'].shape == (6,)
    cached = [*tmp_path.glob('cache/*.pt'), *tmp_path.glob('models/*.pt'),
              *tmp_path.glob('diagnostics/*.json'), *tmp_path.glob('diagnostics/*.npz')]
    before = {path: path.stat().st_mtime_ns for path in cached}

    def fail(*args, **kwargs):
        raise AssertionError('Cache should have been reused')

    monkeypatch.setattr(diagnostic, 'train_student', fail)
    monkeypatch.setattr(diagnostic, 'fit_logistic', fail)
    monkeypatch.setattr(diagnostic, 'pair_costs', fail)
    diagnostic.main()
    assert all(path.stat().st_mtime_ns == timestamp for path, timestamp in before.items())
