import json
import sys
from types import SimpleNamespace

import torch
from torch_geometric import seed_everything
from torch_geometric.data import Data

import fisher_diagnostic
import fisher_grip
from coarsening_grip import train_best
from src.models import GCN


def toy():
    torch.manual_seed(4)
    return Data(x=torch.rand(8, 3), y=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]]),
        train_mask=torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool),
        val_mask=torch.tensor([0, 0, 0, 0, 1, 1, 0, 0], dtype=torch.bool),
        test_mask=torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.bool))


def test_two_domain_checkpoints_match_separate_existing_training_runs():
    data = toy()
    zdata = data.clone()
    ids = torch.arange(8)
    zdata.edge_index = torch.stack([ids, ids])
    zdata.edge_attr = torch.ones(8)
    graph = fisher_diagnostic.identity_graph(data.x[:3], torch.tensor([[.8, .2], [.2, .8], [.7, .3]]))
    args = SimpleNamespace(epoch=6, eval_every=2, lr=.01, weight_decay=5e-4)
    seed_everything(9)
    model = GCN(3, 8, 2, 2, .5)
    result = fisher_grip.train_dual(model, graph, dict(graphless=zdata, gcn_transfer=data), 6, 2)
    for domain, target in (('graphless', zdata), ('gcn_transfer', data)):
        seed_everything(9)
        ref = GCN(3, 8, 2, 2, .5)
        val, test, epoch = train_best(ref, args, target, graph)
        assert (val, test, epoch) == (result[domain]['val'], result[domain]['test'], result[domain]['epoch'])


def test_tiny_source_to_refinement_retraining_and_resume(monkeypatch, tmp_path):
    data = toy()
    diag, out = tmp_path/'diagnostic', tmp_path/'refinement'
    monkeypatch.setattr(fisher_diagnostic, 'get_dataset', lambda _: data)
    monkeypatch.setattr(fisher_grip, 'get_dataset', lambda _: data)
    monkeypatch.setattr(fisher_diagnostic, 'conv_graph_multi', lambda args, d: (d.x, d.x, d.x))
    monkeypatch.setattr(fisher_diagnostic, 'get_kernel_features', lambda *args: data.x.double())
    monkeypatch.setattr(fisher_diagnostic, 'fit_logistic', lambda *args: torch.tensor(
        [[.3, -.3], [.2, -.2], [-.1, .1]], dtype=torch.float64))

    def partition(z, teacher, *args):
        a = torch.arange(8) % 3
        return (torch.stack([z[a == j].mean(0).double() for j in range(3)]),
                torch.stack([teacher[a == j].mean(0) for j in range(3)]), a)

    monkeypatch.setattr(fisher_diagnostic, 'partition', partition)
    monkeypatch.setattr(sys, 'argv', ['fisher_diagnostic.py', '--device', 'cpu', '--output', str(diag),
        '--epoch', '2', '--seeds', '23,24', '--steps', '.01,1', '--anchors', '2', '--batch-size', '3'])
    fisher_diagnostic.main()
    monkeypatch.setattr(sys, 'argv', ['fisher_grip.py', '--device', 'cpu', '--diagnostic-dir', str(diag),
        '--output', str(out), '--epoch', '2', '--eval-every', '1', '--seeds', '26,27',
        '--outer-steps', '2', '--center-steps', '3', '--batch-size', '3'])
    fisher_grip.main()
    summary = json.loads((out/'summary.json').read_text())
    assert len(summary['results']) == 6
    assert len(summary['comparisons']) == 4
    assert summary['seeds'] == [26, 27]
    assert summary['provenance']['seeds'] == [23, 24]
    assert len(list((out/'runs').glob('*.json'))) == 6
    assert len(list((out/'models').glob('*.pt'))) == 12
    paths = [*out.glob('condensed/*.pt'), *out.glob('runs/*.json'), *out.glob('models/*.pt')]
    before = {path: path.stat().st_mtime_ns for path in paths}

    def fail(*args, **kwargs):
        raise AssertionError('Completed artifacts should be reused')

    monkeypatch.setattr(fisher_grip, 'refine', fail)
    monkeypatch.setattr(fisher_grip, 'train_dual', fail)
    fisher_grip.main()
    assert all(path.stat().st_mtime_ns == timestamp for path, timestamp in before.items())
