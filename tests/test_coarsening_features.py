from types import SimpleNamespace
import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from src.coarsening_features import (fixed_coarsening, gcn_operator, optimize_features,
                                    residual_value_grad, feature_diagnostics)
from src.models import GCN
from src.utils import model_training
from coarsening_grip import train_best, measure


def example():
    # Weighted undirected graph, non-unit existing self loop, isolated node.
    ei = torch.tensor([[0, 1, 1, 2, 2, 3, 0], [1, 0, 2, 1, 3, 2, 0]])
    ew = torch.tensor([2., 2., 3., 3., 1., 1., 4.], dtype=torch.float64)
    a = torch.tensor([0, 0, 1, 1, 2])
    ops = fixed_coarsening(ei, ew, 5, a)
    return ei, ew, a, ops


def test_normalization_matches_pyg_with_pooled_loops():
    ei, ew, a, ops = example()
    layer = GCNConv(3, 3, bias=False).double()
    with torch.no_grad():
        layer.lin.weight.copy_(torch.eye(3, dtype=torch.float64))
    X = torch.arange(15, dtype=torch.float64).reshape(5, 3)/7
    C = X[:3]
    assert torch.allclose(layer(X, ei, ew), torch.sparse.mm(ops['P'], X), atol=1e-12)
    assert torch.allclose(layer(C, ops['edge_index'], ops['edge_weight']), ops['Q']@C, atol=1e-12)
    mask = ops['edge_index'][0] == ops['edge_index'][1]
    # Cell 0: original loops 4+1 plus both directions of weight-2 edge = 9.
    assert torch.equal(ops['edge_weight'][mask], torch.tensor([9., 4., 1.], dtype=torch.float64))
    assert torch.equal(ops['counts'], torch.tensor([2, 2, 1]))


def test_analytic_gradient_matches_autograd():
    torch.manual_seed(4)
    C = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    Q = torch.rand(3, 3, dtype=torch.float64)
    H = torch.randn(7, 4, dtype=torch.float64)
    a = torch.arange(7) % 3
    omega = torch.rand(7, dtype=torch.float64)
    ref = (omega*(((Q@C)[a]-H).square().sum(1)+.01**2).sqrt()).mean()
    ref.backward()
    value, gradient, _ = residual_value_grad(C.detach(), Q, H, a, omega, .01)
    assert torch.allclose(ref, value, atol=1e-12)
    assert torch.allclose(C.grad, gradient, atol=1e-12)


def test_convex_fit_certificate_against_known_constrained_medians():
    target = torch.tensor([[-3.], [-2.], [0.], [1.], [3.], [4.]], dtype=torch.float64)
    Q = torch.eye(2, dtype=torch.float64)
    a = torch.tensor([0, 0, 0, 1, 1, 1])
    omega = torch.ones(6, dtype=torch.float64)
    initial = torch.zeros(2, 1, dtype=torch.float64)
    result = optimize_features(target, Q, a, omega, initial, 2.5, steps=400, log=lambda _: None)
    optimum = (target-torch.tensor([[-2.], [2.5]], dtype=torch.float64)[a]).abs().mean().item()
    assert result['final'] <= result['initial']
    assert result['final']-optimum <= result['raw_suboptimality_upper']+1e-10
    assert abs(result['final']-optimum) < .005
    assert result['x'].norm(dim=1).max() <= 2.5+1e-12
    assert np.all(np.diff([r['smooth'] for r in result['history']]) <= 1e-10)


def test_zero_feature_problem():
    result = optimize_features(torch.zeros(3, 2), torch.eye(2), torch.tensor([0, 0, 1]),
        torch.ones(3), torch.zeros(2, 2), 0., log=lambda _: None)
    assert result['final'] == result['raw_suboptimality_upper'] == 0


def test_full_biased_gcn_bound_and_risk_identity():
    ei, ew, a, ops = example()
    torch.manual_seed(7)
    X, C = torch.randn(5, 3), torch.randn(3, 3)
    teacher = torch.rand(5, 2).double()
    teacher /= teacher.sum(1, keepdim=True)
    labels = torch.zeros(3, 2, dtype=torch.float64).index_add_(0, a, teacher)/ops['counts'][:, None]
    data = Data(x=X, edge_index=ei, edge_attr=ew.float())
    graph = Data(x=C, edge_index=ops['edge_index'], edge_attr=ops['edge_weight'].float())
    model = GCN(3, 4, 2, 2, .9)
    with torch.no_grad():
        model.layers[0].bias.copy_(torch.randn(4))
        model.layers[1].bias.copy_(torch.randn(2))
    d = measure(model, data, graph, a, teacher, labels, ops['P'], ops['Q'])
    assert d['logit_gap'] <= d['logit_bound']+1e-5
    assert abs(d['risk_identity_error']) < 1e-6
    assert d['original_teacher_ce'] <= d['teacher_ce_upper']+1e-5
    expected = feature_diagnostics(ops['P'], ops['Q'], X, C, a)
    assert d['D_prop'] == expected['D_prop']


def test_training_protocol_matches_existing_helper():
    ei, ew, a, ops = example()
    seed_everything(2)
    data = Data(x=torch.randn(5, 3), y=torch.zeros(5, dtype=torch.long), edge_index=ei, edge_attr=ew.float(),
        train_mask=torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool),
        val_mask=torch.tensor([0, 0, 1, 1, 0], dtype=torch.bool),
        test_mask=torch.tensor([0, 0, 0, 0, 1], dtype=torch.bool))
    graph = Data(x=torch.randn(3, 3), y=torch.tensor([[.8, .2]]*3), edge_index=ops['edge_index'],
        edge_attr=ops['edge_weight'].float(), train_mask=torch.ones(3, dtype=torch.bool))
    args = SimpleNamespace(lr=.01, weight_decay=5e-4, epoch=6, eval_every=2, dataset_name='cora', device='cpu')
    seed_everything(8)
    model1 = GCN(3, 4, 2, 2, .5)
    val1, test1 = model_training(model1, args, data, graph)
    seed_everything(8)
    model2 = GCN(3, 4, 2, 2, .5)
    val2, test2, _ = train_best(model2, args, data, graph)
    assert (val1, test1) == (val2, test2)
