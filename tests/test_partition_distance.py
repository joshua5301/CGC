"""Numerical bounds, independent objective, nonempty descent and batching."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch_geometric.data import Data
from src.partition_distance import DistanceIdentity, weighted_medians
from src.models import GCN
from src.utils import normalize_adj_sparse


def fixture(batch=2):
    torch.manual_seed(42)
    x = torch.randn(12, 4, dtype=torch.float64)
    # Irregular undirected star plus a short path, and isolated nodes.
    edges = torch.tensor([[0, 0, 0, 0, 0, 6, 7], [1, 2, 3, 4, 5, 7, 8]])
    edges = torch.cat([edges, edges.flip(0)], dim=1)
    a = normalize_adj_sparse(Data(x=x, edge_index=edges)).double().coalesce()
    f = torch.softmax(torch.randn(12, 3, dtype=torch.float64), dim=1)
    assign = torch.arange(12) % 3
    obj = DistanceIdentity(x, a, f, 3, assign, batch_size=batch, mu=.3, log=lambda _: None)
    return obj


def test_dense_objective_and_adjoint():
    obj = fixture()
    a = obj.A.to_dense()
    d = torch.linalg.vector_norm(obj.x[:, None] - obj.z[None], dim=2)
    delta = (a.sum(1) - 1).abs()
    cost = a @ a @ d + (a @ delta + delta)[:, None] * obj.z.norm(dim=1)
    torch.testing.assert_close(obj.feature_cost(0, 3), cost)
    s = torch.nn.functional.one_hot(obj.assign, 3).double()
    direct = cost[torch.arange(12), obj.assign].sum()
    adjoint = ((a.T @ a.T @ s) * d).sum() + ((a @ delta + delta)[:, None] * s * obj.z.norm(dim=1)).sum()
    torch.testing.assert_close(direct, adjoint)


def test_nonlinear_gcn_bound_with_bias_and_mass_correction():
    obj = fixture()
    a = obj.A.to_dense()
    delta = (a.sum(1) - 1).abs()
    # Test many weights, with sizable biases, against actual PyG GCN logits.
    for seed in range(12):
        torch.manual_seed(seed)
        model = GCN(4, 7, 3, 2, dropout=0, normalize=False).double().eval()
        for layer in model.layers:
            layer.bias.data.normal_()
        ids = torch.arange(3)
        edges, values = obj.A.indices(), obj.A.values()
        h = torch.relu(model.layers[0](obj.x, edges.flip(0), values))
        hc = torch.relu(model.layers[0](obj.z, torch.stack([ids, ids]), torch.ones(3, dtype=torch.float64)))
        logits = model.layers[1](h, edges.flip(0), values)
        logits_c = model.layers[1](hc, torch.stack([ids, ids]), torch.ones(3, dtype=torch.float64))
        l1 = torch.linalg.matrix_norm(model.layers[0].lin.weight, ord=2)
        l2 = torch.linalg.matrix_norm(model.layers[1].lin.weight, ord=2)
        bound = l1 * l2 * obj.feature_cost(0, 3) + l2 * model.layers[0].bias.norm() * delta[:, None]
        actual = torch.linalg.vector_norm(logits[:, None] - logits_c[None], dim=2)
        assert torch.all(actual <= bound + 1e-9)
        ce = -(obj.f * logits.log_softmax(1)).sum(1).mean()
        mean_entropy = -(obj.f * obj.f.log()).sum(1).mean()
        kl = (obj.f * (obj.f.log() - obj.y[obj.assign].log())).sum(1).mean()
        student_kl = (obj.y * (obj.y.log() - logits_c.log_softmax(1))).sum(1)
        rhs = mean_entropy + np.sqrt(2) * bound[torch.arange(12), obj.assign].mean() + kl + (obj.counts / 12 * student_kl).sum()
        assert ce <= rhs + 1e-9


def test_mass_correction_is_necessary_for_constant_features():
    obj = fixture()
    a = obj.A.to_dense()
    x = torch.ones(12, 1, dtype=torch.float64)
    # With positive identity weights and zero bias, ReLU is inactive.
    assert (a @ a @ x - 1).abs().max() > .1
    delta = (a.sum(1) - 1).abs()
    assert torch.all((a @ a @ x - 1).abs().flatten() <= a @ delta + delta + 1e-12)


def test_precomputed_operator_preserves_baseline_gcn_predictions():
    obj = fixture()
    edges = obj.A.indices()
    # Original input has no self-loops; both paths insert exactly one per node.
    raw_edges = edges[:, edges[0] != edges[1]]
    standard = GCN(4, 7, 3, 2, dropout=0, normalize=True).double().eval()
    explicit = GCN(4, 7, 3, 2, dropout=0, normalize=False).double().eval()
    explicit.load_state_dict(standard.state_dict())
    original = Data(x=obj.x, edge_index=raw_edges, edge_attr=None)
    precomputed = Data(x=obj.x, edge_index=edges, edge_attr=obj.A.values())
    torch.testing.assert_close(standard(original), explicit(precomputed), rtol=1e-6, atol=1e-7)


def test_descent_batching_and_simplex():
    out = fixture(1).run(8)
    other = fixture(3).run(8)
    j = [r['J'] for r in out['history']]
    assert np.all(np.diff(j) <= 1e-8)
    assert out['cell_counts'].min() > 0
    torch.testing.assert_close(out['Y_cond'].sum(1), torch.ones(3))
    torch.testing.assert_close(out['H_cond'], other['H_cond'])
    assert out['history'][-1]['J'] < j[0]


def test_weighted_median_origin_and_coincident_points():
    x = torch.tensor([[2.], [2.], [5.]], dtype=torch.float64)
    w = torch.tensor([[2., 1.], [2., 1.], [1., 1.]], dtype=torch.float64)
    origin = torch.tensor([0., 10.], dtype=torch.float64)
    start = torch.tensor([[2.], [0.]], dtype=torch.float64)
    result = weighted_medians(x, w, origin, start, 50)
    torch.testing.assert_close(result, start)


def test_gpu_matches_cpu_if_available():
    if not torch.cuda.is_available():
        return
    cpu = fixture()
    gpu = DistanceIdentity(cpu.x.cuda(), cpu.A.cuda(), cpu.f.cuda(), 3, cpu.assign.cuda(),
                           batch_size=2, mu=.3, log=lambda _: None)
    a, b = cpu.run(4), gpu.run(4)
    np.testing.assert_allclose(a['history'][-1]['J'], b['history'][-1]['J'], rtol=1e-6)
    torch.testing.assert_close(a['H_cond'], b['H_cond'].cpu(), rtol=1e-5, atol=1e-5)
