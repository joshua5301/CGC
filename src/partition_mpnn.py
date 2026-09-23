"""Architecture-independent Q=I surrogate; see docs/mpnn_identity_design.md.

P uniformly selects a closed-neighborhood member; it is NOT the student's
aggregation matrix. Minimize mean (R^K D)[v,a_v] + mu KL, R=(I+P)/2.
Structural deletion/metadata remainders and model-class sensitivity constants
are not estimated here. This is a surrogate, not a computed risk certificate.
"""
import torch
from src.partition_distance import DistanceIdentity


def closed_transition(edge_index, n, device=None):
    """Binary closed neighborhoods, P[target,source]; exactly one self-loop.

    Duplicate input edges are ignored. Isolated nodes have only their self-loop.
    Directed inputs retain their direction (incoming message neighborhoods).
    """
    edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or n < 1:
        raise ValueError('edge_index must be 2 by E and n positive')
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= n):
        raise ValueError('edge index out of range')
    ids = torch.arange(n, device=edge_index.device)
    indices = torch.cat([edge_index.flip(0), torch.stack([ids, ids])], dim=1)
    binary = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1], device=indices.device,
                                                       dtype=torch.float64), (n, n)).coalesce()
    indices = binary.indices()
    degree = torch.bincount(indices[0], minlength=n).to(torch.float64)
    p = torch.sparse_coo_tensor(indices, degree[indices[0]].reciprocal(), (n, n)).coalesce()
    return p, degree


class MPNNIdentity(DistanceIdentity):
    def __init__(self, x, edge_index, teacher, m, assign=None, mu=1., seed=0,
                 batch_size=32, median_iters=30, tol=1e-6, depth=2, log=print):
        if not isinstance(depth, int) or depth < 0:
            raise ValueError('depth must be a nonnegative integer')
        self.depth = depth
        p, degree = closed_transition(edge_index, len(x), x.device)
        ids = torch.arange(len(x), device=x.device)
        identity = torch.sparse_coo_tensor(torch.stack([ids, ids]),
                                          torch.ones(len(x), device=x.device, dtype=torch.float64), p.shape)
        r = (0.5 * p + 0.5 * identity).coalesce()
        super().__init__(x, r, teacher, m, assign, mu, seed, batch_size, median_iters, tol, log)
        # Reuse the descent/median engine, but there is NO GCN mass correction.
        self.q.zero_()
        self.degree = degree

    def propagate(self, values, transpose=False):
        operator = self.At if transpose else self.A
        for _ in range(self.depth):
            values = torch.sparse.mm(operator, values)
        return values

    @torch.no_grad()
    def run(self, outer_iters=20):
        self.log(f'mpnn_identity: depth={self.depth}, rho=0.5, mu={self.mu:g}; '
                 'comparison R=(I+P_closed)/2, uniform student loss; no certified risk value')
        result = super().run(outer_iters)
        result['config'].update(objective='mpnn_identity' if self.depth else 'raw_identity',
                                depth=self.depth, rho=0.5, operator='(I+P_closed)/2',
                                mass_correction=False, risk_certificate=False)
        result['diagnostics'] = dict(mean_closed_degree=float(self.degree.mean()),
                                     max_closed_degree=int(self.degree.max()),
                                     mean_excess_neighbors=float((self.degree - 1).mean()),
                                     imbalance_multiplier=float(self.m * self.counts.max() / self.n))
        return result
