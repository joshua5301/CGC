import math

import torch
from torch import nn


def neighborhoods(edge_index, nodes, batch_size=256):
    target, source = edge_index
    keep = target != source
    pairs = torch.unique(target[keep] * nodes + source[keep], sorted=True)
    target, source = pairs // nodes, pairs % nodes
    degree = torch.bincount(target, minlength=nodes)
    starts = degree.cumsum(0) - degree
    groups = []
    for size in degree.unique().tolist():
        if size == 0:
            continue
        ids = (degree == size).nonzero().flatten()
        for chunk in ids.split(batch_size):
            indices = starts[chunk, None] + torch.arange(size, device=degree.device)
            groups.append((chunk, source[indices]))
    return degree, groups


def quantile_coefficients(size, frequencies):
    midpoints = (torch.arange(size, device=frequencies.device, dtype=frequencies.dtype) + .5) / size
    return (2 * (1 + frequencies) / size * torch.sinc(frequencies / size)
            * torch.cos(2 * math.pi * midpoints[:, None] * frequencies))


def aggregate(h, layout, directions, frequencies, block_size=32):
    degree, groups = layout
    outputs = []
    for start in range(0, len(frequencies), block_size):
        freq = frequencies[start:start + block_size]
        projected = h @ directions[:, start:start + block_size]
        output = h.new_zeros(len(h), len(freq))
        coefficients = {}
        for ids, neighbors in groups:
            size = neighbors.shape[1]
            if size not in coefficients:
                coefficients[size] = quantile_coefficients(size, freq)
            values = projected[neighbors].sort(dim=1).values
            output = output.index_copy(0, ids, (values * coefficients[size]).sum(1))
        outputs.append(output)
    return torch.cat(outputs, dim=1)


class FSWEncoder(nn.Module):
    def __init__(self, features, depth=2, width=128, frequency=2., seed=0):
        super().__init__()
        if depth < 1 or width < 2 or frequency <= 0:
            raise ValueError('Require positive depth/frequency and width >= 2')
        self.depth, self.width = depth, width
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer('root_scale', torch.ones(()))
        self.register_buffer('degree_scale', torch.ones(()))
        for layer in range(depth):
            directions = torch.randn(features + layer * (width + 1), width, generator=generator)
            directions = directions / directions.norm(dim=0).clamp_min(1e-12)
            frequencies = torch.rand(width, generator=generator) * frequency
            frequencies[0] = 0
            self.register_buffer(f'directions_{layer}', directions)
            self.register_buffer(f'frequencies_{layer}', frequencies)
            self.register_buffer(f'scale_{layer}', torch.ones(()))

    def forward(self, x, layout, fit=False):
        degree = layout[0].to(x.dtype)
        if fit:
            self.root_scale.copy_(x.square().sum(1).mean().sqrt().clamp_min(1e-12))
            self.degree_scale.copy_(degree.max().clamp_min(1))
        h = x / self.root_scale
        for layer in range(self.depth):
            values = aggregate(h, layout, getattr(self, f'directions_{layer}'),
                               getattr(self, f'frequencies_{layer}'))
            scale = getattr(self, f'scale_{layer}')
            if fit:
                scale.copy_(values.square().sum(1).mean().sqrt().clamp_min(1e-12))
            h = torch.cat((h, values / scale, degree[:, None] / self.degree_scale), dim=1)
        return h

    @torch.no_grad()
    def fit_transform(self, x, layout):
        return self(x, layout, fit=True)


def quotient_edges(edge_index, assignment, nodes, neighbors=8):
    if neighbors < 1:
        raise ValueError('Require at least one quotient neighbor')
    row, col = edge_index
    keep = row != col
    indices = assignment[row[keep]] * nodes + assignment[col[keep]]
    weights = torch.bincount(indices, minlength=nodes * nodes).reshape(nodes, nodes)
    weights = weights + weights.T
    weights.fill_diagonal_(0)
    selected = weights.argsort(dim=1, descending=True, stable=True)[:, :min(neighbors, nodes - 1)]
    mask = torch.zeros_like(weights, dtype=torch.bool)
    mask.scatter_(1, selected, weights.gather(1, selected) > 0)
    return (mask | mask.T).nonzero().T


def normalized_adjacency(edge_index, nodes, dtype=torch.float32):
    adjacency = torch.zeros(nodes, nodes, device=edge_index.device, dtype=dtype)
    adjacency[edge_index[0], edge_index[1]] = 1
    adjacency.fill_diagonal_(1)
    scale = adjacency.sum(1).rsqrt()
    return scale[:, None] * adjacency * scale[None, :]


def realize_graph(x, edge_index, assignment, centers, encoder, neighbors=8,
                  steps=200, lr=.01):
    if steps < 0 or lr <= 0:
        raise ValueError('Require nonnegative realization steps and positive learning rate')
    nodes = len(centers)
    counts = torch.bincount(assignment, minlength=nodes)
    weights = counts.to(x.dtype) / counts.sum()
    means = x.new_zeros(nodes, x.shape[1]).index_add_(0, assignment, x) / counts[:, None]
    edges = quotient_edges(edge_index, assignment, nodes, neighbors)
    layout = neighborhoods(edges, nodes)
    features = nn.Parameter(means.clone())
    optimizer = torch.optim.Adam([features], lr=lr)
    best, best_features, initial = float('inf'), means.clone(), None
    for step in range(steps + 1):
        embedding = encoder(features, layout)
        error = ((embedding - centers).norm(dim=1) * weights).sum()
        value = float(error.detach())
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite FSW realization loss')
        if initial is None:
            initial = value
        if value < best:
            best, best_features = value, features.detach().clone()
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            error.backward()
            optimizer.step()
    with torch.no_grad():
        realized = encoder(best_features, layout)
    return dict(x=best_features.cpu(), adjacency=normalized_adjacency(edges, nodes, x.dtype).cpu(),
                edge_index=edges.cpu(), realized_embedding=realized.cpu(),
                realization_initial=initial, realization_final=best)
