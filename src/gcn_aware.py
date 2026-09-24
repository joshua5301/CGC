import math
import time

import torch
import torch.nn.functional as F
from torch_geometric import seed_everything

from src.models import GCN
from src.risk_partition import risk_partition


def _simplex(value):
    ordered = value.sort(dim=1, descending=True).values
    levels = (ordered.cumsum(1) - 1) / torch.arange(1, value.shape[1] + 1, device=value.device)
    index = (ordered > levels).sum(1).sub(1)
    return (value - levels.gather(1, index[:, None])).clamp_min(0)


def _build_probes(graph, mask, Q, hidden, seeds, epochs, lr, dropout, ridge, head_steps):
    from src.risk_experiment import _forward

    probes = []
    adjacency = graph['adj'].double()
    propagated = torch.sparse.mm(adjacency, graph['x'].double())
    for seed in seeds:
        seed_everything(seed)
        model = GCN(graph['x'].shape[1], hidden, Q.shape[1], 2, dropout).to(Q.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.nll_loss(_forward(model, graph['x'], graph['adj'])[mask], graph['y'][mask])
            loss.backward()
            optimizer.step()
        w0 = model.layers[0].lin.weight.detach().T.double()
        b0 = model.layers[0].bias.detach().double()
        with torch.no_grad():
            z = torch.sparse.mm(adjacency, F.relu(propagated @ w0 + b0))
            z = torch.cat((z, z.new_ones(len(z), 1)), dim=1)
        head = torch.cat((model.layers[1].lin.weight.detach().T.double(),
                          model.layers[1].bias.detach().double()[None, :]), dim=0).requires_grad_()
        optimizer = torch.optim.LBFGS([head], max_iter=head_steps, tolerance_grad=1e-9,
                                      tolerance_change=1e-12, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = -(Q * F.log_softmax(z @ head, dim=1)).sum(1).mean() + ridge / 2 * head.square().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            head = head.detach()
            target = z.T @ (F.softmax(z @ head, dim=1) - Q) / len(z)
            probes.append(dict(w0=w0, b0=b0, head=head, target=target, seed=int(seed),
                               head_stationarity=float((target + ridge * head).norm())))
        del model, optimizer, head, z
    return probes


def _representations(C, probes):
    features, probabilities = [], []
    for probe in probes:
        z = F.relu(C @ probe['w0'] + probe['b0'])
        z = torch.cat((z, z.new_ones(len(z), 1)), dim=1)
        features.append(z)
        probabilities.append(F.softmax(z @ probe['head'], dim=1))
    return torch.stack(features), torch.stack(probabilities)


def _score(C, Y, probes):
    z, p = _representations(C, probes)
    targets = torch.stack([probe['target'] for probe in probes])
    residual = z.transpose(1, 2) @ (p - Y[None, :, :]) / len(C) - targets
    return residual.square().sum((1, 2)).mean()


@torch.no_grad()
def _calibrate(C, Y, probes, max_steps, tolerance):
    z, p = _representations(C, probes)
    m, repeats = len(C), len(probes)
    target = z.transpose(1, 2) @ p / m - torch.stack([probe['target'] for probe in probes])
    gram = (z @ z.transpose(1, 2)).sum(0)
    lipschitz = (2 * torch.linalg.eigvalsh(gram)[-1] / (repeats * m * m)).clamp_min(1e-15)
    Y = _simplex(Y)
    for step in range(max_steps + 1):
        residual = z.transpose(1, 2) @ Y.expand(repeats, -1, -1) / m - target
        gradient = 2 * (z @ residual).mean(0) / m
        gap = float(((gradient * Y).sum(1) - gradient.min(1).values).sum().clamp_min(0))
        if gap <= tolerance or step == max_steps:
            break
        Y = _simplex(Y - gradient / lipschitz)
    return Y, dict(steps=step, gap=gap, converged=gap <= tolerance,
                    objective=float(residual.square().sum((1, 2)).mean()))


def gcn_aware_partition(H, Q, m, B, graph, train_mask, hidden=256, seed=0,
                        max_sweeps=5, block_size=1024, atol=1e-12, rtol=1e-10,
                        init_sweeps=30, probe_seeds=(1000, 1001), probe_epochs=100,
                        probe_lr=0.01, probe_dropout=0.5, head_ridge=0.001,
                        head_steps=200, label_steps=1000, label_tolerance=1e-8,
                        proposal_nodes=8192):
    if min(probe_epochs, head_steps, label_steps, proposal_nodes, block_size) < 1 or not probe_seeds:
        raise ValueError('Require positive iteration limits and nonempty probe seeds')
    if min(max_sweeps, init_sweeps, atol, rtol, label_tolerance) < 0 or head_ridge <= 0:
        raise ValueError('Invalid GCN-aware optimization settings')
    if not 0 <= probe_dropout < 1 or probe_lr <= 0:
        raise ValueError('Invalid probe optimizer settings')
    if H.is_cuda:
        torch.cuda.synchronize(H.device)
    started = time.perf_counter()
    initial = risk_partition(H, Q, m, B, seed=seed, max_sweeps=init_sweeps,
                             block_size=block_size, atol=atol, rtol=rtol, return_assignment=True)
    X, Q = H.detach().double(), Q.detach().double()
    assignment = initial['assignment'].to(H.device).clone()
    counts = torch.bincount(assignment, minlength=m).double()
    sums = X.new_zeros(m, X.shape[1]).index_add_(0, assignment, X)
    C = sums / counts[:, None]
    Y = Q.new_zeros(m, Q.shape[1]).index_add_(0, assignment, Q) / counts[:, None]
    Y = _simplex(Y)
    probes = _build_probes(graph, train_mask, Q, hidden, probe_seeds, probe_epochs,
                           probe_lr, probe_dropout, head_ridge, head_steps)
    with torch.no_grad():
        objective = float(_score(C, Y, probes))
    if not math.isfinite(objective):
        raise FloatingPointError('Nonfinite initial gradient mismatch')
    history = [objective]
    calibrations, move_history = [], []

    def calibrate():
        nonlocal Y, objective
        labels, diagnostic = _calibrate(C, Y, probes, label_steps, label_tolerance)
        value = float(_score(C, labels, probes))
        if not math.isfinite(value) or value > objective + 1e-9 * max(1.0, abs(objective)):
            raise FloatingPointError('Label calibration increased gradient mismatch')
        if value <= objective:
            Y, objective = labels, value
        else:
            diagnostic['converged'] = False
        calibrations.append(diagnostic)
        history.append(objective)

    @torch.no_grad()
    def apply_moves(ids, destinations):
        nonlocal counts, sums, C, objective
        sources = assignment[ids]
        next_counts = counts + torch.bincount(destinations, minlength=m) - torch.bincount(sources, minlength=m)
        if bool((next_counts > 0).all()):
            next_sums = sums.clone()
            next_sums.index_add_(0, sources, -X[ids])
            next_sums.index_add_(0, destinations, X[ids])
            centers = next_sums / next_counts[:, None]
            value = float(_score(centers, Y, probes))
            if not math.isfinite(value):
                raise FloatingPointError('Nonfinite candidate gradient mismatch')
            if value < objective - (atol + rtol * max(1.0, abs(objective))):
                counts, sums, C, objective = next_counts, next_sums, centers, value
                assignment[ids] = destinations
                return len(ids)
        if len(ids) == 1:
            return 0
        middle = len(ids) // 2
        return apply_moves(ids[:middle], destinations[:middle]) + apply_moves(ids[middle:], destinations[middle:])

    calibrate()
    generator = torch.Generator(device=H.device).manual_seed(seed)
    status = 'iteration_limit'
    for _ in range(max_sweeps):
        moved = 0
        selected = torch.randperm(len(X), generator=generator, device=H.device)[:proposal_nodes]
        for ids in selected.split(block_size):
            centers = C.detach().requires_grad_()
            gradient, = torch.autograd.grad(_score(centers, Y, probes), centers)
            with torch.no_grad():
                sources = assignment[ids]
                removal = ((C[sources] - X[ids]) * gradient[sources]).sum(1) / (counts[sources] - 1).clamp_min(1)
                addition = (X[ids] @ gradient.T - (C * gradient).sum(1)[None, :]) / (counts + 1)[None, :]
                delta = removal[:, None] + addition
                delta.scatter_(1, sources[:, None], torch.inf)
                delta[counts[sources] <= 1] = torch.inf
                best_delta, destinations = delta.min(1)
                take = best_delta < 0
                if bool(take.any()):
                    moved += apply_moves(ids[take], destinations[take])
        history.append(objective)
        before_calibration = objective
        calibrate()
        move_history.append(moved)
        if moved == 0 and before_calibration - objective <= atol + rtol * max(1.0, abs(objective)):
            status = 'proposal_stalled'
            break
    with torch.no_grad():
        cx, cy = C.float(), Y.float()
        serialized_score = float(_score(cx.double(), cy.double(), probes))
        label_residual = float((Y.sum(1) - 1).abs().max())
        if not bool(torch.isfinite(Y).all()) or float(Y.min()) < -1e-10 or label_residual > 1e-8:
            raise FloatingPointError('Calibrated labels violate the simplex')
    result = dict(x=cx.cpu(), y=cy.cpu(), counts=counts.long().cpu(), assignment=assignment.cpu(),
                  J=objective, history=history, objective_name='mean_squared_head_gradient_gap',
                  sweeps=len(move_history), moves=move_history, converged=False, status=status,
                  initial_risk_J=initial['J'], label_gap=calibrations[-1]['gap'],
                  label_converged=calibrations[-1]['converged'], label_residual=label_residual,
                  calibrations=calibrations, serialized_score=serialized_score,
                  head_stationarity_max=max(probe['head_stationarity'] for probe in probes),
                  probe_states=[{k: v.cpu() if torch.is_tensor(v) else v for k, v in probe.items()}
                                for probe in probes], B=float(B), seed=int(seed))
    result['seconds'] = time.perf_counter() - started
    return result
