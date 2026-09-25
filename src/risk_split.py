import torch


def split_delta(error, a, b, weight, alpha, beta):
    norm2 = error.square().sum()
    change = (-2 * weight * ((a @ error) * b).sum(-1)
              + weight.square() * a.square().sum(-1) * b.square().sum(-1))
    next_norm = (norm2 + change).clamp_min(0).sqrt()
    denominator = next_norm + norm2.sqrt()
    norm_delta = torch.where(denominator > 0, change / denominator.clamp_min(1e-300), 0.)
    return -alpha * weight * a.square().sum(-1) + beta * norm_delta


@torch.no_grad()
def split_partition(X, Q, m, B, generator, random_directions=2):
    n = len(X)
    alpha, beta = B * B / 4, 2 * B
    energy, moment = X.square().sum(1).mean(), X.T @ Q / n
    cells = {0: torch.arange(n, device=X.device)}
    stats = {0: (X.sum(0), Q.sum(0), n)}
    candidates = {}

    def objective():
        values = list(stats.values())
        sums = torch.stack([v[0] for v in values])
        labels = torch.stack([v[1] for v in values])
        counts = X.new_tensor([v[2] for v in values])
        variance = (energy - (sums.square().sum(1) / counts).sum() / n).clamp_min(0)
        error = moment - (sums / counts[:, None]).T @ labels / n
        return variance, error, alpha * variance + beta * error.norm()

    def proposals(cell):
        ids = cells[cell]
        if len(ids) < 2:
            return []
        x, q = X[ids], Q[ids]
        xc, qc = x - x.mean(0), q - q.mean(0)
        vx, vq = xc.square().sum(), qc.square().sum()
        eta = (vq / vx).sqrt() if float(vx) > 0 and float(vq) > 0 else X.new_tensor(1.)
        wx, wq = (alpha + B * eta).sqrt(), (B / eta).sqrt()
        projections = []
        for data in (xc, qc):
            direction = torch.randn(data.shape[1], generator=generator, device=X.device, dtype=X.dtype)
            for _ in range(8):
                direction = data.T @ (data @ direction)
                direction = direction / direction.norm().clamp_min(1e-30)
            projections.append(data @ direction)
        for _ in range(random_directions):
            dx = torch.randn(X.shape[1], generator=generator, device=X.device, dtype=X.dtype)
            dq = torch.randn(Q.shape[1], generator=generator, device=X.device, dtype=X.dtype)
            projections.append(wx * (xc @ dx) + wq * (qc @ dq))
        result = []
        for projection in projections:
            order = torch.argsort(projection, stable=True)
            cuts = sorted({max(1, min(len(ids) - 1, int(len(ids) * f))) for f in (.25, .5, .75)})
            for cut in cuts:
                left = ids[order[:cut]]
                sx, sq = X[left].sum(0), Q[left].sum(0)
                tx, tq, count = stats[cell]
                a = sx / cut - (tx - sx) / (count - cut)
                b = sq / cut - (tq - sq) / (count - cut)
                result.append(dict(cell=cell, left=left, sx=sx, sq=sq, a=a, b=b,
                                   weight=cut * (count - cut) / (n * count)))
        return result

    variance, error, value = objective()
    history, deltas = [float(value)], []
    candidates[0] = proposals(0)
    for new_cell in range(1, m):
        pool = [p for group in candidates.values() for p in group]
        a, b = torch.stack([p['a'] for p in pool]), torch.stack([p['b'] for p in pool])
        weights = X.new_tensor([p['weight'] for p in pool])
        changes = split_delta(error, a, b, weights, alpha, beta)
        index = int(changes.argmin())
        chosen, predicted = pool[index], float(changes[index])
        parent, left = chosen['cell'], chosen['left']
        marker = torch.zeros(n, dtype=torch.bool, device=X.device)
        marker[left] = True
        right = cells[parent][~marker[cells[parent]]]
        sx, sq, count = stats[parent]
        stats[parent] = (chosen['sx'], chosen['sq'], len(left))
        stats[new_cell] = (sx - chosen['sx'], sq - chosen['sq'], count - len(left))
        cells[parent], cells[new_cell] = left, right
        variance, error, value = objective()
        actual = float(value) - history[-1]
        if abs(actual - predicted) > 1e-8 * max(1., abs(history[-1])):
            raise FloatingPointError('Split delta disagrees with full objective')
        history.append(float(value))
        deltas.append(actual)
        if new_cell + 1 < m:
            candidates[parent], candidates[new_cell] = proposals(parent), proposals(new_cell)
    assignment = torch.empty(n, dtype=torch.long, device=X.device)
    for cell, ids in cells.items():
        assignment[ids] = cell
    return assignment, dict(split_history=history, split_deltas=deltas,
                            forced_splits=sum(v > 0 for v in deltas))
