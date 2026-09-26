import hashlib
import json
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from src.tree_distance import _neighbors, _write_json


def array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def pair_scale(distance):
    pairs = np.asarray(distance)[np.triu_indices(len(distance), 1)]
    median = float(np.median(pairs)) if len(pairs) else 0.
    return median if median > 0 else (float(np.median(pairs[pairs > 0])) if (pairs > 0).any() else 1.)


def transition_neighbors(edge_index, nodes, self_loops=False):
    neighbors, _ = _neighbors(np.asarray(edge_index)[::-1], nodes, self_loops)
    return [ids if len(ids) else np.array([i], dtype=np.int64) for i, ids in enumerate(neighbors)]


def uniform_transport(cost):
    cost = np.ascontiguousarray(cost, dtype=np.float64)
    a, b = cost.shape
    if not a or not b:
        raise ValueError('Probability measures must have nonempty support')
    if a == 1 or b == 1:
        return float(cost.mean())
    if a == b:
        row, col = linear_sum_assignment(cost)
        return float(cost[row, col].mean())
    import ot
    value, log = ot.emd2(np.full(a, 1. / a), np.full(b, 1. / b), cost,
                         numItermax=100000, log=True)
    if log.get('warning') or not np.isfinite(value):
        raise RuntimeError(f'Probability OT did not converge: {log}')
    return max(0., float(value))


def probability_ot_distances(x, edge_index, query_ids, output_dir, depth=2,
                             root_weight=.5, self_loops=False, checkpoint_rows=16):
    x = np.ascontiguousarray(x, dtype=np.float64)
    query_ids = np.asarray(query_ids, dtype=np.int64)
    if x.ndim != 2 or not np.isfinite(x).all() or not len(x):
        raise ValueError('Require finite node features')
    if query_ids.ndim != 1 or len(query_ids) < 2 or len(np.unique(query_ids)) != len(query_ids) or query_ids.min() < 0 or query_ids.max() >= len(x):
        raise ValueError('Require at least two distinct graph node IDs')
    if int(depth) != depth or depth < 0 or not 0 < root_weight < 1 or checkpoint_rows < 1:
        raise ValueError('Invalid probability OT settings')
    depth = int(depth)
    neighbors = transition_neighbors(edge_index, len(x), self_loops)
    _, edges = _neighbors(edge_index, len(x), self_loops)
    protocol = dict(version=1, input_sha256=array_digest(x, edges, query_ids),
                    depth=depth, root_weight=root_weight, self_loops=self_loops,
                    transition='uniform_incoming_neighbors_isolated_self', dtype='float64')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Probability OT cache mismatch; use a different output directory')
    if not path.exists() and any(output.glob('distance_*.npy')):
        raise ValueError('Probability OT cache has no protocol')
    _write_json(path, protocol)
    domains = [None] * (depth + 1)
    domains[depth] = np.sort(query_ids)
    for level in range(depth - 1, -1, -1):
        ids = domains[level + 1]
        domains[level] = np.unique(np.concatenate([ids, *[neighbors[i] for i in ids]]))
    previous, records = None, []
    for level, ids in enumerate(domains):
        size = len(ids)
        distance_path = output / f'distance_{level}.npy'
        state_path = output / f'depth_{level}.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(next_row=0, seconds=0.)
        if not 0 <= state['next_row'] <= size or (state['next_row'] and not distance_path.exists()):
            raise ValueError('Invalid probability OT checkpoint')
        matrix = (np.load(distance_path, mmap_mode='r+') if distance_path.exists() else
                  open_memmap(distance_path, mode='w+', dtype='float64', shape=(size, size)))
        if matrix.shape != (size, size) or matrix.dtype != np.dtype('float64'):
            raise ValueError('Probability OT cache shape or dtype mismatch')
        if level:
            positions = np.searchsorted(domains[level - 1], ids)
            supports = [np.searchsorted(domains[level - 1], neighbors[i]) for i in ids]
        start_row = state['next_row']
        pairs = size * (size - 1) // 2
        initial = start_row * (2 * size - start_row - 1) // 2
        if start_row < size:
            with tqdm(total=pairs, initial=initial, desc=f'Probability OT depth {level}', unit='pairs') as bar:
                for begin in range(start_row, size, checkpoint_rows):
                    started = time.perf_counter()
                    end = min(begin + checkpoint_rows, size)
                    if not level:
                        matrix[begin:end] = cdist(x[ids[begin:end]], x[ids])
                        matrix[np.arange(begin, end), np.arange(begin, end)] = 0.
                    else:
                        for i in range(begin, end):
                            matrix[i, i] = 0.
                            for j in range(i + 1, size):
                                transport = uniform_transport(previous[np.ix_(supports[i], supports[j])])
                                value = root_weight * previous[positions[i], positions[j]] + (1 - root_weight) * transport
                                matrix[i, j] = matrix[j, i] = value
                    if not np.isfinite(matrix[begin:end]).all():
                        raise FloatingPointError('Nonfinite probability OT distances')
                    matrix.flush()
                    state.update(next_row=end, seconds=state['seconds'] + time.perf_counter() - started)
                    _write_json(state_path, state)
                    bar.update(sum(size - i - 1 for i in range(begin, end)))
        records.append(dict(depth=level, nodes=size, pairs=pairs, seconds=state['seconds'],
                            storage_mib=8 * size * size / 2**20))
        del matrix
        previous = np.load(distance_path, mmap_mode='r')
    order = np.searchsorted(domains[-1], query_ids)
    return np.array(previous[np.ix_(order, order)]), records


def multiscale_distance(feature_stages, query_ids):
    blocks, scales = [], []
    for stage in feature_stages:
        block = np.asarray(stage, dtype=np.float64)[query_ids]
        scale = pair_scale(cdist(block, block))
        blocks.append(block / scale)
        scales.append(scale)
    embedding = np.concatenate(blocks, axis=1) / np.sqrt(len(blocks))
    return cdist(embedding, embedding), scales


def neighborhood_mmd_distance(x, edge_index, query_ids, depth=2, width=512,
                              root_weight=.5, seed=2026, self_loops=False):
    if width < 2 or width % 2 or depth < 1 or not 0 < root_weight < 1:
        raise ValueError('Require positive depth, even RFF width, and root weight in (0, 1)')
    z = np.asarray(x, dtype=np.float64)
    neighbors = transition_neighbors(edge_index, len(z), self_loops)
    degree = np.array([len(ids) for ids in neighbors])
    transition = csr_matrix((np.repeat(1. / degree, degree),
                             (np.repeat(np.arange(len(z)), degree), np.concatenate(neighbors))),
                            shape=(len(z), len(z)))
    rng = np.random.default_rng(seed)
    records = []
    for level in range(1, depth + 1):
        scale = pair_scale(cdist(z[query_ids], z[query_ids]))
        omega = rng.normal(size=(z.shape[1], width // 2)) / scale
        projected = z @ omega
        phi = np.concatenate((np.cos(projected), np.sin(projected)), axis=1) / np.sqrt(width // 2)
        mean = transition @ phi
        mean_scale = pair_scale(cdist(mean[query_ids], mean[query_ids]))
        z = np.concatenate((np.sqrt(root_weight) * z / scale,
                            np.sqrt(1 - root_weight) * mean / mean_scale), axis=1)
        records.append(dict(depth=level, bandwidth=scale, mean_scale=mean_scale, width=width))
    return cdist(z[query_ids], z[query_ids]), records


def build_node_distances(x, edge_index, feature_stages, query_ids, output_dir,
                         depth=2, root_weight=.5, self_loops=False, rff_width=512,
                         rff_seed=2026, checkpoint_rows=16):
    if depth < 1 or len(feature_stages) <= max(2, depth):
        raise ValueError('Provide X, SX, ..., S^max(2, depth)X')
    if any(np.shape(stage) != np.shape(x) or not np.isfinite(stage).all() for stage in feature_stages) or not np.array_equal(feature_stages[0], x):
        raise ValueError('Feature stages must start with X and have matching finite shapes')
    start = time.perf_counter()
    matrices = dict(grip_S2X=cdist(feature_stages[2][query_ids], feature_stages[2][query_ids]))
    records = [dict(method='grip_S2X', seconds=time.perf_counter() - start)]
    start = time.perf_counter()
    matrices['multiscale'], scales = multiscale_distance(feature_stages[:depth + 1], query_ids)
    records.append(dict(method='multiscale', seconds=time.perf_counter() - start))
    start = time.perf_counter()
    matrices['neighborhood_mmd'], kernel = neighborhood_mmd_distance(
        x, edge_index, query_ids, depth, rff_width, root_weight, rff_seed, self_loops)
    records.append(dict(method='neighborhood_mmd', seconds=time.perf_counter() - start))
    raw_scale = pair_scale(cdist(np.asarray(x)[query_ids], np.asarray(x)[query_ids]))
    matrices['probability_ot'], transport = probability_ot_distances(
        np.asarray(x, dtype=np.float64) / raw_scale, edge_index, query_ids, Path(output_dir) / 'probability_ot',
        depth, root_weight, self_loops, checkpoint_rows)
    records.append(dict(method='probability_ot', seconds=sum(row['seconds'] for row in transport)))
    matrices = {name: matrices[name] for name in ('grip_S2X', 'multiscale', 'probability_ot', 'neighborhood_mmd')}
    metadata = dict(multiscale_scales=scales, ot_feature_scale=raw_scale,
                    mmd_layers=kernel, ot_layers=transport, timing=records)
    return matrices, metadata
