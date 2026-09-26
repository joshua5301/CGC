import hashlib
import json
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm


def _write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def _neighbors(edge_index, nodes, self_loops):
    edges = np.asarray(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError('Require integer edge_index with shape (2, E)')
    if edges.size and (edges.min() < 0 or edges.max() >= nodes):
        raise ValueError('Edge index outside the feature matrix')
    row, col = edges
    keep = row != col
    pairs = row[keep].astype(np.int64) * nodes + col[keep]
    if self_loops:
        pairs = np.concatenate((pairs, np.arange(nodes) * (nodes + 1)))
    pairs = np.unique(pairs)
    degree = np.bincount(pairs // nodes, minlength=nodes)
    offsets = np.concatenate(([0], np.cumsum(degree)))
    source = pairs % nodes
    return [source[offsets[i]:offsets[i + 1]] for i in range(nodes)], pairs


def _assignment_cost(previous, blank, neighbors, i, j):
    left, right = neighbors[i], neighbors[j]
    a, b = len(left), len(right)
    if not a:
        return float(blank[right].sum(dtype=np.float64))
    if not b:
        return float(blank[left].sum(dtype=np.float64))
    cost = np.array(previous[np.ix_(left, right)], dtype=np.float64)
    baseline = 0.
    if a < b:
        baseline = float(blank[right].sum(dtype=np.float64))
        cost -= blank[right][None, :]
    elif b < a:
        baseline = float(blank[left].sum(dtype=np.float64))
        cost -= blank[left, None]
    row, col = linear_sum_assignment(cost)
    return max(0., baseline + float(cost[row, col].sum()))


def benchmark_tree_pairs(previous, blank, neighbors, samples=5000, seed=0):
    nodes = len(neighbors)
    pairs = nodes * (nodes - 1) // 2
    if samples < 1 or not pairs:
        return {}
    rng = np.random.default_rng(seed)

    def measure(left, right):
        start = time.perf_counter()
        for i, j in zip(left, right):
            _assignment_cost(previous, blank, neighbors, int(i), int(j))
        return (time.perf_counter() - start) / len(left)

    left = rng.integers(nodes, size=samples)
    right = rng.integers(nodes - 1, size=samples)
    right += right >= left
    for i, j in zip(left[:min(32, samples)], right[:min(32, samples)]):
        _assignment_cost(previous, blank, neighbors, int(i), int(j))
    average = measure(left, right)
    degree = np.array([len(v) for v in neighbors])
    high = np.flatnonzero(degree >= np.quantile(degree, .95))
    left = rng.choice(high, size=min(samples, 512))
    right = rng.integers(nodes - 1, size=len(left))
    right += right >= left
    high_average = measure(left, right)
    return dict(benchmark_pairs=samples, mean_pair_us=average * 1e6,
                high_degree_pair_us=high_average * 1e6,
                estimated_full_depth_minutes=pairs * average / 60)


def exact_tree_distances(x, edge_index, output_dir, max_depth=3, weight=1.,
                         dtype='float64', self_loops=False, checkpoint_rows=64,
                         benchmark_pairs=5000, seed=0, on_benchmark=None):
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Require a nonempty finite feature matrix')
    if max_depth < 0 or checkpoint_rows < 1 or benchmark_pairs < 0 or not np.isfinite(weight) or weight <= 0:
        raise ValueError('Invalid depth, checkpoint, benchmark, or weight setting')
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('Require float32 or float64 distance storage')
    nodes = len(x)
    neighbors, edges = _neighbors(edge_index, nodes, self_loops)
    digest = hashlib.sha256()
    digest.update(x.tobytes())
    digest.update(edges.tobytes())
    protocol = dict(version=1, input_sha256=digest.hexdigest(), shape=list(x.shape),
                    weight=float(weight), dtype=dtype.name, self_loops=bool(self_loops))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    protocol_path = directory / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Distance cache belongs to another graph or metric; use a new output directory')
    else:
        if any(directory.glob('distance_*.npy')) or any(directory.glob('depth_*.json')):
            raise ValueError('Distance cache has no protocol; use a new output directory')
        _write_json(protocol_path, protocol)
    root_blank = np.linalg.norm(x, axis=1)
    total_pairs = nodes * (nodes - 1) // 2
    records = []
    base = previous = previous_blank = None
    for depth in range(max_depth + 1):
        path = directory / f'distance_{depth}.npy'
        blank_path = directory / f'blank_{depth}.npy'
        state_path = directory / f'depth_{depth}.json'
        state = (json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists()
                 else dict(depth=depth, next_row=0, seconds=0.))
        start_row = state['next_row']
        if not 0 <= start_row <= nodes:
            raise ValueError('Invalid checkpoint row')
        if start_row and not path.exists():
            raise ValueError('Checkpoint distance file is missing')
        if path.exists():
            matrix = np.load(path, mmap_mode='r+')
            if matrix.shape != (nodes, nodes) or matrix.dtype != dtype:
                raise ValueError('Distance cache shape or dtype mismatch')
        else:
            matrix = open_memmap(path, mode='w+', dtype=dtype, shape=(nodes, nodes))
        blank = (root_blank if depth == 0 else root_blank + weight * np.array(
            [previous_blank[index].sum(dtype=np.float64) for index in neighbors]))
        if not np.isfinite(blank).all():
            raise FloatingPointError('Nonfinite distance to blank tree')
        with blank_path.with_suffix('.tmp').open('wb') as handle:
            np.save(handle, blank.astype(dtype))
        blank_path.with_suffix('.tmp').replace(blank_path)
        if start_row < nodes:
            if depth and benchmark_pairs and 'mean_pair_us' not in state:
                state.update(benchmark_tree_pairs(previous, previous_blank, neighbors, benchmark_pairs, seed + depth))
                _write_json(state_path, state)
                if on_benchmark is not None:
                    on_benchmark(dict(depth=depth, **{k: v for k, v in state.items() if k not in ('depth', 'next_row', 'seconds')}))
            initial = start_row if depth == 0 else start_row * (2 * nodes - start_row - 1) // 2
            with tqdm(total=nodes if depth == 0 else total_pairs, initial=initial,
                      desc=f'Exact tree depth {depth}', unit='rows' if depth == 0 else 'pairs') as progress:
                for begin in range(start_row, nodes, checkpoint_rows):
                    started = time.perf_counter()
                    end = min(begin + checkpoint_rows, nodes)
                    if depth == 0:
                        matrix[begin:end] = cdist(x[begin:end], x, metric='euclidean')
                        matrix[np.arange(begin, end), np.arange(begin, end)] = 0
                        progress.update(end - begin)
                    else:
                        for i in range(begin, end):
                            matrix[i, i] = 0
                            for j in range(i + 1, nodes):
                                value = base[i, j] + weight * _assignment_cost(previous, previous_blank, neighbors, i, j)
                                matrix[i, j] = matrix[j, i] = value
                            progress.update(nodes - i - 1)
                    if not np.isfinite(matrix[begin:end]).all():
                        raise FloatingPointError('Nonfinite tree distances')
                    matrix.flush()
                    state.update(next_row=end, seconds=state['seconds'] + time.perf_counter() - started)
                    _write_json(state_path, state)
        record = dict(depth=depth, nodes=nodes, max_degree=max(map(len, neighbors)),
                      mean_degree=float(np.mean([len(v) for v in neighbors])),
                      storage_mib=nodes * nodes * dtype.itemsize / 2**20,
                      seconds=state['seconds'], path=str(path), blank_path=str(blank_path),
                      **{k: v for k, v in state.items() if k not in ('depth', 'next_row', 'seconds')})
        records.append(record)
        del matrix
        previous = np.load(path, mmap_mode='r')
        previous_blank = np.load(blank_path, mmap_mode='r')
        if depth == 0:
            base = previous
        _write_json(directory / 'summary.json', records)
    return records
