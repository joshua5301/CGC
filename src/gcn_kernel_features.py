import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.local_ce_distance import run_local_ce_comparison, saved_student_fingerprints
from src.node_distances import array_digest
from src.tree_distance import _neighbors, _write_json


def _sparse(matrix, device):
    matrix = matrix.tocoo()
    return torch.sparse_coo_tensor(
        torch.as_tensor(np.stack([matrix.row, matrix.col]), device=device),
        torch.as_tensor(matrix.data, dtype=torch.float64, device=device),
        matrix.shape, device=device).coalesce()


def relu_covariance(covariance):
    scale = covariance.diag().clamp_min(0).sqrt()
    product = scale[:, None] * scale[None, :]
    correlation = (covariance / product.clamp_min(torch.finfo(covariance.dtype).tiny)).clamp(-1, 1)
    correlation = torch.where(1 - correlation.abs() < 8 * torch.finfo(covariance.dtype).eps,
                              correlation.sign(), correlation)
    angle = correlation.acos()
    derivative = (1 - angle / torch.pi) * (product > 0)
    value = product * ((1 - correlation.square()).clamp_min(0).sqrt()
                       + (torch.pi - angle) * correlation) / torch.pi
    return value, derivative


def gcn_two_layer_kernels(x, edge_index, ids=None, device='cuda'):
    x = torch.as_tensor(x, dtype=torch.float64, device=device)
    ids = np.arange(len(x)) if ids is None else np.asarray(ids, dtype=np.int64)
    edges, weights = gcn_norm(torch.as_tensor(edge_index, dtype=torch.long),
                              num_nodes=len(x), add_self_loops=True, dtype=torch.float64)
    propagation = csr_matrix((weights.numpy(), (edges[1].numpy(), edges[0].numpy())),
                             shape=(len(x), len(x)))
    propagated = torch.sparse.mm(_sparse(propagation, device), x) / np.sqrt(x.shape[1])
    covariance = propagated @ propagated.T
    value, derivative = relu_covariance(covariance)
    query = _sparse(propagation[ids], device)

    def sandwich(matrix):
        left = torch.sparse.mm(query, matrix)
        result = torch.sparse.mm(query, left.T).T
        return (result + result.T) / 2

    return dict(gcn2_nngp=sandwich(value),
                gcn2_ntk=sandwich(value + covariance * derivative))


def kernel_features(kernel):
    kernel = (kernel + kernel.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    scale = eigenvalues.abs().max().clamp_min(torch.finfo(kernel.dtype).tiny)
    if eigenvalues.min() < -1e-8 * scale:
        raise ValueError('Kernel has a materially negative eigenvalue')
    features = eigenvectors * eigenvalues.clamp_min(0).sqrt()[None, :]
    distance = torch.cdist(features, features, compute_mode='donot_use_mm_for_euclid_dist')
    distance.fill_diagonal_(0)
    error = torch.linalg.norm(features @ features.T - kernel) / torch.linalg.norm(kernel).clamp_min(1e-300)
    return features, distance, dict(min_eigenvalue=float(eigenvalues.min()),
                                    relative_reconstruction_error=float(error),
                                    feature_width=features.shape[1])


def run_gcn_kernel_study(result_dir, x, edge_index, previous_local_dir=None,
                         models=('gcn', 'sage', 'gin'), device='cuda'):
    result_dir = Path(result_dir)
    source = json.loads((result_dir / 'protocol.json').read_text())
    models, fingerprints = saved_student_fingerprints(source, models)
    graph = source['source_protocol']['distance_protocol']
    _, canonical = _neighbors(edge_index, len(x), graph['self_loops'])
    digest = hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes() + canonical.tobytes()).hexdigest()
    if digest != graph['input_sha256']:
        raise ValueError('Graph/features do not match the original student cache')
    ids = np.load(Path(source['source_dir']) / 'probe_ids.npy')
    distances, previous = {}, None
    fractions, ks, calibration_fraction, split_seed = [.01, .02, .05, .1], [1, 5, 10, 20], .25, 2026
    if previous_local_dir is not None:
        previous_local_dir = Path(previous_local_dir)
        previous = json.loads((previous_local_dir / 'protocol.json').read_text())
        if Path(previous['source_result']).resolve() != result_dir.resolve():
            raise ValueError('Previous local evaluation belongs to another comparison')
        if any(previous['source_logits_sha256'].get(name) != digest for name, digest in fingerprints.items()):
            raise ValueError('Previous evaluation used different students')
        with np.load(previous_local_dir / 'distances.npz') as saved:
            for name, digest in previous.get('extra_sha256', {}).items():
                if name in ('gcn2_nngp', 'gcn2_ntk'):
                    continue
                distance = saved[name].copy()
                if array_digest(distance) != digest:
                    raise ValueError(f'Previous distance changed: {name}')
                distances[name] = distance
        fractions, ks = previous['fractions'], previous['ks']
        calibration_fraction, split_seed = previous['calibration_fraction'], previous['split_seed']
    config = dict(version=1, input_sha256=array_digest(x, edge_index, ids),
                  architecture='S sqrt(2)ReLU(SXW/sqrt(d)) a/sqrt(h)',
                  weights='iid_standard_normal', biases=False, dropout=False,
                  trainable='W_and_a', normalization='PyG_gcn_norm_self_loops',
                  dtype='float64', device=str(device), torch=str(torch.__version__),
                  feature_factorization='full_psd_eigendecomposition_on_probe_nodes')
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    output = result_dir / 'gcn_kernel_features' / key
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / 'protocol.json', config)
    path = output / 'features.npz'
    if not path.exists():
        started = time.perf_counter()
        kernels = gcn_two_layer_kernels(x, edge_index, ids, device)
        artifacts, diagnostics = dict(probe_ids=ids), []
        for name, kernel in kernels.items():
            features, distance, detail = kernel_features(kernel)
            artifacts.update({name + '_kernel': kernel.cpu().numpy(),
                              name + '_features': features.cpu().numpy(),
                              name + '_distance': distance.cpu().numpy()})
            diagnostics.append(dict(method=name, **detail))
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **artifacts)
        pd.DataFrame(diagnostics).to_csv(output / 'diagnostics.csv', index=False)
        _write_json(output / 'timing.json', dict(seconds=time.perf_counter() - started))
        temporary.replace(path)
    with np.load(path) as saved:
        distances.update({name: saved[name + '_distance'].copy() for name in ('gcn2_nngp', 'gcn2_ntk')})
    report = run_local_ce_comparison(
        result_dir, x, edge_index, fractions, ks, calibration_fraction, split_seed,
        models=models, extra_distances=distances,
        extra_protocol=dict(kernel=config, kernel_folder=str(output), previous_local=previous))
    report['kernel_folder'] = str(output)
    report['kernel_diagnostics'] = pd.read_csv(output / 'diagnostics.csv')
    return report
