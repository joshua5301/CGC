import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric
from tqdm.auto import tqdm

from src.empirical_ntk_study import _save, gram_distance, make_network, sketch_grams
from src.local_ce_distance import run_local_ce_comparison, saved_student_fingerprints
from src.node_distances import array_digest
from src.ntk_readout_study import readout_features
from src.tree_distance import _neighbors, _write_json


def student_specification(source, output_width):
    original = source['source_protocol']
    if original['layers'] != 2:
        raise ValueError('Current ProbeGNN factory supports two-layer students only')
    return dict(hidden=int(original['hidden']), outputs=int(output_width), dropout=float(original['dropout']),
                layers=int(original['layers']), forward_mode='eval', factory='src.gnn_distance_probe.ProbeGNN',
                parameter_metric='native_euclidean', kernel_state='initialization',
                initialization='PyG_defaults', optimizer_equivalence=False)


def run_student_matched_kernel(previous_local_dir, x, edge_index,
                                network_seeds=tuple(range(5000, 5008)), projections=512,
                                sketch_seeds=(6000, 7000), device='cuda'):
    if projections < 1 or not network_seeds or not sketch_seeds or len(set(network_seeds)) != len(network_seeds):
        raise ValueError('Require positive projection count and distinct nonempty network seeds')
    previous = json.loads((Path(previous_local_dir) / 'protocol.json').read_text())
    result_dir = Path(previous['source_result'])
    source = json.loads((result_dir / 'protocol.json').read_text())
    models, fingerprints = saved_student_fingerprints(source, previous['models'])
    if fingerprints != previous['source_logits_sha256']:
        raise ValueError('Original frozen students changed')
    original = source['source_protocol']
    if set(network_seeds) & set(original['model_seeds']):
        raise ValueError('Use independent distance initialization seeds')
    run = Path(source['source_dir'])
    ids = np.load(run / 'probe_ids.npy')
    with np.load(run / 'teacher_predictions.npz') as saved:
        classes = saved['probabilities'].shape[1]
    spec = student_specification(source, classes)
    graph = original['distance_protocol']
    _, canonical = _neighbors(edge_index, len(x), graph['self_loops'])
    digest = hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes() + canonical.tobytes()).hexdigest()
    if digest != graph['input_sha256']:
        raise ValueError('Original graph/features changed')
    config = dict(version=1, student=spec, models=models, source_protocol=original,
                  input_sha256=array_digest(x, edge_index, ids), network_seeds=list(network_seeds),
                  projections=projections, sketch_seeds=list(sketch_seeds), device=str(device),
                  torch=str(torch.__version__), pyg=str(torch_geometric.__version__), tf32=False,
                  dtype='float32_derivatives_float64_grams')
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    folder = result_dir / 'student_matched_kernel' / key
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(folder / 'protocol.json', config)
    tx = torch.as_tensor(x, dtype=torch.float32, device=device)
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    queries = torch.as_tensor(ids, dtype=torch.long, device=device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checks, metadata, distances = [], [], {}
    with np.load(Path(previous_local_dir) / 'distances.npz') as saved:
        for name, digest in previous.get('extra_sha256', {}).items():
            if name.startswith('matchedstudent_'):
                continue
            value = saved[name].copy()
            if array_digest(value) != digest:
                raise ValueError(f'Previous distance changed: {name}')
            distances[name] = value
    for architecture in models:
        filenames = [name for name in fingerprints if name.startswith(architecture + '_')]
        for seed in original['model_seeds']:
            model = make_network(tx, edges, architecture, spec['hidden'], classes, seed, spec['dropout'])
            with torch.no_grad():
                initial = model(tx, edges)[queries].cpu().numpy()
            for name in filenames:
                if not name.endswith(f'_seed{seed}.npz'):
                    continue
                with np.load(run / name) as saved:
                    expected = saved['initial']
                    if expected.shape != initial.shape or not np.allclose(expected, initial, rtol=1e-5, atol=2e-6):
                        raise ValueError(f'Student initialization cannot be reproduced: {name}')
                    checks.append(dict(model=architecture, student_file=name,
                                       initial_max_error=float(np.abs(expected - initial).max())))
        totals = {name: np.zeros((len(ids), len(ids))) for name in ('hidden', 'readout', 'internal', 'full', 'logits')}
        for seed in tqdm(network_seeds, desc=f'Student-matched {architecture}'):
            path = folder / f'{architecture}_seed{seed}.npz'
            if not path.exists():
                model = make_network(tx, edges, architecture, spec['hidden'], classes, seed, spec['dropout'])
                features, logits, names, biases = readout_features(model, tx, edges, architecture)
                features, logits = features[queries].double(), logits[queries].double()
                hidden = (features @ features.T).cpu().numpy()
                internal = np.zeros_like(hidden)
                for replica in sketch_seeds:
                    gram = sketch_grams(model, tx, edges, queries, [projections],
                                        replica + 100003 * seed, exclude_names=names)[projections]
                    internal += gram / len(sketch_seeds)
                _save(path, hidden=hidden, readout=hidden + biases, internal=internal,
                      full=hidden + biases + internal, logits=(logits @ logits.T / classes).cpu().numpy(),
                      feature_width=features.shape[1])
            with np.load(path) as saved:
                for family in totals:
                    totals[family] += saved[family] / len(network_seeds)
                feature_width = int(saved['feature_width'])
        for family, gram in totals.items():
            name = f'matchedstudent_{architecture}_{family}'
            distances[name] = gram_distance(gram)
            metadata.append(dict(method=name, architecture=architecture, width=spec['hidden'],
                                 outputs=classes, feature_width=feature_width, family=family))
    report = run_local_ce_comparison(result_dir, x, edge_index, previous['fractions'], previous['ks'],
                                     previous['calibration_fraction'], previous['split_seed'], models=models,
                                     extra_distances=distances, extra_protocol=dict(matched=config, folder=str(folder)))
    report.update(student=spec, metadata=pd.DataFrame(metadata), checks=pd.DataFrame(checks))
    report['detail'] = report['summary'].merge(report['metadata'], on='method', how='inner')
    for name in ('metadata', 'checks', 'detail'):
        report[name].to_csv(Path(report['folder']) / f'{name}.csv', index=False)
    return report
