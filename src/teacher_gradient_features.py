from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call
from tqdm.auto import tqdm

from src.ntk_readout_study import readout_features
from src.risk_experiment import _fingerprint
from src.tree_distance import _write_json


def jacobian_features(model, x, edges, projections, seed, exclude_names=()):
    names, parameters = zip(*model.named_parameters())
    generator = torch.Generator(device=x.device).manual_seed(seed)

    def output(*values):
        return functional_call(model, dict(zip(names, values)), (x, edges))

    blocks = []
    for _ in tqdm(range(projections), desc=f'Gradient features: seed {seed}', leave=False):
        direction = tuple(torch.randint(0, 2, p.shape, device=p.device, generator=generator).to(p.dtype) * 2 - 1
                          for p in parameters)
        direction = tuple(torch.zeros_like(v) if name in exclude_names else v
                          for name, v in zip(names, direction))
        _, product = torch.autograd.functional.jvp(output, parameters, direction, create_graph=False)
        blocks.append(product.detach())
    return torch.cat(blocks, dim=1) / np.sqrt(projections * blocks[0].shape[1])


def full_gradient_features(model, x, edges, cache_root, identity, projections=512, seeds=(6000, 7000)):
    if projections < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Require positive projections and distinct nonempty sketch seeds')
    model.eval()
    config = dict(version=1, identity=identity, projections=projections, seeds=list(seeds),
                  construction='exact_readout_plus_sketched_internal_trace_NTK',
                  channel_reduction='mean', parameter_metric='native_euclidean')
    folder = Path(cache_root) / _fingerprint(config)
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(folder / 'protocol.json', config)
    hidden, _, names, biases = readout_features(model, x, edges, 'gcn')
    blocks = [hidden.detach(), hidden.new_full((len(x), 1), np.sqrt(biases))]
    for seed in seeds:
        path = folder / f'internal_{seed}.pt'
        if path.exists():
            features = torch.load(path, map_location=x.device, weights_only=True)
        else:
            features = jacobian_features(model, x, edges, projections, seed, names)
            if not bool(torch.isfinite(features).all()):
                raise FloatingPointError('Nonfinite gradient features')
            temporary = path.with_suffix('.tmp')
            torch.save(features.cpu(), temporary)
            temporary.replace(path)
        blocks.append(features / np.sqrt(len(seeds)))
    return torch.cat(blocks, dim=1), config
