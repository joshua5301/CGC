import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.partition import EPS, cell_means
from src.risk_analysis import _save_csv
from src.risk_experiment import _fingerprint, _prepare_dataset


@torch.no_grad()
def describe_partition(H, Q, artifact, weight):
    X, Q = H.double(), Q.double().clamp_min(EPS)
    a = artifact['assignment'].to(X.device)
    centers = artifact['x'].to(X.device).double()
    m, N, K = len(centers), len(X), Q.shape[1]
    n = torch.bincount(a, minlength=m).double()
    labels = cell_means(Q, a, m).clamp_min(EPS)
    feature = (X - centers[a]).norm(dim=1) / artifact['dist_scale']
    kl = weight * (Q * (Q.log() - labels[a].log())).sum(1) / artifact['kl_scale']
    cost = feature + kl
    error = abs(float(cost.mean()) - artifact['final_J'])
    if error > 1e-5 * max(1., abs(artifact['final_J'])):
        raise FloatingPointError('Saved partition and reconstructed objective disagree')
    mass = n / N
    cluster_feature = torch.zeros(m, dtype=X.dtype, device=X.device).index_add_(0, a, feature) / N
    cluster_kl = torch.zeros_like(cluster_feature).index_add_(0, a, kl) / N
    hard, rep_class = Q.argmax(1), labels.argmax(1)
    hist = torch.bincount(a * K + hard, minlength=m * K).reshape(m, K).double()
    purity = hist.max(1).values / n
    entropy = -(labels * labels.log()).sum(1)
    small = n <= N / (2 * m)
    large = torch.argsort(n, descending=True)[:max(1, math.ceil(.1 * m))]
    distances = torch.cdist(centers, centers) / artifact['dist_scale']
    distances.fill_diagonal_(torch.inf)
    same = rep_class[:, None] == rep_class[None, :]
    within = distances.masked_fill(~same, torch.inf).min(1).values
    between = distances.masked_fill(same, torch.inf).min(1).values
    valid = torch.isfinite(within) & torch.isfinite(between)
    margin = ((between[valid] - within[valid]) / (between[valid] + within[valid]).clamp_min(EPS)).mean()
    uniform_label_mass, original_label_mass = labels.mean(0), Q.mean(0)
    sorted_n = n.sort().values
    rank = torch.arange(1, m + 1, device=X.device, dtype=X.dtype)
    summary = dict(J=artifact['final_J'], objective_reconstruction_error=error,
        size_gini=float((2 * (rank * sorted_n).sum() / (m * n.sum())) - (m + 1) / m),
        effective_clusters=float(1 / mass.square().sum()),
        size_min=float(n.min()), size_median=float(n.median()), size_max=float(n.max()),
        small_cluster_fraction=float(small.double().mean()), small_node_mass=float(mass[small].sum()),
        mass_uniform_tv=float(.5 * (mass - 1 / m).abs().sum()),
        largest10_node_mass=float(mass[large].sum()),
        largest10_J=float((cluster_feature + cluster_kl)[large].sum()),
        largest10_J_fraction=float((cluster_feature + cluster_kl)[large].sum() / cost.mean().clamp_min(EPS)),
        teacher_purity_mass=float((mass * purity).sum()), teacher_purity_uniform=float(purity.mean()),
        representative_entropy=float(entropy.mean()),
        label_mass_tv=float(.5 * (uniform_label_mass - original_label_mass).abs().sum()),
        representative_nn=float(distances.min(1).values.mean()),
        representative_teacher_margin=float(margin), margin_valid_representatives=int(valid.sum()))
    clusters = pd.DataFrame(dict(cluster=np.arange(m), size=n.cpu().numpy(), mass=mass.cpu().numpy(),
        feature_J=cluster_feature.cpu().numpy(), weighted_kl_J=cluster_kl.cpu().numpy(),
        total_J=(cluster_feature + cluster_kl).cpu().numpy(),
        per_node_cost=((cluster_feature + cluster_kl) / mass).cpu().numpy(),
        teacher_class=rep_class.cpu().numpy(), teacher_purity=purity.cpu().numpy(),
        teacher_entropy=entropy.cpu().numpy(), small=small.cpu().numpy()))
    classes = []
    for c in range(K):
        nodes, reps = hard == c, rep_class == c
        classes.append(dict(teacher_class=c, node_count=int(nodes.sum()), representative_count=int(reps.sum()),
            node_share=float(nodes.double().mean()), representative_share=float(reps.double().mean()),
            soft_node_share=float(original_label_mass[c]), soft_representative_share=float(uniform_label_mass[c]),
            soft_share_shift=float(uniform_label_mass[c] - original_label_mass[c]),
            feature_J=float(feature[nodes].sum() / N), weighted_kl_J=float(kl[nodes].sum() / N),
            total_J=float(cost[nodes].sum() / N),
            per_node_cost=float(cost[nodes].mean()) if bool(nodes.any()) else float('nan')))
    return summary, clusters, pd.DataFrame(classes)


def analyze_grip_seed_structure(source_dir, output_dir, datasets=None, reference_seed=1234,
                                case_dirs=None, device='cuda'):
    root, out, device = Path(source_dir), Path(output_dir), torch.device(device)
    out.mkdir(parents=True, exist_ok=True)
    cases = [Path(p) for p in case_dirs] if case_dirs else sorted(p.parent for p in root.glob('*/protocol.json'))
    selected, seen = [], set()
    for case in cases:
        protocol = json.loads((case / 'protocol.json').read_text(encoding='utf-8'))
        key = (protocol['dataset'], protocol['ratio'])
        if datasets is not None and (key[0] not in datasets or key[1] not in datasets[key[0]]):
            continue
        if key in seen:
            raise ValueError(f'Multiple saved cases for {key}; specify case_dirs explicitly')
        seen.add(key)
        selected.append((case, protocol))
    if not selected:
        raise ValueError('No saved seed-cost cases found')
    expected = {(dataset, ratio) for dataset, ratios in (datasets or {}).items() for ratio in ratios}
    if expected - seen:
        raise ValueError(f'Missing saved cases: {sorted(expected - seen)}')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    summaries, clusters, classes = [], [], []
    for case, protocol in selected:
        dataset, ratio = protocol['dataset'], protocol['ratio']
        train, mask, validation, testing, H = _prepare_dataset(dataset, protocol['data_dir'], device)
        teacher_protocol = {k: protocol[k] for k in ('revision', 'torch', 'dataset', 'data_dir',
            'teacher_seed', 'teacher_kernel', 'gamma', 'T', 'basis')}
        Q = torch.load(root / f'teacher_{_fingerprint(teacher_protocol)}.pt', map_location=device, weights_only=True)
        costs = pd.read_csv(case / 'costs.csv')
        eligible = costs[costs.converged & costs.nodes.eq(costs.requested_nodes)]
        if reference_seed not in eligible.partition_seed.values:
            raise ValueError('Reference partition missing or ineligible')
        for seed in tqdm(eligible.partition_seed.astype(int), desc=f'{dataset} {ratio:g} structure'):
            artifact = torch.load(case / f'{seed}.pt', map_location='cpu', weights_only=True)
            stats, cells, labels = describe_partition(H, Q, artifact, protocol['params']['kl_weight'])
            identity = dict(dataset=dataset, ratio=ratio, partition_seed=seed)
            summaries.append(dict(**identity, **stats))
            clusters.append(cells.assign(**identity))
            classes.append(labels.assign(**identity))
        del train, mask, validation, testing, H, Q, artifact
        torch.cuda.empty_cache()
    runs = pd.DataFrame(summaries)
    comparisons = []
    keys = ['dataset', 'ratio', 'partition_seed']
    for (dataset, ratio), group in runs.groupby(['dataset', 'ratio']):
        reference = group[group.partition_seed == reference_seed].iloc[0]
        others = group[group.partition_seed != reference_seed]
        for metric in runs.columns.difference(keys):
            values = others[metric].dropna()
            value = reference[metric]
            std = values.std()
            comparisons.append(dict(dataset=dataset, ratio=ratio, metric=metric, reference=value,
                other_mean=values.mean(), other_std=std,
                z_score=(value - values.mean()) / std if std > 0 else float('nan'),
                percentile=100 * ((values < value).sum() + .5 * (values == value).sum()) / len(values)
                    if len(values) and pd.notna(value) else float('nan')))
    report = dict(runs=runs, reference_comparison=pd.DataFrame(comparisons),
                  clusters=pd.concat(clusters, ignore_index=True), classes=pd.concat(classes, ignore_index=True))
    size_groups = []
    for identity, cells in report['clusters'].groupby(keys):
        cells = cells.copy()
        cells['size_group'] = pd.cut(cells['size'] / cells['size'].mean(),
            bins=[0, .5, 1, 2, np.inf], labels=['<=0.5x', '0.5-1x', '1-2x', '>2x'])
        grouped = cells.groupby('size_group', observed=False).agg(
            representatives=('cluster', 'size'), node_mass=('mass', 'sum'),
            feature_J=('feature_J', 'sum'), weighted_kl_J=('weighted_kl_J', 'sum'), total_J=('total_J', 'sum'))
        size_groups.append(grouped.reset_index().assign(**dict(zip(keys, identity))))
    report['size_groups'] = pd.concat(size_groups, ignore_index=True)
    (out / 'sources.json').write_text(json.dumps(dict(reference_seed=reference_seed,
        cases=[str(case) for case, _ in selected], label_source='teacher_predictions',
        student_training=False), indent=2), encoding='utf-8')
    for name, frame in report.items():
        _save_csv(frame, out / f'{name}.csv')
    return report
