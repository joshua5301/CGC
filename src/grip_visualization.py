import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import hsv_to_rgb
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch_geometric import seed_everything

from src.grip_reliability import save_json, save_tensor
from src.hyperparams import BEST_HYPERPARAMS_DICT
from src.partition import partition
from src.risk_experiment import _fingerprint, _prepare_dataset
from src.teacher import get_teacher_labels
from src.utils import BUDGET


def align_labels(reference, candidate):
    ref_ids, ref = np.unique(reference, return_inverse=True)
    candidate_ids, other = np.unique(candidate, return_inverse=True)
    overlap = np.zeros((len(ref_ids), len(candidate_ids)), dtype=np.int64)
    np.add.at(overlap, (ref, other), 1)
    rows, columns = linear_sum_assignment(-overlap)
    mapping = {int(candidate_ids[c]): int(ref_ids[r]) for r, c in zip(rows, columns)}
    next_id = int(ref_ids.max()) + 1
    for c in candidate_ids:
        if int(c) not in mapping:
            mapping[int(c)] = next_id
            next_id += 1
    return np.array([mapping[int(c)] for c in candidate]), mapping


def compare_partitions(first, second):
    aligned, _ = align_labels(first, second)
    return dict(ARI=adjusted_rand_score(first, second),
                NMI=normalized_mutual_info_score(first, second),
                reassigned_percent=100 * float(np.mean(first != aligned)))


def render_case(xy, q, panels, root, title):
    aligned = {}
    seeds = list(dict.fromkeys(seed for seed, stage in panels))
    anchor = panels[(seeds[0], 'initial')]['assignment']
    for seed in seeds:
        initial = panels[(seed, 'initial')]['assignment']
        aligned[(seed, 'initial')] = align_labels(anchor, initial)[0]
        aligned[(seed, 'final')] = align_labels(aligned[(seed, 'initial')], panels[(seed, 'final')]['assignment'])[0]
    count = 1 + max(int(a.max()) for a in aligned.values())
    palette = hsv_to_rgb(np.column_stack(((np.arange(count) * .61803398875) % 1,
                                          np.full(count, .65), np.full(count, .85))))
    class_palette = plt.get_cmap('tab10')(np.arange(q.shape[1]) % 10)
    figures, cluster_rows = {}, []
    for kind in ('clusters', 'representatives'):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
        for row, seed in enumerate(seeds):
            for col, stage in enumerate(('initial', 'final')):
                ax = axes[row, col]
                panel = panels[(seed, stage)]
                assignment = panel['assignment']
                colors = palette[aligned[(seed, stage)]] if kind == 'clusters' else class_palette[q.argmax(1)]
                ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=6, alpha=.8 if kind == 'clusters' else .12,
                           linewidths=0, rasterized=True)
                if kind == 'representatives':
                    for cluster in np.unique(assignment):
                        mask = assignment == cluster
                        location = xy[mask].mean(0)
                        soft = q[mask].mean(0)
                        label = int(soft.argmax())
                        ax.scatter(*location, s=25 + 8 * np.sqrt(mask.sum()),
                                   color=class_palette[label], edgecolors='black', linewidths=.6)
                        cluster_rows.append(dict(seed=seed, stage=stage, cluster=int(cluster),
                            nodes=int(mask.sum()), teacher_class=label, x=float(location[0]), y=float(location[1]),
                            **{f'prob_{c}': float(p) for c, p in enumerate(soft)}))
                ax.set_title(f'Seed {seed} | {stage} | clusters={len(np.unique(assignment))} | J={panel["J"]:.5f}')
                ax.set_xticks([])
                ax.set_yticks([])
        subtitle = 'Matched cluster colors' if kind == 'clusters' else 'Teacher-class colors; marker = member mean in 2D; size = cluster size'
        fig.suptitle(f'{title}\n{subtitle}')
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(root / f'{kind}.png', dpi=180)
        fig.savefig(root / f'{kind}.pdf', dpi=180)
        figures[kind] = fig
    clusters = pd.DataFrame(cluster_rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for row, seed in enumerate(seeds):
        for col, stage in enumerate(('initial', 'final')):
            subset = clusters[(clusters.seed == seed) & (clusters.stage == stage)]
            counts = subset.teacher_class.value_counts().reindex(range(q.shape[1]), fill_value=0)
            axes[row, col].bar(counts.index, counts.values, color=class_palette)
            axes[row, col].set_title(f'Seed {seed} | {stage}')
            axes[row, col].set_xlabel('Teacher class of mean soft label')
            axes[row, col].set_ylabel('Representative count')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(root / 'class_allocation.png', dpi=180)
    fig.savefig(root / 'class_allocation.pdf')
    figures['class_allocation'] = fig
    clusters.to_csv(root / 'clusters.csv', index=False)
    return figures


def visualize_grip(datasets, output_dir, configs=None, partition_seeds=(0, 1234),
                   teacher_seed=0, tsne_seed=42, perplexity=30, pca_dim=50,
                   tsne_steps=1500, grip_steps=1000, data_dir='/content/data/', device='cuda'):
    if len(partition_seeds) != 2 or len(set(partition_seeds)) != 2:
        raise ValueError('Specify two distinct partition seeds')
    root, device = Path(output_dir), torch.device(device)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    protocol = dict(datasets=datasets, configs={f'{d}:{r}': v for (d, r), v in (configs or {}).items()},
        partition_seeds=list(partition_seeds), teacher_seed=teacher_seed, tsne_seed=tsne_seed,
        perplexity=perplexity, pca_dim=pca_dim, tsne_steps=tsne_steps,
        grip_steps=grip_steps, data_dir=str(data_dir), revision=revision)
    root = root / _fingerprint(protocol)
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / 'protocol.json', protocol)
    figures, rows = {}, []
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for dataset, ratios in datasets.items():
        train, mask, validation, testing, h = _prepare_dataset(dataset, data_dir, device)
        if len(h) > 15000:
            raise ValueError('This full-node visualization targets small graphs (<=15000 nodes)')
        xy_path = root / f'{dataset}_tsne.npy'
        if xy_path.exists():
            xy = np.load(xy_path)
        else:
            x = h.cpu().numpy()
            dim = min(pca_dim, x.shape[1], len(x) - 1)
            reduced = PCA(n_components=dim, svd_solver='randomized', random_state=tsne_seed).fit_transform(x)
            xy = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto',
                      max_iter=tsne_steps, random_state=tsne_seed).fit_transform(reduced)
            np.save(xy_path, xy)
        teachers = {}
        for ratio in ratios:
            kernel, gamma, temperature, mu, _ = BEST_HYPERPARAMS_DICT[(dataset, ratio)]
            config = dict(teacher_kernel=kernel, gamma=gamma, T=temperature, kl_weight=mu, basis=3000)
            config.update((configs or {}).get((dataset, ratio), {}))
            key = _fingerprint({k: v for k, v in config.items() if k != 'kl_weight'})
            if key not in teachers:
                path = root / f'{dataset}_teacher_{key}.pt'
                if path.exists():
                    teachers[key] = torch.load(path, map_location=device, weights_only=True)
                else:
                    seed_everything(teacher_seed)
                    teachers[key] = get_teacher_labels(h, mask, train['y'], config['teacher_kernel'],
                        config['gamma'], config['T'], config['basis'])
                    save_tensor(path, teachers[key].cpu())
            q = teachers[key]
            case = root / f'{dataset}_{ratio:g}'
            case.mkdir(exist_ok=True)
            panels = {}
            for seed in partition_seeds:
                path = case / f'{seed}.pt'
                if path.exists():
                    result = torch.load(path, weights_only=True)
                else:
                    result = partition(h, q, BUDGET[(dataset, ratio)], kl_weight=config['kl_weight'],
                        seed=seed, iters=grip_steps, return_diagnostics=True, return_initial_state=True)
                    save_tensor(path, result)
                for stage in ('initial', 'final'):
                    assignment = result['initial_assignment' if stage == 'initial' else 'assignment'].numpy()
                    panels[(seed, stage)] = dict(assignment=assignment, J=result[f'{stage}_J'])
                rows.append(dict(dataset=dataset, ratio=ratio, comparison=f'{seed}: initial -> final',
                    initial_J=result['initial_J'], final_J=result['final_J'], converged=result['converged'],
                    **compare_partitions(panels[(seed, 'initial')]['assignment'], panels[(seed, 'final')]['assignment'])))
            for stage in ('initial', 'final'):
                rows.append(dict(dataset=dataset, ratio=ratio, comparison=f'{stage}: seeds {partition_seeds[0]} vs {partition_seeds[1]}',
                    **compare_partitions(*(panels[(seed, stage)]['assignment'] for seed in partition_seeds))))
            figures[(dataset, ratio)] = render_case(xy, q.cpu().numpy(), panels, case, f'{dataset} | ratio={ratio:g}')
        del train, mask, validation, testing, h, q, teachers
        torch.cuda.empty_cache()
    metrics = pd.DataFrame(rows)
    metrics.to_csv(root / 'metrics.csv', index=False)
    return dict(metrics=metrics, figures=figures, folder=str(root))
