import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.convex_representatives import convex_features, identity_hidden
from src.dataloader import get_dataset
from src.empirical_ntk_study import make_network
from src.node_distances import array_digest
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student
from src.tree_distance import _write_json


@torch.no_grad()
def linear_oracle(h, gradient, assignment=None, block_size=16384):
    best = h.new_full((len(gradient),), torch.inf)
    ids = torch.zeros(len(gradient), dtype=torch.long, device=h.device)
    clusters = torch.arange(len(gradient), device=h.device)
    for start in range(0, len(h), block_size):
        scores = gradient @ h[start:start + block_size].T
        if assignment is not None:
            scores.masked_fill_(clusters[:, None] != assignment[start:start + block_size][None, :], torch.inf)
        value, index = scores.min(1)
        better = value < best
        best = torch.minimum(best, value)
        ids = torch.where(better, start + index, ids)
    return ids, best


def save_state(path, state):
    temporary = Path(path).with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def refine_hull(model, h, targets, initial, path, assignment=None, max_steps=20000,
                tolerance=1e-6, block_size=16384, checkpoint_every=250):
    if max_steps < 1 or tolerance <= 0 or block_size < 1 or checkpoint_every < 1:
        raise ValueError('Require positive solver settings')
    h, targets, x = h.double(), targets.double(), initial.double().clone()
    layer = model.layers[0]
    weight = layer.lin.weight.detach().double()
    bias = None if layer.bias is None else layer.bias.detach().double()
    scale = targets.square().sum(1).clamp_min(1)
    history, vertices, rates, start = [], [], [], 0
    path = Path(path)
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['complete']:
            return saved
        x = saved['x'].to(h.device)
        history = saved['history']
        vertices, rates = list(saved['vertices'].unbind()), list(saved['rates'].unbind())
        start = saved['steps']

    def record(step, status, complete):
        value = dict(x=x.cpu(), steps=step, status=status, converged=status == 'stationary',
                     complete=complete, history=history,
                     vertices=torch.stack(vertices) if vertices else torch.empty(0, len(x), dtype=torch.long),
                     rates=torch.stack(rates) if rates else torch.empty(0, len(x), dtype=torch.float64),
                     max_relative_gap=float(relative_gap.max()))
        save_state(path, value)
        return value

    for step in tqdm(range(start, max_steps + 1), desc='Cluster hull' if assignment is not None else 'Global hull'):
        preactivation = F.linear(x, weight, bias)
        error = preactivation.relu() - targets
        losses = error.square().sum(1)
        gradient = (2 * error * (preactivation > 0)) @ weight
        ids, minimum = linear_oracle(h, gradient, assignment, block_size)
        gap = ((gradient * x).sum(1) - minimum).clamp_min(0)
        relative_gap = gap / scale
        active = relative_gap > tolerance
        if step % 25 == 0 or not bool(active.any()) or step == max_steps:
            history.append(dict(step=step, reconstruction_mse=float(losses.mean()),
                                max_relative_gap=float(relative_gap.max())))
        if not bool(active.any()):
            return record(step, 'stationary', True)
        if step == max_steps:
            return record(step, 'iteration_limit', True)
        direction = h[ids] - x
        delta = F.linear(direction, weight)
        alpha = active.to(x.dtype)
        for _ in range(40):
            proposed = (preactivation + alpha[:, None] * delta).relu()
            new_loss = (proposed - targets).square().sum(1)
            failed = active & (new_loss > losses - 1e-4 * alpha * gap)
            if not bool(failed.any()):
                break
            alpha[failed] *= .5
        else:
            return record(step, 'line_search_stalled', True)
        x += alpha[:, None] * direction
        vertices.append(ids.cpu())
        rates.append(alpha.cpu())
        if (step + 1) % checkpoint_every == 0:
            record(step + 1, 'running', False)


def run_convex_expansion(previous_dir, teacher_run, output_dir, max_steps=20000, tolerance=1e-6,
                          block_size=16384, seeds=tuple(range(100, 110)), data_dir='/content/data/', device='cuda'):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Require distinct nonempty student seeds')
    previous = Path(previous_dir)
    old = json.loads((previous / 'protocol.json').read_text())
    protocol, params = old['source_protocol'], old['params']
    if protocol['settings']['loss_weighting'] != 'uniform':
        raise ValueError('Require the uniform CE source experiment')
    source = Path(old['source'])
    key = _fingerprint(dict(mode='hidden', T=params['T'], kl_weight=params['kl_weight']))
    state = torch.load(source / f'partition_{key}.pt', map_location='cpu', weights_only=True)
    digest = array_digest(state['assignment'].numpy(), state['x'].numpy(), state['y'].numpy(), state['metric_centers'].numpy())
    if digest != old['partition']:
        raise ValueError('Source partition changed')
    prior = torch.load(previous / 'convex.pt', map_location='cpu', weights_only=True)
    graph = get_dataset(SimpleNamespace(dataset_name=protocol['dataset'], raw_data_dir=str(data_dir).rstrip('/') + '/'))
    if array_digest(graph.x.numpy(), graph.edge_index.numpy(), graph.y.numpy(), graph.train_mask.numpy(),
                    graph.val_mask.numpy(), graph.test_mask.numpy()) != protocol['graph']:
        raise ValueError('Source graph changed')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    teacher = protocol['teacher']
    model = make_network(graph.x.to(device), graph.edge_index.to(device), 'gcn', teacher['hidden'],
                          len(teacher['classes']), teacher['seed'], teacher['dropout'])
    weights = torch.load(Path(teacher_run) / 'teacher_state.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(weights)
    if array_digest(*[p.cpu().numpy() for p in model.state_dict().values()]) != protocol['teacher_state']:
        raise ValueError('Source teacher changed')
    model.eval().requires_grad_(False)
    _, _, validation, testing, h = _prepare_dataset(protocol['dataset'], data_dir, device)
    config = dict(version=1, previous=str(previous), previous_protocol=old,
                  initial=array_digest(prior['x'].numpy(), prior['weights'].numpy()), max_steps=max_steps,
                  tolerance=tolerance, block_size=block_size, seeds=list(seeds),
                  solver='Frank_Wolfe_Armijo', torch=str(torch.__version__), device=str(device))
    folder = Path(output_dir) / _fingerprint(config)
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(folder / 'protocol.json', config)
    targets, assignment = state['metric_centers'].to(device), state['assignment'].to(device)
    initial_weights = prior['weights'].to(device)
    totals = initial_weights.new_zeros(len(targets)).index_add_(0, assignment, initial_weights)
    if not bool(torch.isfinite(initial_weights).all()) or bool((initial_weights < 0).any()) or not torch.allclose(totals, torch.ones_like(totals)):
        raise ValueError('Invalid saved convex coefficients')
    initial = convex_features(h.double(), assignment, initial_weights / totals[assignment], len(targets))
    if not torch.allclose(initial.float(), prior['x'].to(device), atol=1e-5, rtol=1e-4):
        raise ValueError('Saved coefficients do not reproduce the previous representatives')
    within = refine_hull(model, h, targets, initial, folder / 'within.pt',
                         assignment, max_steps, tolerance, block_size)
    global_fit = refine_hull(model, h, targets, within['x'].to(device), folder / 'global.pt',
                             None, max_steps, tolerance, block_size)
    variants = dict(median=state['x'], within_1000=prior['x'], within_refined=within['x'], global_refined=global_fit['x'])
    fits = dict(within_refined=within, global_refined=global_fit)
    summary, repeats, histories = [], [], []
    for method, fit in fits.items():
        histories.extend(dict(method=method, **row) for row in fit['history'])
    for method, features in variants.items():
        cx, cy = features.float().to(device), state['y'].to(device)
        with torch.no_grad():
            reconstruction = float((identity_hidden(model, cx) - targets).square().sum(1).mean())
        rows = []
        for seed in tqdm(seeds, desc=f'{method}: paired students'):
            path = folder / f'{method}_{seed}.json'
            if path.exists():
                row = json.loads(path.read_text())
            else:
                val, test, epoch = _train_student(cx, cy, validation, params, seed, protocol['settings'], testing=testing)
                row = dict(method=method, seed=seed, validation=val * 100, test=test * 100, epoch=epoch)
                _write_json(path, row)
            rows.append(row)
        frame = pd.DataFrame(rows)
        repeats.extend(rows)
        fit = fits.get(method, {})
        summary.append(dict(method=method, reconstruction_mse=reconstruction,
                            final_val=frame.validation.mean(), final_val_std=frame.validation.std(ddof=0),
                            test_mean=frame.test.mean(), test_std=frame.test.std(ddof=0),
                            status=fit.get('status', 'reference'), steps=fit.get('steps'),
                            max_relative_gap=fit.get('max_relative_gap')))
    summary, repeats, history = pd.DataFrame(summary), pd.DataFrame(repeats), pd.DataFrame(histories)
    paired = repeats.pivot(index='seed', columns='method', values=['validation', 'test'])
    differences = pd.DataFrame({f'{method}_{metric}_delta': paired[metric][method] - paired[metric]['within_refined']
                                for method in ('within_1000', 'global_refined') for metric in ('validation', 'test')})
    for name, frame in [('summary', summary), ('final_seeds', repeats), ('history', history)]:
        frame.to_csv(folder / f'{name}.csv', index=False)
    differences.to_csv(folder / 'paired_deltas.csv')
    return dict(summary=summary, history=history, paired=differences.reset_index(), folder=str(folder))


def plot_convex_expansion(report):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for method, table in report['history'].groupby('method'):
        axes[0].plot(table.step, table.reconstruction_mse, label=method)
        axes[1].semilogy(table.step, table.max_relative_gap.clip(lower=1e-16), label=method)
    axes[0].set(xlabel='Additional steps', ylabel='Mean squared hidden distance')
    axes[1].set(xlabel='Additional steps', ylabel='Maximum relative FW gap')
    axes[0].legend()
    table = report['summary']
    for mean, std, label, offset in [('final_val', 'final_val_std', 'Validation', -.15), ('test_mean', 'test_std', 'Test', .15)]:
        axes[2].errorbar(torch.arange(len(table)).numpy() + offset, table[mean], yerr=table[std], fmt='o', label=label)
    axes[2].set(xticks=range(len(table)), xticklabels=table.method, ylabel='Accuracy (%)')
    axes[2].tick_params(axis='x', labelrotation=30)
    axes[2].legend()
    fig.savefig(Path(report['folder']) / 'comparison.png', dpi=180)
    return fig
