"""Check local Fisher/KL clustering costs for a graphless student on Z=P^2 X.

This is a diagnostic, not a new condensation method or a raw-graph GCN benchmark.
"""
import argparse
import csv
import hashlib
import json
import os
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch_geometric import seed_everything
from torch_geometric.data import Data

from sweep_distance import atomic_json, atomic_torch, code_digest, data_digest, digest
from src.dataloader import get_dataset, set_dataset
from src.models import GCN
from src.partition import partition
from src.teacher import get_kernel_features, fit_logistic
from src.utils import budget, conv_graph_multi, soft_label_ce


def parameters(model):
    """Float64 copies for cancellation-sensitive KL diagnostics; dropout is off."""
    return tuple(t.detach().double().clone() for t in (
        model.layers[0].lin.weight, model.layers[0].bias,
        model.layers[1].lin.weight, model.layers[1].bias))


def logits_and_hidden(x, weights):
    w0, b0, w1, b1 = weights
    pre = F.linear(x, w0, b0)
    return F.linear(pre.relu(), w1, b1), pre


@torch.no_grad()
def pair_costs(x, c, teacher, weights):
    """KL(p_x||p_c), its input/logit quadratic approximations, exact CE identity.

No dense feature-by-feature Fisher matrix is constructed. Input JVP is exact
away from ReLU kinks (zero uses PyTorch's derivative convention).
"""
    zx, hx = logits_and_hidden(x, weights)
    zc, hc = logits_and_hidden(c, weights)
    lp, lq = zx.log_softmax(1), zc.log_softmax(1)
    p = lp.exp()
    dz = zc-zx
    w0, _, w1, _ = weights
    tangent = F.linear(F.linear(c-x, w0)*(hx > 0), w1)

    def quadratic(direction):
        # Weighted variance, with a centered computation to avoid cancellation.
        centered = direction-(p*direction).sum(1, keepdim=True)
        return .5*(p*centered.square()).sum(1)

    kl_signed = (p*(lp-lq)).sum(1)
    correction = ((p-teacher)*dz).sum(1)
    ce_delta = (teacher*(lp-lq)).sum(1)
    identity_error = ce_delta-kl_signed-correction
    if kl_signed.min() < -1e-10 or identity_error.abs().max() > 1e-9:
        raise RuntimeError('KL nonnegativity / CE decomposition check failed')
    return dict(kl=kl_signed.clamp_min(0), quadratic=quadratic(tangent),
        logit_quadratic=quadratic(dz), correction=correction, ce_delta=ce_delta,
        identity_error=identity_error,
        gate_fraction=((hx > 0) != (hc > 0)).double().mean(1),
        gate_any=((hx > 0) != (hc > 0)).any(1).double(),
        euclidean=(c-x).square().sum(1),
        tangent_error=(dz-tangent).norm(dim=1),
        base_kink=(hx == 0).any(1).double())


def correlation(a, b):
    if len(a) < 2 or np.ptp(a) <= 1e-15 or np.ptp(b) <= 1e-15:
        return None
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else None


def summarize(values):
    kl, q, lq = (values[k] for k in ('kl', 'quadratic', 'logit_quadratic'))
    denominator = float(np.mean(kl))
    usable = denominator > 1e-12
    return dict(n=len(kl), kl_mean=denominator, quadratic_mean=float(q.mean()),
        relative_mae=float(np.abs(q-kl).mean()/denominator) if usable else None,
        logit_relative_mae=float(np.abs(lq-kl).mean()/denominator) if usable else None,
        spearman=correlation(kl, q),
        correction_abs_over_kl=float(np.abs(values['correction']).mean()/denominator) if usable else None,
        correction_mean=float(values['correction'].mean()),
        correction_abs_mean=float(np.abs(values['correction']).mean()),
        ce_delta_mean=float(values['ce_delta'].mean()),
        ce_delta_abs_mean=float(np.abs(values['ce_delta']).mean()),
        ce_approx_abs_error=float(np.abs(q+values['correction']-values['ce_delta']).mean()),
        gate_any=float(values['gate_any'].mean()), gate_fraction=float(values['gate_fraction'].mean()),
        identity_max_error=float(np.abs(values['identity_error']).max()),
        base_kink_fraction=float(values['base_kink'].mean()))


def ranking_report(exact, approximate):
    """Compare costs within each anchor; ties at the actual KL minimum are accepted."""
    selected = approximate.argmin(1)
    minimum = exact.min(1)
    chosen = exact[np.arange(len(exact)), selected]
    regret = np.maximum(0., chosen-minimum)
    tolerance = 1e-10+1e-7*np.abs(minimum)
    correlations = [correlation(a, b) for a, b in zip(exact, approximate)]
    valid = [value for value in correlations if value is not None]
    return dict(anchors=len(exact), candidates=exact.shape[1],
        spearman_mean=float(np.mean(valid)) if valid else None, spearman_valid=len(valid),
        top1_agreement=float(np.mean(regret <= tolerance)),
        excess_kl_mean=float(regret.mean()), chosen_kl_mean=float(chosen.mean()),
        oracle_kl_mean=float(minimum.mean()))


def atomic_npz(path, **arrays):
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temp, path)


def save_csv(path, rows):
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def identity_graph(x, y):
    ids = torch.arange(len(x), device=x.device)
    return Data(x=x.float(), y=y.float(), edge_index=torch.stack([ids, ids]),
        edge_attr=torch.ones(len(x), device=x.device),
        train_mask=torch.ones(len(x), dtype=torch.bool, device=x.device))


def train_student(model, graph, epochs):
    """Uniform soft CE, existing optimizer schedule, fixed final epoch; no val/test selection."""
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=5e-4)
    for epoch in range(1, epochs+1):
        if epoch == epochs//2:
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
        model.train()
        loss = soft_label_ce(model(graph), graph.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == epochs:
            print(f'Epoch {epoch}: uniform soft CE={float(loss):.6f}', flush=True)
    model.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='/content/drive/MyDrive/GRIP_cora_fisher_diagnostic')
    p.add_argument('--raw-data-dir', default='/content/data/')
    # Latest GRIP validation-selected triple from the Cora coarsening full sweep.
    p.add_argument('--gamma', type=float, default=.001)
    p.add_argument('--temperature', type=float, default=10.)
    p.add_argument('--coefficient', type=float, default=1.)
    p.add_argument('--dropout', type=float, default=.9)
    p.add_argument('--epoch', type=int, default=1000)
    p.add_argument('--seeds', default='23,24,25')
    p.add_argument('--steps', default='0.01,0.03,0.1,0.3,1')
    p.add_argument('--anchors', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    seeds = sorted(set(int(s) for s in args.seeds.split(',')))
    steps = sorted(set(float(s) for s in args.steps.split(',')))
    if (not seeds or min(seeds) < 0 or not steps or not np.isfinite(steps).all()
            or min(steps) <= 0 or max(steps) > 1 or 1. not in steps
            or args.epoch < 2 or args.anchors < 1 or args.batch_size < 1
            or not np.isfinite([args.gamma, args.temperature, args.coefficient, args.dropout]).all()
            or min(args.gamma, args.temperature) <= 0 or args.coefficient < 0 or not 0 <= args.dropout < 1):
        p.error('Invalid settings; steps must include 1 and lie in (0,1]')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.output).resolve()
    for folder in ('cache', 'models', 'diagnostics'):
        (out/folder).mkdir(parents=True, exist_ok=True)
    logfile = out/f'diagnostic_{datetime.now():%Y%m%d_%H%M%S}.log'
    print('GPU:', torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU')
    print(f'Results: {out}\nLog: {logfile}')
    print(f'Cora .052; gamma={args.gamma:g}, T={args.temperature:g}, mu={args.coefficient:g}; '
          f'{len(seeds)} graphless student fits, dropout={args.dropout:g}, epochs={args.epoch}.')
    print('Input Z=P^2 X; identity edges at both train and evaluation. No raw-graph GCN accuracy comparison.')
    try:
        from IPython import get_ipython
        from IPython.display import display
        handle = display('Preparing Fisher diagnostic...', display_id=True) if get_ipython() else None
    except ImportError:
        handle = None

    def progress(message):
        if handle:
            handle.update(message)
        else:
            print('\r'+message.ljust(110), end='', flush=True)

    train = SimpleNamespace(dataset_name='cora', ratio=.052, device=args.device,
        raw_data_dir=str(Path(args.raw_data_dir).resolve())+'/')
    records, rank_records, model_stats, manifests = [], [], [], []
    with logfile.open('w', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            seed_everything(0)
            train, data, _, _ = set_dataset(train, get_dataset(train))
            _, _, h2 = conv_graph_multi(train, data)
        base = dict(source=code_digest(), runner=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            data=data_digest(data), dataset='cora', ratio=.052, gamma=args.gamma, T=args.temperature,
            mu=args.coefficient, kernel='relu', basis=3000, condensation_seed=0,
            torch=str(torch.__version__), device=args.device, threads=4)
        cache = out/'cache'/f'{digest(base)}.pt'
        if cache.exists():
            artifact = torch.load(cache, map_location='cpu', weights_only=True)
        else:
            progress('Fitting teacher and GRIP partition...')
            with redirect_stdout(log), redirect_stderr(log):
                seed_everything(0)
                features = get_kernel_features(h2, 'relu', 3000)
                labels = F.one_hot(data.y[data.train_mask], train.num_class).double()
                w = fit_logistic(features[data.train_mask], labels, args.gamma)
                teacher = ((features@w)/args.temperature).softmax(1).clamp_min(1e-12)
                teacher /= teacher.sum(1, keepdim=True)
                c, y, assignment = partition(h2, teacher, budget(train), args.coefficient)
                artifact = dict(Z=h2.cpu(), F=teacher.cpu(), C=c.cpu(), Y=y.cpu(), a=assignment.cpu())
                atomic_torch(cache, artifact)
                del features, w, teacher, c, y, assignment
        Z, teacher, C = (artifact[k].double().to(args.device) for k in ('Z', 'F', 'C'))
        assignment = artifact['a'].to(args.device)
        m, n = len(C), len(Z)
        counts = torch.bincount(assignment, minlength=m).double()
        means = torch.zeros_like(C).index_add_(0, assignment, Z)/counts[:, None]
        graph = identity_graph(C, artifact['Y'].to(args.device))
        anchors = np.random.default_rng(0).choice(n, size=min(n, args.anchors), replace=False)
        conf = dict(**base, epoch=args.epoch, dropout=args.dropout, hidden=256,
            student='GCN-with-identity-edges=MLP', checkpoint='fixed-final-epoch', loss='uniform-soft-CE',
            artifact_sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
        probe_config = dict(**conf, steps=steps, anchors=anchors.tolist(), batch_size=args.batch_size)

        for seed in seeds:
            model_key = digest(dict(**conf, seed=seed))
            model_path = out/'models'/f'{model_key}.pt'
            seed_everything(seed)
            model = GCN(Z.shape[1], 256, train.num_class, 2, args.dropout).to(args.device)
            if model_path.exists():
                model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
                model.eval()
            else:
                progress(f'Training graphless student, seed={seed}...')
                with redirect_stdout(log), redirect_stderr(log):
                    train_student(model, graph, args.epoch)
                    atomic_torch(model_path, {k: v.detach().cpu() for k, v in model.state_dict().items()})
            weights = parameters(model)
            probe_key = digest(dict(**probe_config, seed=seed,
                model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest()))
            report_path = out/'diagnostics'/f'{probe_key}.json'
            array_path = out/'diagnostics'/f'{probe_key}.npz'
            manifests.append(dict(seed=seed, model=str(model_path), report=str(report_path), arrays=str(array_path)))
            atomic_json(out/'manifest.json', dict(config=probe_config, seeds=seeds, entries=manifests))
            if report_path.exists() and array_path.exists():
                report = json.loads(report_path.read_text(encoding='utf-8'))
                if report['config'] != probe_config or report['seed'] != seed:
                    raise ValueError(f'Cache mismatch: {report_path}')
            else:
                progress(f'Diagnosing local approximation and candidate rankings, seed={seed}...')
                arrays, current, rankings = {}, [], []
                with torch.no_grad():
                    # Check that the analytic model really is the identity-edge student.
                    reference = model(graph).double()
                    predicted, _ = logits_and_hidden(graph.x.double(), weights)
                    parity = float((reference-predicted.log_softmax(1)).abs().max())
                    if parity > 5e-4:
                        raise RuntimeError(f'Identity-edge/MLP parity failed: {parity}')
                    zx, _ = logits_and_hidden(Z, weights)
                    zc, _ = logits_and_hidden(C, weights)
                    pz = zx.log_softmax(1)
                    stats = dict(seed=seed, teacher_ce=float(-(teacher*pz).sum(1).mean()),
                        teacher_kl=float((teacher*(teacher.log()-pz)).sum(1).mean()),
                        condensed_uniform_ce=float(-(graph.y.double()*zc.log_softmax(1)).sum(1).mean()),
                        numerical_model_parity=parity)
                    for endpoint, centers in (('grip', C), ('cell_mean', means)):
                        for step in steps:
                            collected = {}
                            for start in range(0, n, args.batch_size):
                                x = Z[start:start+args.batch_size]
                                endpoint_x = centers[assignment[start:start+len(x)]]
                                result = pair_costs(x, x+step*(endpoint_x-x), teacher[start:start+len(x)], weights)
                                for name, value in result.items():
                                    collected.setdefault(name, []).append(value.cpu().numpy())
                            values = {name: np.concatenate(parts) for name, parts in collected.items()}
                            prefix = f'{endpoint}_t{step:g}_'
                            arrays.update({prefix+name: value for name, value in values.items()})
                            current.append(dict(seed=seed, endpoint=endpoint, step=step, **summarize(values)))
                    # Rank ALL actual GRIP center candidates per anchor at full displacement.
                    matrices = {name: [] for name in ('kl', 'quadratic', 'euclidean')}
                    flat_count = len(anchors)*m
                    for start in range(0, flat_count, args.batch_size):
                        flat = np.arange(start, min(start+args.batch_size, flat_count))
                        nodes = torch.as_tensor(anchors[flat//m], device=args.device)
                        cells = torch.as_tensor(flat % m, device=args.device)
                        result = pair_costs(Z[nodes], C[cells], teacher[nodes], weights)
                        for name in matrices:
                            matrices[name].append(result[name].cpu().numpy())
                    matrices = {name: np.concatenate(parts).reshape(len(anchors), m) for name, parts in matrices.items()}
                    arrays.update({'ranking_'+name: value for name, value in matrices.items()})
                    arrays.update(anchors=anchors, assignment=assignment.cpu().numpy())
                    for name in ('quadratic', 'euclidean'):
                        rankings.append(dict(seed=seed, cost=name,
                            **ranking_report(matrices['kl'], matrices[name])))
                report = dict(config=probe_config, seed=seed, model=stats, probes=current, rankings=rankings)
                atomic_npz(array_path, **arrays)
                atomic_json(report_path, report)
            records.extend(report['probes'])
            rank_records.extend(report['rankings'])
            model_stats.append(report['model'])
            save_csv(out/'probes.csv', records)
            save_csv(out/'rankings.csv', rank_records)
            del model, weights

    def average(rows, key):
        values = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(values)) if values else None

    def display_number(value, scale=1., digits=3):
        return 'n/a' if value is None else f'{value*scale:.{digits}f}'

    if handle:
        handle.update('Complete. Frozen graphless-student diagnostic below.')
    else:
        print()
    print('GRIP endpoint: t=1 reaches its assigned representative; averages across student seeds.')
    print('t       KL mean   Fisher relMAE%  Logit relMAE%  gate-cross%  |correction|/KL')
    for step in steps:
        rows = [r for r in records if r['endpoint'] == 'grip' and r['step'] == step]
        print(f'{step:<5g} {average(rows,"kl_mean"):10.3g} '
              f'{display_number(average(rows,"relative_mae"),100):>15s} '
              f'{display_number(average(rows,"logit_relative_mae"),100):>14s} '
              f'{display_number(average(rows,"gate_any"),100):>12s} '
              f'{display_number(average(rows,"correction_abs_over_kl")):>16s}')
    print('All-center ranking by actual KL (t=1; no label-KL penalty):')
    print('cost        Spearman   top1%   excess KL')
    for name in ('quadratic', 'euclidean'):
        rows = [r for r in rank_records if r['cost'] == name]
        print(f'{name:10s} {display_number(average(rows,"spearman_mean")):>9s} '
              f'{display_number(average(rows,"top1_agreement"),100):>7s} '
              f'{average(rows,"excess_kl_mean"):.4g}')
    identity_error = max(r['identity_max_error'] for r in records)
    print(f'Exact CE identity max error: {identity_error:.3g}. Cell-mean probes and per-seed details saved.')
    atomic_json(out/'summary.json', dict(config=probe_config, seeds=seeds, models=model_stats,
        probes=records, rankings=rank_records, identity_max_error=identity_error,
        scope='Frozen graphless students on propagated features, fixed final epoch, float64 diagnostic. '
        'No val/test label selection; no retraining transfer or global-risk certificate. '
        'Fisher is local and may fail across ReLU boundaries. Correction term is not omitted.'))
    print('Full results: summary.json | probes.csv | rankings.csv | diagnostics/*.npz')


if __name__ == '__main__':
    main()
