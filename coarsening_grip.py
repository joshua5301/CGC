"""Paired comparison: GRIP identity, fixed coarse means, optimized coarse features."""
import argparse
import copy
import csv
import hashlib
import json
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.data import Data

from sweep_distance import atomic_json, atomic_torch, code_digest, data_digest, digest
from extend_robust import paired_stats
from src.dataloader import get_dataset, set_dataset
from src.models import GCN
from src.partition import partition
from src.teacher import get_kernel_features, fit_logistic
from src.utils import BUDGET, budget, conv_graph_multi, soft_label_ce
from src.coarsening_features import fixed_coarsening, optimize_features, feature_diagnostics, gcn_operator


def train_best(model, args, data, graph):
    """Same optimizer/schedule/strict-best-val rule as model_training; restore best for diagnostics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, best_test, best_epoch, state = 0., 0., None, None
    for epoch in range(1, args.epoch+1):
        if epoch == args.epoch//2:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr*.1, weight_decay=args.weight_decay)
        model.train()
        loss = soft_label_ce(model(graph)[graph.train_mask], graph.y[graph.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % args.eval_every and epoch != args.epoch:
            continue
        with torch.no_grad():
            model.eval()
            pred = model(data).argmax(1)
            # Integer counts / Python division match the existing evaluation exactly.
            val = int((pred[data.val_mask] == data.y[data.val_mask]).sum())/int(data.val_mask.sum())
            test = int((pred[data.test_mask] == data.y[data.test_mask]).sum())/int(data.test_mask.sum())
        if val > best_val:
            best_val, best_test, best_epoch = val, test, epoch
            state = copy.deepcopy(model.state_dict())
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: loss={float(loss):.6f}, val={best_val:.4f}, test={best_test:.4f}', flush=True)
    if state is not None:
        model.load_state_dict(state)
        model.eval()
    return best_val, best_test, best_epoch


@torch.no_grad()
def measure(model, data, graph, assignment, teacher, labels, P, Q):
    """Deterministic eval-mode same-parameter bounds and exact teacher-risk decomposition."""
    model.eval()
    def representations(g):
        hidden = model.layers[0](g.x, g.edge_index, g.edge_attr).relu()
        logits = model.layers[1](hidden, g.edge_index, g.edge_attr)
        return hidden.double(), logits.double()
    h, z = representations(data)
    hc, zc = representations(graph)
    a, teacher, labels = assignment.to(z.device), teacher.to(z.device).double(), labels.to(z.device).double()
    diagnostic = feature_diagnostics(P, Q, data.x, graph.x, a)
    w0 = float(torch.linalg.matrix_norm(model.layers[0].lin.weight.double(), ord=2))
    w1 = float(torch.linalg.matrix_norm(model.layers[1].lin.weight.double(), ord=2))
    M1 = float(hc.norm(dim=1).max())
    bound = w1*(w0*diagnostic['D_prop']+M1*diagnostic['D_struct'])
    logit_gap = float((z-zc[a]).norm(dim=1).mean())
    lp, lpc = z.log_softmax(1), zc.log_softmax(1)
    original = float(-(teacher*lp).sum(1).mean())
    lifted = float(-(teacher*lpc[a]).sum(1).mean())
    cell_ce = -(labels*lpc).sum(1)
    uniform = float(cell_ce.mean())
    pi = torch.bincount(a, minlength=len(labels)).double()/len(a)
    mass = float(((pi-1/len(labels))*cell_ce).sum())
    identity_error = original-uniform-(original-lifted)-mass
    if logit_gap > bound+1e-4*max(1., bound) or abs(identity_error) > 1e-5:
        raise RuntimeError('GCN bound / exact risk decomposition failed')
    return dict(**diagnostic, hidden_gap=float((h-hc[a]).norm(dim=1).mean()),
        logit_gap=logit_gap, logit_bound=bound, W0_norm=w0, W1_norm=w1, M1=M1,
        original_teacher_ce=original, condensed_uniform_ce=uniform,
        prediction_ce_delta=original-lifted, signed_mass_correction=mass,
        risk_identity_error=identity_error,
        teacher_ce_upper=uniform+np.sqrt(2)*bound+mass,
        scope='Post-fit same-parameter eval bound; not a training/generalization certificate')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('cora', 'citeseer'), default='cora')
    p.add_argument('--ratio', type=float, default=None)
    p.add_argument('--output', default=None)
    p.add_argument('--raw-data-dir', default='/content/data/')
    # Most recent full-sweep GRIP winner by selection validation, not by test.
    p.add_argument('--gamma', type=float, default=.01)
    p.add_argument('--temperature', type=float, default=None)
    p.add_argument('--coefficient', type=float, default=None)
    p.add_argument('--dropout', type=float, default=.9)
    p.add_argument('--repeat', type=int, default=10)
    p.add_argument('--seed-start', type=int, default=3)
    p.add_argument('--epoch', type=int, default=1000)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--feature-steps', type=int, default=1000)
    p.add_argument('--feature-tolerance', type=float, default=1e-3)
    p.add_argument('--smooth-relative', type=float, default=1e-4)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    # Citeseer teacher/partition settings use the first reported validation tie;
    # dropout stays .9 to retain the current coarsening comparison protocol.
    defaults = dict(cora=(.052, 5., 2., 'relu'), citeseer=(.036, .2, .1, 'erf'))
    ratio, temperature, coefficient, kernel = defaults[args.dataset]
    if args.ratio is None:
        args.ratio = ratio
    if args.temperature is None:
        args.temperature = temperature
    if args.coefficient is None:
        args.coefficient = coefficient
    if args.output is None:
        args.output = f'/content/drive/MyDrive/GRIP_{args.dataset}_coarsening_features'
    if (args.dataset, args.ratio) not in BUDGET:
        p.error(f'Unsupported ratio for {args.dataset}: {args.ratio}')
    if (not np.isfinite([args.ratio, args.gamma, args.temperature, args.coefficient, args.dropout,
                        args.feature_tolerance, args.smooth_relative]).all()
            or not 0 < args.ratio <= 1 or args.gamma <= 0 or args.temperature <= 0 or args.coefficient < 0
            or not 0 <= args.dropout < 1 or args.repeat < 2 or args.seed_start < 0
            or not 1 <= args.eval_every <= args.epoch or args.feature_steps < 1
            or min(args.feature_tolerance, args.smooth_relative) <= 0):
        raise ValueError('Invalid experiment settings')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.output).resolve()
    for folder in ('teachers', 'condensed', 'runs'):
        (out/folder).mkdir(parents=True, exist_ok=True)
    logfile = out/f'coarsening_{datetime.now():%Y%m%d_%H%M%S}.log'
    print('GPU:', torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU')
    print(f'Results: {out}\nLog: {logfile}')
    print(f'Fixed {args.dataset} {args.ratio:g}: gamma={args.gamma:g}, T={args.temperature:g}, mu={args.coefficient:g}, dropout={args.dropout:g}')
    try:
        from IPython import get_ipython
        from IPython.display import display
        handle = display('Preparing paired coarsening experiment...', display_id=True) if get_ipython() else None
    except ImportError:
        handle = None
    def progress(message):
        if handle:
            handle.update(message)
        else:
            print('\r'+message.ljust(115), end='', flush=True)
    train = SimpleNamespace(dataset_name=args.dataset, ratio=args.ratio, device=args.device,
        raw_data_dir=str(Path(args.raw_data_dir).resolve())+'/', n_dim=256,
        lr=.01, weight_decay=5e-4, epoch=args.epoch, eval_every=args.eval_every)
    rows, entries = [], []
    with logfile.open('w', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            seed_everything(0)
            train, data, _, _ = set_dataset(train, get_dataset(train))
            _, _, h2 = conv_graph_multi(train, data)
        base = dict(source=code_digest(), runner=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            data=data_digest(data), dataset=args.dataset, ratio=args.ratio, gamma=args.gamma, T=args.temperature,
            kernel=kernel, basis=3000, condensation_seed=0, torch=str(torch.__version__),
            device=args.device, threads=4)
        path = out/'teachers'/f'{digest(base)}.pt'
        if path.exists():
            teacher = torch.load(path, map_location=args.device, weights_only=True)
        else:
            progress('Fitting teacher...')
            with redirect_stdout(log), redirect_stderr(log):
                seed_everything(0)
                features = get_kernel_features(h2, kernel, 3000)
                labels = F.one_hot(data.y[data.train_mask], train.num_class).double()
                w = fit_logistic(features[data.train_mask], labels, args.gamma)
                teacher = ((features@w)/args.temperature).softmax(1).clamp_min(1e-12)
                teacher /= teacher.sum(1, keepdim=True)
                atomic_torch(path, teacher.cpu())
                del features, w
        config = dict(**base, coefficient=args.coefficient, m=budget(train))
        grip_path = out/'condensed'/f'{digest(dict(**config, variant="grip"))}.pt'
        if grip_path.exists():
            grip = torch.load(grip_path, map_location='cpu', weights_only=True)
        else:
            progress('Computing shared GRIP partition and labels...')
            with redirect_stdout(log), redirect_stderr(log):
                seed_everything(0)
                x, y, a = partition(h2, teacher, budget(train), args.coefficient)
                ids = torch.arange(len(x))
                grip = dict(x=x.cpu(), y=y.cpu(), assign=a.cpu(), edge_index=torch.stack([ids, ids]),
                            edge_weight=torch.ones(len(x), dtype=torch.float64))
                atomic_torch(grip_path, grip)
        operators = fixed_coarsening(data.edge_index, data.edge_attr, len(data.x), grip['assign'])
        a = grip['assign'].to(args.device)
        m = len(grip['x'])
        counts = operators['counts'].to(args.device)
        mean = torch.zeros((m, data.num_features), dtype=torch.float64, device=args.device)
        mean.index_add_(0, a, data.x.double())
        mean /= counts[:, None]
        P, Q = operators['P'].to(args.device), operators['Q'].to(args.device)
        target = torch.sparse.mm(P, data.x.double())
        omega = torch.sparse.sum(P, dim=0).to_dense()
        radius = float(data.x.double().norm(dim=1).max())
        opt_config = dict(**config, feature_steps=args.feature_steps, tolerance=args.feature_tolerance,
                          smooth_relative=args.smooth_relative, radius=radius)
        opt_path = out/'condensed'/f'{digest(dict(**opt_config, variant="coarse_optimized"))}.pt'
        if opt_path.exists():
            optimized = torch.load(opt_path, map_location='cpu', weights_only=True)
        else:
            progress('Optimizing fixed-coarsening features...')
            with redirect_stdout(log), redirect_stderr(log):
                result = optimize_features(target, Q, a, omega, mean, radius,
                    args.feature_steps, args.feature_tolerance, args.smooth_relative)
            optimized = dict(x=result.pop('x').cpu(), y=grip['y'], assign=grip['assign'],
                edge_index=operators['edge_index'], edge_weight=operators['edge_weight'], optimization=result)
            atomic_torch(opt_path, optimized)
        artifacts = dict(grip=grip, coarse_mean=dict(x=mean.cpu(), y=grip['y'], assign=grip['assign'],
            edge_index=operators['edge_index'], edge_weight=operators['edge_weight']), coarse_optimized=optimized)
        feature_stats, fit_diagnostics = {}, {}
        for name, artifact in artifacts.items():
            fit_diagnostics[name] = []
            conf = dict(**(opt_config if name == 'coarse_optimized' else config), variant=name)
            artifact_path = grip_path if name == 'grip' else (opt_path if name == 'coarse_optimized'
                else out/'condensed'/f'{digest(conf)}.pt')
            if not artifact_path.exists():
                atomic_torch(artifact_path, artifact)
            # Model consumes raw pooled edges, normalizing exactly once internally.
            graph = Data(x=artifact['x'].float().to(args.device), y=artifact['y'].float().to(args.device),
                edge_index=artifact['edge_index'].to(args.device), edge_attr=artifact['edge_weight'].float().to(args.device),
                train_mask=torch.ones(m, dtype=torch.bool, device=args.device))
            actual_Q = gcn_operator(graph.edge_index, graph.edge_attr, m).to_dense().to(args.device)
            feature_stats[name] = feature_diagnostics(P, actual_Q, data.x, graph.x, a)
            student = dict(**conf, epoch=args.epoch, eval_every=args.eval_every, dropout=args.dropout,
                lr=.01, weight_decay=5e-4, student='GCN-2-256', loss='uniform-soft-CE',
                artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest())
            key = digest(student)
            entries.append(dict(id=key, config=student, artifact=str(artifact_path)))
            atomic_json(out/'manifest.json', dict(entries=entries, arguments=vars(args),
                feature_diagnostics=feature_stats, optimization=optimized['optimization']))
            for seed in range(args.seed_start, args.seed_start+args.repeat):
                progress(f'Student {name}, seed {seed} ({len(rows)+1}/{3*args.repeat})')
                run_path = out/'runs'/f'{key}_{seed}.json'
                if run_path.exists():
                    result = json.loads(run_path.read_text(encoding='utf-8'))
                    if result['config'] != student or result['seed'] != seed:
                        raise ValueError(f'Cached run mismatch: {run_path}')
                else:
                    with redirect_stdout(log), redirect_stderr(log):
                        seed_everything(seed)
                        model = GCN(data.num_features, 256, train.num_class, 2, args.dropout).to(args.device)
                        val, test, epoch = train_best(model, train, data, graph)
                        diag = None if epoch is None else measure(model, data, graph, a, teacher,
                            grip['y'], P, actual_Q)
                    result = dict(config=student, seed=seed, val=100*val, test=100*test,
                                  best_epoch=epoch, diagnostics=diag)
                    atomic_json(run_path, result)
                    del model
                row = dict(variant=name, seed=seed, val=result['val'], test=result['test'])
                rows.append(row)
                if result['diagnostics'] is not None:
                    fit_diagnostics[name].append(result['diagnostics'])
    with (out/'student_runs.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if handle:
        handle.update('Complete. Fixed-partition paired comparisons below.')
    else:
        print()
    measured = {name: {metric: float(np.mean([d[metric] for d in values])) for metric in
        ('logit_gap', 'logit_bound', 'condensed_uniform_ce', 'original_teacher_ce', 'signed_mass_correction')}
        if values else None for name, values in fit_diagnostics.items()}
    print('Diagnostics: features and mean best-checkpoint measurements')
    print('variant             D_prop       D_struct   logit gap   student CE')
    for name, diag in feature_stats.items():
        extra = (f'{measured[name]["logit_gap"]:11.5f} {measured[name]["condensed_uniform_ce"]:12.5f}'
                 if measured[name] else '        n/a          n/a')
        print(f'{name:19s} {diag["D_prop"]:11.6f} {diag["D_struct"]:12.6f}'+extra)
    opt = optimized['optimization']
    print(f'Feature solver: {opt["status"]}, steps={opt["steps"]}, '
          f'raw objective gap <= {opt["raw_suboptimality_upper"]:.6g} (float64 feature problem only)')
    mean_test = [r['test'] for r in rows if r['variant'] == 'coarse_mean']
    grip_test = [r['test'] for r in rows if r['variant'] == 'grip']
    summary = []
    print('variant             val +/- sd       test +/- sd      delta vs coarse_mean [95% CI]')
    for name in artifacts:
        values = np.array([[r['val'], r['test']] for r in rows if r['variant'] == name])
        avg, sd = values.mean(0), values.std(0, ddof=1)
        paired = paired_stats(values[:, 1], mean_test)
        summary.append(dict(variant=name, val=float(avg[0]), val_std=float(sd[0]),
            test=float(avg[1]), test_std=float(sd[1]), versus_mean=paired,
            versus_grip=paired_stats(values[:, 1], grip_test)))
        print(f'{name:19s} {avg[0]:.2f} +/- {sd[0]:.2f}   {avg[1]:.2f} +/- {sd[1]:.2f}   '
              f'{paired["mean"]:+.2f} [{paired["low"]:+.2f}, {paired["high"]:+.2f}]')
    atomic_json(out/'summary.json', dict(results=summary, feature_diagnostics=feature_stats, student_diagnostics=measured,
        optimization=opt, caveat='Pointwise paired student-seed intervals, fixed partition/labels. '
        'Numerical feature-gap certificate is not a risk or generalization certificate.'))
    print('Full results: student_runs.csv | summary.json | runs/*.json (prediction/risk diagnostics)')


if __name__ == '__main__':
    main()
