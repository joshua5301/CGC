import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.initialization_study import _save_json
from src.risk_analysis import _save_csv
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student, _forward


def confusion_metrics(matrix):
    matrix = np.asarray(matrix)
    support, predicted, tp = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    recall = np.divide(tp, support, out=np.full(len(tp), np.nan), where=support > 0)
    precision = np.divide(tp, predicted, out=np.full(len(tp), np.nan), where=predicted > 0)
    overall = dict(accuracy=100 * tp.sum() / matrix.sum(), macro_recall=100 * np.nanmean(recall),
        class0_precision=100 * precision[0], class0_recall=100 * recall[0],
        predicted0_share=100 * predicted[0] / matrix.sum(), false_positive0=int(predicted[0] - tp[0]),
        error_1_to_0=int(matrix[1, 0]), error_3_to_0=int(matrix[3, 0]),
        error_13_to_0=int(matrix[1, 0] + matrix[3, 0]))
    classes = pd.DataFrame(dict(class_id=np.arange(len(tp)), support=support, predicted_count=predicted,
        correct=tp, recall=100 * recall, precision=100 * precision,
        accuracy_contribution=100 * tp / matrix.sum()))
    return overall, classes


def run_grip_confusion_study(source_dir, output_dir, datasets, dropouts,
                              student_seeds=tuple(range(300, 310)), partition_seeds=(0, 1234),
                              epochs=1000, eval_every=10, hidden=256, lr=.01, weight_decay=.0005,
                              device='cuda', case_dirs=None):
    if len(partition_seeds) != 2 or len(set(partition_seeds)) != 2 or not student_seeds or len(set(student_seeds)) != len(student_seeds):
        raise ValueError('Require two distinct partition seeds and unique nonempty student seeds')
    source, out, device = Path(source_dir), Path(output_dir), torch.device(device)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    cases = [Path(p) for p in case_dirs] if case_dirs else sorted(p.parent for p in source.glob('*/protocol.json'))
    selected = {}
    for case in cases:
        p = json.loads((case / 'protocol.json').read_text(encoding='utf-8'))
        key = (p['dataset'], p['ratio'])
        if key[0] not in datasets or key[1] not in datasets[key[0]]:
            continue
        if key in selected:
            raise ValueError(f'Multiple cases for {key}; specify case_dirs')
        selected[key] = (case, p)
    expected = {(d, r) for d, ratios in datasets.items() for r in ratios}
    if set(selected) != expected:
        raise ValueError(f'Missing saved cases: {expected - set(selected)}')
    protocol = dict(revision=revision, torch=str(torch.__version__),
        sources=[dict(folder=str(c), protocol=p) for c, p in selected.values()],
        dropouts={f'{d}:{r}': v for (d, r), v in dropouts.items()},
        student_seeds=list(student_seeds), partition_seeds=list(partition_seeds),
        epochs=epochs, eval_every=eval_every, hidden=hidden, lr=lr, weight_decay=weight_decay,
        layers=2, loss='uniform_soft_ce')
    root = out / _fingerprint(protocol)
    root.mkdir(parents=True, exist_ok=True)
    _save_json(root / 'protocol.json', protocol)
    _save_json(out / 'latest.json', dict(folder=str(root)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden, loss_weighting='uniform')
    records, class_records, matrices = [], [], []
    for (dataset, ratio), (case, p) in selected.items():
        train, mask, validation, testing, H = _prepare_dataset(dataset, p['data_dir'], device)
        params = dict(dropout=dropouts[(dataset, ratio)], lr=lr, weight_decay=weight_decay)
        for ps in partition_seeds:
            artifact = torch.load(case / f'{ps}.pt', map_location=device, weights_only=True)
            if not artifact['converged'] or artifact['nodes'] != p['budget']:
                raise ValueError('Comparison requires converged partitions at the requested budget')
            folder = root / f'{dataset}_{ratio:g}_{ps}'
            folder.mkdir(exist_ok=True)
            for ss in tqdm(student_seeds, desc=f'{dataset} {ratio:g} partition={ps}'):
                path = folder / f'{ss}.pt'
                if path.exists():
                    saved = torch.load(path, map_location='cpu', weights_only=True)
                else:
                    val, _, epoch, model = _train_student(artifact['x'], artifact['y'], validation,
                        params, ss, settings, return_model=True)
                    saved = dict(best_epoch=epoch, best_validation=val)
                    with torch.no_grad():
                        for split, (graph, selection) in [('valid', validation), ('test', testing)]:
                            logits = _forward(model, graph['x'], graph['adj'])
                            target = graph['y']
                            if selection is not None:
                                logits, target = logits[selection], target[selection]
                            saved[split] = dict(prediction=logits.argmax(1).cpu(),
                                                logits=logits.cpu(), target=target.cpu())
                    temporary = path.with_suffix('.tmp')
                    torch.save(saved, temporary)
                    temporary.replace(path)
                    del model
                K = artifact['y'].shape[1]
                for split in ('valid', 'test'):
                    data = saved[split]
                    cm = torch.bincount(data['target'] * K + data['prediction'], minlength=K * K).reshape(K, K).numpy()
                    overall, classes = confusion_metrics(cm)
                    identity = dict(dataset=dataset, ratio=ratio, partition_seed=ps, student_seed=ss, split=split)
                    records.append(dict(**identity, best_epoch=saved['best_epoch'], **overall))
                    class_records.append(classes.assign(**identity))
                    matrices.append(pd.DataFrame(dict(true_class=np.repeat(np.arange(K), K),
                        predicted_class=np.tile(np.arange(K), K), count=cm.ravel())).assign(**identity))
        del train, mask, validation, testing, H, artifact
        torch.cuda.empty_cache()
    runs, classes, confusion = pd.DataFrame(records), pd.concat(class_records), pd.concat(matrices)
    keys = ['dataset', 'ratio', 'split']
    metrics = list(confusion_metrics(np.eye(6, dtype=int))[0])
    groups = keys + ['partition_seed']
    summary = runs.groupby(groups)[metrics].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    pair_keys = keys + ['student_seed']
    baseline = runs[runs.partition_seed == partition_seeds[0]].set_index(pair_keys)[metrics]
    reference = runs[runs.partition_seed == partition_seeds[1]].set_index(pair_keys)[metrics]
    paired = (reference - baseline).add_prefix('delta_').reset_index()
    paired_summary = paired.groupby(keys)[[f'delta_{m}' for m in metrics]].agg(['mean', 'std'])
    paired_summary.columns = ['_'.join(c) for c in paired_summary.columns]
    report = dict(runs=runs, summary=summary, paired=paired, paired_summary=paired_summary.reset_index(),
        classes=classes, class_summary=classes.groupby(groups + ['class_id']).agg(
            support=('support', 'first'), recall=('recall', 'mean'), precision=('precision', 'mean'),
            accuracy_contribution=('accuracy_contribution', 'mean')).reset_index(),
        confusion=confusion, confusion_mean=confusion.groupby(groups + ['true_class', 'predicted_class'])['count'].mean().reset_index())
    for name, frame in report.items():
        _save_csv(frame, root / f'{name}.csv')
    report['folder'] = str(root)
    return report
