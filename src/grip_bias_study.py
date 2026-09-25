from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.grip_confusion_study import run_grip_confusion_study, confusion_metrics
from src.initialization_study import _save_json
from src.risk_analysis import _save_csv
from src.risk_experiment import _fingerprint


def suppress_class0(logits, delta):
    adjusted = logits.clone()
    adjusted[:, 0] -= delta
    return adjusted.argmax(1)


def select_delta(validation_runs, deltas):
    curve = []
    for delta in sorted(set(deltas)):
        values = [100 * float(suppress_class0(r['logits'], delta).eq(r['target']).double().mean())
                  for r in validation_runs]
        curve.append(dict(delta=float(delta), validation=float(np.mean(values))))
    return max(curve, key=lambda r: (r['validation'], -r['delta']))['delta'], curve


def run_grip_bias_study(source_dir, output_dir, datasets, dropouts, deltas,
                        calibration_seeds=tuple(range(300, 305)), final_seeds=tuple(range(400, 410)),
                        **settings):
    deltas = sorted(set(float(d) for d in deltas))
    if not deltas or deltas[0] != 0 or not np.isfinite(deltas).all():
        raise ValueError('Require finite nonnegative deltas including zero')
    if not calibration_seeds or not final_seeds or set(calibration_seeds) & set(final_seeds):
        raise ValueError('Require disjoint nonempty calibration and final student seeds')
    raw = run_grip_confusion_study(source_dir, Path(output_dir) / 'models', datasets, dropouts,
        student_seeds=tuple(calibration_seeds) + tuple(final_seeds), partition_seeds=(0, 1234), **settings)
    model_root = Path(raw['folder'])
    protocol = dict(model_folder=str(model_root), deltas=deltas,
                    calibration_seeds=list(calibration_seeds), final_seeds=list(final_seeds))
    root = Path(output_dir) / f'bias_{_fingerprint(protocol)}'
    root.mkdir(parents=True, exist_ok=True)
    _save_json(root / 'protocol.json', protocol)
    curves, choices = [], []
    for dataset, ratios in datasets.items():
        for ratio in ratios:
            for ps in (0, 1234):
                folder = model_root / f'{dataset}_{ratio:g}_{ps}'
                validation_runs = [torch.load(folder / f'{s}.pt', weights_only=True)['valid'] for s in calibration_seeds]
                delta, curve = select_delta(validation_runs, deltas)
                identity = dict(dataset=dataset, ratio=ratio, partition_seed=ps)
                curves.extend(dict(**identity, **r) for r in curve)
                choices.append(dict(**identity, delta=delta, at_upper_boundary=delta == max(deltas),
                    calibration_val=next(r['validation'] for r in curve if r['delta'] == delta)))
    selected = pd.DataFrame(choices)
    _save_csv(selected, root / 'selected.csv')
    records, class_records = [], []
    for choice in choices:
        dataset, ratio, ps = choice['dataset'], choice['ratio'], choice['partition_seed']
        folder = model_root / f'{dataset}_{ratio:g}_{ps}'
        for seed in final_seeds:
            saved = torch.load(folder / f'{seed}.pt', weights_only=True)
            for correction, delta in [('raw', 0.), ('suppressed', choice['delta'])]:
                for split in ('valid', 'test'):
                    data = saved[split]
                    pred = suppress_class0(data['logits'], delta)
                    K = data['logits'].shape[1]
                    cm = torch.bincount(data['target'] * K + pred, minlength=K * K).reshape(K, K).numpy()
                    metrics, classes = confusion_metrics(cm)
                    identity = dict(dataset=dataset, ratio=ratio, partition_seed=ps,
                        student_seed=seed, correction=correction, delta=delta, split=split)
                    records.append(dict(**identity, **metrics))
                    class_records.append(classes.assign(**identity))
    runs, classes = pd.DataFrame(records), pd.concat(class_records, ignore_index=True)
    keys = ['dataset', 'ratio', 'split']
    metric_names = list(metrics)
    summary = runs.groupby(keys + ['partition_seed', 'correction', 'delta'])[metric_names].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    pairs = []
    for (dataset, ratio, split), group in runs.groupby(keys):
        def values(ps, correction):
            return group[(group.partition_seed == ps) & (group.correction == correction)].set_index('student_seed')[metric_names]
        comparisons = {
            'seed0_suppressed_minus_raw': values(0, 'suppressed') - values(0, 'raw'),
            'seed1234_raw_minus_seed0_raw': values(1234, 'raw') - values(0, 'raw'),
            'seed1234_raw_minus_seed0_suppressed': values(1234, 'raw') - values(0, 'suppressed'),
            'seed1234_suppressed_minus_seed0_suppressed': values(1234, 'suppressed') - values(0, 'suppressed'),
        }
        for comparison, frame in comparisons.items():
            pairs.append(frame.add_prefix('delta_').reset_index().assign(
                dataset=dataset, ratio=ratio, split=split, comparison=comparison))
    paired = pd.concat(pairs, ignore_index=True)
    paired_summary = paired.groupby(keys + ['comparison'])[[f'delta_{m}' for m in metric_names]].agg(['mean', 'std'])
    paired_summary.columns = ['_'.join(c) for c in paired_summary.columns]
    report = dict(selected=selected, calibration_curve=pd.DataFrame(curves), runs=runs,
        summary=summary.reset_index(), paired=paired, paired_summary=paired_summary.reset_index(), classes=classes)
    for name, frame in report.items():
        _save_csv(frame, root / f'{name}.csv')
    report['folder'] = str(root)
    return report
