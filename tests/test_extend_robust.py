import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from extend_robust import CAPS, paired_stats, report


def test_paired_interval_uses_differences():
    baseline = np.arange(10, dtype=float)
    result = paired_stats(baseline+.5, baseline)
    assert result['mean'] == result['low'] == result['high'] == .5
    assert result['wins'] == 10
    values = baseline + np.linspace(-1, 1, 10)
    regular = paired_stats(values, baseline)
    adjusted = paired_stats(values, baseline, comparisons=7)
    assert abs(regular['mean']) < 1e-12
    assert adjusted['low'] < regular['low'] < 0 < regular['high'] < adjusted['high']


def test_report_separates_old_selection_and_new_seeds(tmp_path, capsys):
    (tmp_path/'runs').mkdir()
    manifest = []
    for i, cap in enumerate(CAPS):
        config = dict(radius_cap=cap)
        entry = dict(id=str(i), config=config)
        manifest.append(entry)
        for seed in range(5):
            # Largest cap wins old validation; another wins fresh validation/test.
            val = .7+i*.01 if seed < 3 else .9-i*.01
            test = .8 if i == 0 else .82
            row = dict(student_seed=seed, config=config, val=val, test=test)
            (tmp_path/'runs'/f'{i}_d0.9_r{seed}.json').write_text(json.dumps(row))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    report(tmp_path, 5)
    result = json.loads((tmp_path/'confirmation.json').read_text())
    assert result['selected_by_old_validation'] == CAPS[-1]
    assert result['fresh_seeds'] == [3, 4]
    assert abs(result['rows'][1]['paired_test']['mean']-2) < 1e-9
