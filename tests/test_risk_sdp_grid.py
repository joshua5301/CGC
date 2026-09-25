from pathlib import Path

import pandas as pd
import torch

import src.risk_sdp_grid as module


def test_grid_reuses_partitions_and_defers_test_until_selection(tmp_path, monkeypatch):
    calls, partitions = [], []
    monkeypatch.setattr(module, '_prepare_dataset', lambda *args: (None, None, 'valid', 'test', None))

    def study(configs, output_dir, **kwargs):
        config = configs[0]
        partitions.append(config.copy())
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for method in ('sdp', 'surrogate'):
            torch.save(dict(x=torch.zeros(2, 3), y=torch.full((2, 2), .5)), folder / f'{method}_0.pt')
            rows.append(dict(method=method, seed=0, folder=str(folder), J_final=.5,
                lower_bound=.4, gap_upper=.1, solver_status='optimal', sdp_seconds=1., clusters=2))
        return dict(summary=pd.DataFrame(rows))

    def train(x, y, validation, params, seed, settings, testing=None):
        calls.append((params['gamma'], params['dropout'], seed, testing))
        if testing is not None:
            assert (tmp_path / 'latest.json').exists()
            assert list(tmp_path.glob('*/selected.csv'))
        return .5 + params['gamma'] / 10 + params['dropout'] / 100, .6 if testing else None, 10

    monkeypatch.setattr(module, 'run_sdp_study', study)
    monkeypatch.setattr(module, '_train_student', train)
    grid = dict(teacher_kernel=['relu'], gamma=[.01, .1], T=[1.], basis=[10], B=[1.],
                dropout=[.1, .9], lr=[.01], weight_decay=[.0005])
    options = dict(dataset='cora', ratio=.052, grid=grid, output_dir=tmp_path,
                   partition_seeds=[0], search_seeds=[0, 1], final_seeds=[100, 101], device='cpu')
    report = module.run_sdp_grid(**options)
    assert len(partitions) == 2
    assert len(report['search']) == 8
    assert len(calls) == 20
    assert all(call[3] is None for call in calls[:16])
    assert all(call[:2] == (.1, .9) and call[2] in (100, 101) and call[3] == 'test' for call in calls[16:])
    assert set(report['summary'].gamma) == {.1}
    assert set(report['summary'].dropout) == {.9}
    module.run_sdp_grid(**options)
    assert len(calls) == 20
