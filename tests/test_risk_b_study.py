import pandas as pd

from src.risk_b_study import select_best_b


def test_selects_separate_b_by_validation_with_stable_ties():
    grid = pd.DataFrame(dict(dataset=['citeseer'] * 3, ratio=[.009] * 3,
                             B=[.5, 1., 2.], B_base=[1.] * 3, B_factor=[.5, 1., 2.]))
    summary = pd.concat([grid.assign(init='surrogate', validation_mean=[73., 76., 75.]),
                         grid.assign(init='split', validation_mean=[77., 74., 77.])], ignore_index=True)
    summary = summary.drop(columns=['B_base', 'B_factor'])
    selected = select_best_b(summary, grid).set_index('init')
    assert selected.loc['surrogate', 'B'] == 1.
    assert selected.loc['split', 'B'] == .5
    assert not selected.loc['surrogate', 'grid_boundary']
    assert selected.loc['split', 'grid_boundary']
