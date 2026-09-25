import pandas as pd

from src.grip_seed_costs import compare_costs


def test_comparison_excludes_unconverged_and_wrong_budget():
    rows = [dict(dataset='citeseer', ratio=.009, partition_seed=seed, final_J=value,
        final_feature=value / 2, final_weighted_kl=value / 2, converged=converged,
        nodes=nodes, requested_nodes=30) for seed, value, converged, nodes in
        [(1234, 1., True, 30), (0, .9, True, 30), (1, 1.0000001, True, 30),
         (2, 1.1, True, 30), (3, .8, False, 30), (4, .7, True, 29)]]
    costs, summary = compare_costs(pd.DataFrame(rows))
    assert costs.comparison.tolist() == ['reference', 'lower', 'near_equal', 'higher', 'excluded', 'excluded']
    assert summary['reference_rank'] == 2
    assert summary['lower'] == summary['near_equal'] == summary['higher'] == 1
    assert summary['excluded_seeds'] == 2
    rows[0]['converged'] = False
    _, summary = compare_costs(pd.DataFrame(rows))
    assert not summary['reference_valid']
    assert summary['comparable_seeds'] == 0
    assert summary['reference_rank'] is None
