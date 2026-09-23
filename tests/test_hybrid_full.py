import pytest

from hybrid_full import choose, grid, TEMPERATURES, MUS, LAMBDAS


def test_full_grid_size_and_controls():
    assert len(grid(TEMPERATURES, True))*len(grid(MUS))*(len(grid(LAMBDAS))+1) == 504
    assert 0. in grid(MUS) and 0. in grid(LAMBDAS)
    assert 2. in grid(TEMPERATURES) and .5 in grid(MUS)
    assert grid('1,0,1,0.1') == [0., .1, 1.]


@pytest.mark.parametrize('value,positive', [('nan', False), ('inf', False), ('-1', False), ('0', True), ('', False)])
def test_reject_invalid_grid(value, positive):
    with pytest.raises(ValueError):
        grid(value, positive)


def test_group_selection_uses_only_validation_with_deterministic_ties():
    rows = [
        dict(id='g1', method='grip', T=2., mu=.5, lam=0., val=82., test=10.),
        dict(id='g2', method='grip', T=.5, mu=.1, lam=0., val=81., test=100.),
        dict(id='z', method='hybrid', T=2., mu=.5, lam=0., val=83., test=1.),
        dict(id='p1', method='hybrid', T=2., mu=.5, lam=.1, val=80., test=100.),
        dict(id='p2', method='hybrid', T=2., mu=.5, lam=.01, val=80., test=0.),
    ]
    result = choose(list(reversed(rows)))
    assert {k: r['id'] for k, r in result.items()} == {'grip': 'g1', 'zero': 'z', 'positive': 'p2'}
    assert set(choose(rows[:3])) == {'grip', 'zero'}
