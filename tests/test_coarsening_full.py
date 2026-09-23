import numpy as np
import pytest

from coarsening_full import choose, paired_values, VARIANTS


def rows():
    result = []
    for variant in VARIANTS:
        for index, (gamma, temperature, mu) in enumerate(((.01, 2., .5), (.1, 5., 2.))):
            result.append(dict(id=f'{variant}{index}', variant=variant, gamma=gamma,
                T=temperature, mu=mu, val=80.+(index == (variant == 'coarse_optimized')),
                test=99.-index))
    return result


def test_selection_never_uses_test_and_preserves_matched_settings():
    data = rows()
    selected = choose(data)
    assert selected['coarse_mean']['id'] == 'coarse_mean0'
    assert selected['coarse_optimized']['id'] == 'coarse_optimized1'
    assert selected['mean_at_optimized']['id'] == 'coarse_mean1'
    assert selected['optimized_at_mean']['id'] == 'coarse_optimized0'
    expected = {label: row['id'] for label, row in selected.items()}
    for row in data:
        row['test'] = -row['test']
    assert {label: row['id'] for label, row in choose(data[::-1]).items()} == expected


def test_exact_validation_ties_have_deterministic_resolution():
    data = rows()
    for row in data:
        row['val'] = 80.
    result = choose(data[::-1])
    assert all(row['gamma'] == .01 for row in result.values())
    assert result['mean_at_optimized']['id'] == result['coarse_mean']['id']
    assert result['optimized_at_mean']['id'] == result['coarse_optimized']['id']


def test_pairing_uses_explicit_seed_identity_and_rejects_missing_or_duplicates():
    data = [dict(id='a', stage='confirmation', seed=s, val=s+.5, test=s+1.) for s in (14, 13)]
    data.append(dict(id='a', stage='selection', seed=0, val=99., test=99.))
    assert np.array_equal(paired_values(data, 'a', [13, 14]), [[13.5, 14.], [14.5, 15.]])
    with pytest.raises(ValueError):
        paired_values(data, 'a', [13, 14, 15])
    with pytest.raises(ValueError):
        paired_values(data+[data[0]], 'a', [13, 14])
