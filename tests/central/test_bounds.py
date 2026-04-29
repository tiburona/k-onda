import pint
import pytest

from k_onda.central import DimBounds, DimPair


def test_dim_pair_lo_hi_properties():
    bounds = DimPair([1, 2])

    assert bounds.lo == 1
    assert bounds.hi == 2

    bounds.lo = 3
    bounds.hi = 4

    assert bounds[0] == 3
    assert bounds[1] == 4


def test_dim_bounds_lo_defaults_to_only_dim():
    bounds = DimBounds({'time': DimPair([1, 2])})

    assert bounds.lo == 1


def test_dim_bounds_lo_preserves_multi_dim_mapping():
    bounds = DimBounds({
        'time': DimPair([1, 2]),
        'frequency': DimPair([3, 4]),
    })

    assert bounds.lo == {
        'time': 1,
        'frequency': 3,
    }


def test_dim_bounds_lo_preserves_list_shape():
    bounds = DimBounds({'time': [DimPair([1, 2]), DimPair([3, 4])]})

    assert bounds.lo == [1, 3]


def test_dim_bounds_add_adds_both_bounds():
    base = DimBounds({'time': DimPair([10, 20])})
    other = DimBounds({'time': DimPair([1, 2])})

    result = base + other

    assert result['time'][0] == 11
    assert result['time'][1] == 22
    assert base['time'][0] == 10
    assert base['time'][1] == 20


def test_dim_bounds_accumulate_ignores_unmatched_dims():
    base = DimBounds({'time': DimPair([10, 20])})
    other = DimBounds({'frequency': DimPair([1, 2])})

    base.accumulate(other)

    assert list(base) == ['time']


def test_dim_bounds_add_lo_adds_only_lower_bound_with_existing_dim_matching():
    base = DimBounds({
        'time': DimPair([10, 20]),
        'frequency': DimPair([100, 200]),
    })
    other = DimBounds({
        'time': DimPair([1, 2]),
        'frequency': DimPair([3, 4]),
    })

    result = base.add_lo(other)

    assert result['time'][0] == 11
    assert result['time'][1] == 20
    assert result['frequency'][0] == 103
    assert result['frequency'][1] == 200


def test_dim_bounds_add_lo_mutates_in_place():
    base = DimBounds({'time': DimPair([10, 20])})
    other = DimBounds({'time': DimPair([1, 2])})

    base.add_lo(other)

    assert base['time'][0] == 11
    assert base['time'][1] == 20


def test_dim_bounds_add_lo_applies_single_source_to_list_target():
    base = DimBounds({'time': [DimPair([10, 20]), DimPair([30, 40])]})
    other = DimBounds({'time': DimPair([1, 2])})

    base.add_lo(other)

    assert base['time'][0][0] == 11
    assert base['time'][0][1] == 20
    assert base['time'][1][0] == 31
    assert base['time'][1][1] == 40


def test_dim_bounds_add_lo_expands_single_target_for_list_source():
    base = DimBounds({'time': DimPair([10, 20])})
    other = DimBounds({'time': [DimPair([1, 2]), DimPair([3, 4])]})

    base.add_lo(other)

    assert base['time'][0][0] == 11
    assert base['time'][0][1] == 20
    assert base['time'][1][0] == 13
    assert base['time'][1][1] == 20


def test_dim_bounds_add_lo_uses_metadim_equivalence():
    metadims = {'lfp_time': 'time', 'time': 'time'}
    base = DimBounds(
        {'lfp_time': DimPair([10, 20])},
        metadim_of=lambda dim: metadims[dim],
    )
    other = DimBounds({'time': DimPair([1, 2])})

    base.add_lo(other)

    assert base['lfp_time'][0] == 11
    assert base['lfp_time'][1] == 20


def test_dim_bounds_add_lo_preserves_unitful_upper_bound():
    ureg = pint.UnitRegistry()
    base = DimBounds({'time': DimPair([10 * ureg.s, 20 * ureg.s])})
    other = DimBounds({'time': DimPair([1 * ureg.s, 2 * ureg.s])})

    base.add_lo(other)

    assert base['time'][0] == 11 * ureg.s
    assert base['time'][1] == 20 * ureg.s


def test_dim_bounds_add_lo_raises_for_unmatched_dim():
    base = DimBounds({'time': DimPair([10, 20])})
    other = DimBounds({'frequency': DimPair([1, 2])})

    with pytest.raises(ValueError, match='dim frequency not found'):
        base.add_lo(other)
