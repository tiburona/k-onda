from collections.abc import Iterable

import pint


DIM_DEFAULT_UNITS = {'time': 's', 'frequency': 'Hz'}



def is_unitful(value):
    if isinstance(value, Iterable):
        if all([isinstance(v, pint.Quantity) for v in value]):
            return True
        elif all([not isinstance(v, pint.Quantity) for v in value]):
            return False
        else:
            raise ValueError("Array with some unitful and some plain values " \
            "passed to is_unitful")
    return isinstance(value, pint.Quantity)


def w_units(value, dim=None, units=None, ureg=None):
    if is_unitful(value):
        return value
    if units is None:
        if dim in DIM_DEFAULT_UNITS:
            units = DIM_DEFAULT_UNITS[dim]
        else:
            raise ValueError("units were not provided")
    if ureg is None:
        raise ValueError("ureg was not provided")
    if isinstance(value, Iterable):
        return [v * ureg(units) for v in value]
    else:
        return value * ureg(units)