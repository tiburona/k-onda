from collections.abc import Iterable

import pint
import numpy as np

from k_onda.central import types

DIM_DEFAULT_UNITS = {'time': 's', 'frequency': 'Hz'}



def is_unitful(value):
    if isinstance(value, (types.Locus, types.LocusSet)):
        return True
    if isinstance(value, pint.Quantity):
        return True
    if isinstance(value, Iterable):
        if all([isinstance(v, pint.Quantity) for v in value]):
            return True
        elif all([not isinstance(v, pint.Quantity) for v in value]):
            return False
        else:
            raise ValueError("Array with some unitful and some plain values " \
            "passed to is_unitful")
    return False


def w_units(value, dim=None, units=None, ureg=None):
    if is_unitful(value):
        return value
    if units is None:
        if dim in DIM_DEFAULT_UNITS:
            units = DIM_DEFAULT_UNITS[dim]
        else:
            raise ValueError("units were not provided")
    if ureg is None:
        ureg = pint.get_application_registry()
    if isinstance(units, str):
        units = ureg(units)
    if isinstance(value, Iterable):
        return np.array(value) * units
    else:
        return value * units