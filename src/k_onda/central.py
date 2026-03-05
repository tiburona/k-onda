import pint
import pint_xarray
from typing import Protocol, runtime_checkable

ureg = pint.UnitRegistry()
pint.set_application_registry(ureg)
pint_xarray.setup_registry(ureg)

SAMPLING_RATE = 30000 * ureg.Hz
LFP_SAMPLING_RATE = 2000 * ureg.Hz

ureg.define("raw_sample = second / 30000 = rs")
ureg.define("lfp_sample = second / 2000 = ls")


@runtime_checkable
class SignalLike(Protocol):
    data: ...


class Schema:
    def __init__(self, dims):
        self.dims = dims