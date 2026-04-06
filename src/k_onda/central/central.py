
import pint
import pint_xarray

ureg = pint.UnitRegistry()
pint.set_application_registry(ureg)
pint_xarray.setup_registry(ureg)

SAMPLING_RATE = 30000 * ureg.Hz
LFP_SAMPLING_RATE = 2000 * ureg.Hz

ureg.define("raw_sample = second / 30000 = rs")
ureg.define("lfp_sample = second / 2000 = ls")


operations = {
    '==': lambda a, b: a == b,
    '<': lambda a, b: a < b,
    '>': lambda a, b: a > b,
    '<=': lambda a, b: a <= b,
    '>=': lambda a, b: a >= b,
    'in': lambda a, b: a in b,
    '!=': lambda a, b: a != b,
    'not in': lambda a, b: a not in b
    }





    
