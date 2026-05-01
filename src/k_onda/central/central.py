import pint
import pint_xarray


ureg = pint.UnitRegistry()
pint.set_application_registry(ureg)
pint_xarray.setup_registry(ureg)


operations = {
    "==": lambda a, b: a == b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "in": lambda a, b: a in b,
    "!=": lambda a, b: a != b,
    "not in": lambda a, b: a not in b,
}
