from collections.abc import MutableMapping
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


@runtime_checkable
class SignalLike(Protocol):
    data: ...


class Schema:
    def __init__(self, *dims, selectable_dims=None):
        dims = list(dims)
        self.dims = set(dims)
        if selectable_dims is None:
            self._selectable_dims = set()
        else:
            self._selectable_dims = set(selectable_dims)
        self.selectable_dims = set(dims) | self._selectable_dims


class DatasetSchema(MutableMapping):
    def __init__(self, key_schemas):
        self.key_schemas = key_schemas  # dict[str, Schema]

    def __getitem__(self, key):
        return self.key_schemas[key]

    def __setitem__(self, key, value):
        self.key_schemas[key] = value

    def __delitem__(self, key):
        del self.key_schemas[key]

    def __iter__(self):
        return iter(self.key_schemas)

    def __len__(self):
        return len(self.key_schemas)

    @property
    def dims(self):
        # union: for Selector, a dim is 'available' here if any key has it
        return set.union(*(s.dims for s in self.key_schemas.values()))
    
    @property
    def selectable_dims(self):
        return set.union(*(s.selectable_dims for s in self.key_schemas.values()))
    
    def replace_key(self, key, new_schema):
        return DatasetSchema({**self.key_schemas, key: new_schema})
    
    def add_key(self, key, new_schema):
        if key in self.key_schemas:
            raise ValueError(f"Key '{key}' already exists in DatasetSchema")
        return DatasetSchema({**self.key_schemas, key: new_schema})
    
