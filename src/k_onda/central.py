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


@runtime_checkable
class SignalLike(Protocol):
    data: ...


class Schema:
    def __init__(self, dims):
        self.dims = dims


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
    
    def replace_key(self, key, new_schema):
        return DatasetSchema({**self.key_schemas, key: new_schema})
    
    def add_key(self, key, new_schema):
        if key in self.key_schemas:
            raise ValueError(f"Key '{key}' already exists in DatasetSchema")
        return DatasetSchema({**self.key_schemas, key: new_schema})
    
