from collections.abc import MutableMapping


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