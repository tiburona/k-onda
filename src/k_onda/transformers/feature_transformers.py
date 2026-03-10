import xarray as xr
import numpy as np
from functools import partial

from .feature_registry import feature_registry
from k_onda.central import Schema
from .core import Transform, Transformer



class ExtractFeatures(Transformer):

    def __init__(self, *features, registry=feature_registry, group_by=None):
        self.features = features
        self.registry = registry
        self.group_by = group_by
        self.funcs = [self.registry[feature] for feature in self.features]


    def __call__(self, inputs):
        from k_onda.signals import IndexedSignal
        from k_onda.sources import Collection
        
        input = inputs[0]
        self._validate_input(input)

        if isinstance(input, Collection):
            input = input.group_by(self.group_by)

        arrays = [[func(val) for func in self.funcs] 
                  for val in input.values()]

        transform = self._get_transform(list(input.keys()), arrays)

        return IndexedSignal(
            inputs=(input,),
            transform=transform,
            data_schema=Schema('index', 'features')
        )
    
    def _get_transform(self, keys, arrays):
        return partial(self._apply, keys, arrays)
    
    def _validate_input(self, input):
        from k_onda.sources import SignalMap, CollectionMap, Collection

        if isinstance(input, Collection) and self.group_by is None:
            raise ValueError("If ExtractFeatures is called on Collection, group_on must"
            "be defined.")

        if not isinstance(input, (SignalMap, CollectionMap, Collection)):
            raise ValueError("ExtractFeatures is only defined on SignalMap, CollectionMap,"
            "and Collection.")
    
    def _apply(self, keys, arrays):
        
        values = [[sig.data for sig in row] for row in arrays]

        return xr.DataArray(
            np.array(values),
            dims = ('index', 'feature'),
            coords = {
                'index': keys,
                'feature': list(self.features)
            }

        )
        
        