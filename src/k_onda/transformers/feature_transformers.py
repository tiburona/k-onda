import xarray as xr
import numpy as np
from functools import partial

from .feature_registry import feature_registry
from k_onda.central import Schema
from .core import Transform, Transformer



class ExtractFeatures(Transformer):

    def __init__(self, *features, registry=feature_registry):
        self.features = features
        self.registry = registry
        self.funcs = [self.registry[feature] for feature in self.features]


    def __call__(self, map_inputs):
        from k_onda.signals import IndexedSignal
        
        map_input = map_inputs[0]
        self._validate_input(map_input)

        arrays = [[func(val) for func in self.funcs] 
                  for val in map_input.values()]

        transform = self._get_transform(list(map_input.keys()), arrays)

        return IndexedSignal(
            inputs=(map_input,),
            transform=transform,
            data_schema=Schema('index', 'features')
        )
    
    def _get_transform(self, keys, arrays):
        return partial(self._apply, keys, arrays)
    
    def _validate_input(self, map_input):
        from k_onda.sources import SignalMap, CollectionMap
        if not isinstance(map_input, (SignalMap, CollectionMap)):
            raise ValueError("ExtractFeatures needs a map-type input.")
    
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
        
        