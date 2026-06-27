import xarray as xr
import numpy as np

from .feature_registry import feature_registry
from k_onda.central import Schema, type_registry, AxisInfo, AxisKind
from .core import Transformer
from k_onda.utils import np_from_xr


class ExtractFeatures(Transformer):
    def __init__(self, *features, registry=feature_registry, group_by=None):
        self.features = features
        self.registry = registry
        self.group_by = group_by
        self.funcs = [self.registry[feature] for feature in self.features]

    def __call__(self, inputs):

        input = inputs[0]
        self._validate_input(input)

        if isinstance(input, type_registry.Collection):
            input = input.group_by(self.group_by)

        rows = [[func(val) for func in self.funcs] for val in input.values()]
        flat_inputs = tuple(sig for row in rows for sig in row)

        return type_registry.IndexedSignal(
            inputs=flat_inputs,
            transform=None,
            transformer=self,
            data_schema=None,
            apply_kwargs=self._make_apply_kwargs(input, rows)
            
        )
    
    def make_output_schema(self, *input_schemas, key_spec):
        return Schema(
            axes=[
                AxisInfo(name="index", kind=AxisKind.OBSERVATION_INDEX),
                AxisInfo(name="feature", kind=AxisKind.ORDINAL_INDEX),
            ]
            )

    def _make_apply_kwargs(self, input, rows):
        return {
            "keys": list(input.keys()),
            "n_rows": len(rows),
            "n_features": len(self.features),
        }

    def _validate_input(self, input):

        if isinstance(input, type_registry.Collection) and self.group_by is None:
            raise ValueError(
                "If ExtractFeatures is called on Collection, group_by mustbe defined."
            )

        if not isinstance(
            input, (type_registry.SignalMap, type_registry.CollectionMap, type_registry.Collection)
        ):
            raise ValueError(
                "ExtractFeatures is only defined on SignalMap, CollectionMap,"
                "and Collection."
            )

    def _apply(self, *feature_data, keys, n_rows, n_features):

        feature_units = {}
        values = []

        idx = 0

        for i in range(n_rows):
            row_vals = []
            for j in range(n_features):
                d = feature_data[idx]
                idx += 1

                arr, units = np_from_xr(d)
                if i == 0:
                    feature_units[self.features[j]] = units
                row_vals.append(arr)
            values.append(row_vals)

        return xr.DataArray(
            np.array(values),
            dims=("index", "feature"),
            coords={"index": keys, "feature": list(self.features)},
        ).assign_attrs(
            {"feature_units": feature_units, "transformer": "extractfeatures"}
        )
