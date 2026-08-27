from collections.abc import Mapping

import xarray as xr
import numpy as np

from .feature_registry import feature_registry
from k_onda.central import CoordInfo, Schema, type_registry as tr, AxisInfo, AxisKind
from .core import Transformer
from k_onda.utils import np_from_xr


class ExtractFeatures(Transformer):
    accepted_data_types = (xr.DataArray,)

    def __init__(self, *features, registry=feature_registry, group_by=None):
        self._validate_configuration(features, registry, group_by)

        self.features = features
        self.registry = registry
        self.group_by = group_by
        self.funcs = [self.registry[feature] for feature in self.features]

    def _validate_configuration(self, features, registry, group_by):
        if not features:
            raise ValueError(
                f"{self.format_call()}: provide at least one feature."
            )
        if any(not isinstance(feature, str) or not feature for feature in features):
            raise TypeError(
                f"{self.format_call()}: every feature name must be a non-empty "
                "string."
            )
        if len(set(features)) != len(features):
            raise ValueError(
                f"{self.format_call()}: feature names cannot be repeated."
            )
        if not isinstance(registry, Mapping):
            raise TypeError(
                f"{self.format_call()}: registry must be a mapping from feature "
                "names to callables."
            )

        unknown = [feature for feature in features if feature not in registry]
        if unknown:
            known = ", ".join(sorted(registry))
            raise ValueError(
                f"{self.format_call()}: unknown feature names: {unknown!r}. "
                f"Registered features: {known}."
            )
        noncallable = [
            feature for feature in features if not callable(registry[feature])
        ]
        if noncallable:
            raise TypeError(
                f"{self.format_call()}: registered features must be callable: "
                f"{noncallable!r}."
            )

        if group_by is not None and not (
            callable(group_by) or isinstance(group_by, str) and group_by
        ):
            raise TypeError(
                f"{self.format_call()}: group_by must be a non-empty string, a "
                "callable, or None."
            )

    def __call__(self, input):

        self._validate_input(input)

        if isinstance(input, tr.Collection):
            input = input.group_by(self.group_by)

        rows = [[func(val) for func in self.funcs] for val in input.values()]
        invalid_results = sorted(
            {
                type(result).__name__
                for row in rows
                for result in row
                if not isinstance(result, tr.Signal)
            }
        )
        if invalid_results:
            raise TypeError(
                f"{self.format_call()}: every registered feature must return a "
                f"Signal; received result types {invalid_results!r}."
            )

        flat_inputs = tuple(sig for row in rows for sig in row)

        return tr.IndexedSignal(
            inputs=flat_inputs,
            transform=None,
            transformer=self,
            data_schema=None,
            apply_kwargs=self._make_apply_kwargs(input, rows)
            
        )
    
    def make_output_schema(self, *input_schemas, key_spec):
        for input_schema in input_schemas:
            self._validate_data_schema(input_schema)

        return Schema(
            axes=[
                AxisInfo(name="index", kind=AxisKind.OBSERVATION_INDEX),
                AxisInfo(
                    name="feature",
                    kind=AxisKind.AXIS,
                    coords=(
                        CoordInfo(
                            name="feature",
                            scale="nominal",
                            ordering="unordered",
                        ),
                    ),
                ),
            ]
            )

    def _validate_data_schema(self, input_schema):
        super()._validate_data_schema(input_schema)
        if input_schema.dim_names:
            raise NotImplementedError(
                f"{self.format_call()}: assembling non-scalar feature results "
                "is not implemented yet."
            )

    def _make_apply_kwargs(self, input, rows):
        return {
            "keys": list(input.keys()),
            "n_rows": len(rows),
            "n_features": len(self.features),
        }

    def _validate_input(self, input):
        if not isinstance(
            input, (tr.SignalMap, tr.CollectionMap, tr.Collection)
        ):
            raise TypeError(
                f"{self.format_call()}: input must be a SignalMap, CollectionMap, "
                "or Collection."
            )
        if not len(input):
            raise ValueError(
                f"{self.format_call()}: cannot extract features from an empty "
                f"{type(input).__name__}."
            )

        if isinstance(input, tr.Collection):
            if self.group_by is None:
                raise ValueError(
                    f"{self.format_call()}: group_by is required for Collection "
                    "input."
                )
        elif self.group_by is not None:
            raise ValueError(
                f"{self.format_call()}: group_by applies only to Collection input; "
                f"received {type(input).__name__}."
            )

        if isinstance(input, tr.SignalMap):
            invalid_values = sorted(
                {
                    type(value).__name__
                    for value in input.values()
                    if not isinstance(value, tr.Signal)
                }
            )
            if invalid_values:
                raise TypeError(
                    f"{self.format_call()}: every SignalMap value must be a Signal; "
                    f"received {invalid_values!r}."
                )

        if isinstance(input, tr.CollectionMap) and any(
            not len(collection) for collection in input.values()
        ):
            raise ValueError(
                f"{self.format_call()}: CollectionMap groups cannot be empty."
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
