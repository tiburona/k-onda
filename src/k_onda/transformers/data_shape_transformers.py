
from copy import deepcopy
from functools import partial
import xarray as xr

from .core import Transformer, Transform
from k_onda.central import Schema, DatasetSchema
from .feature_registry import feature_registry


class StackSignals(Transformer):
    name = 'stack_signals'

    """Concatenate component signals so downstream calculations can be vectorized."""

    def __init__(self, dim=None):
        self.dim = dim

    def output_schema(self, *input_schemas):
        stacking_dim = self.dim or 'members'
        if isinstance(input_schemas[0], DatasetSchema):
            return DatasetSchema({
                key: Schema(*schema.dims, stacking_dim, selectable_dims=schema._selectable_dims)
                for key, schema in input_schemas[0].items()
            })

        dims = set(input_schemas[0].dims)
        dims.add(self.dim or 'members')
        return Schema(dims)

    def __call__(self, collection):
        from ..signals import SignalStack
        input_schemas = [s.data_schema for s in collection.signals]
        output_schema = self.output_schema(*input_schemas)
        transform = self._get_transform()
        return SignalStack(output_schema, collection=collection, transform=transform, 
                           transformer=self)

    def _get_transform(self):
        return Transform(self._apply)

    def _gather_datasets(self, signals):
        keys = signals[0].data.keys()
        data = {}
        boundaries = [0]

        for i, key in enumerate(keys):
            arrays = []
            for signal in signals:
                arr = signal.data[key]
                arrays.append(arr)
                increment = arr.sizes[self.dim] if self.dim else 1
                if i == 0:
                    boundaries.append(boundaries[-1] + increment)

            data[key] = xr.concat(
                arrays, dim=self.dim or 'members', combine_attrs='no_conflicts'
            )

        dataset = xr.Dataset(data)
        dataset.attrs['boundaries'] = boundaries
        dataset.attrs['stack_dim'] = self.dim

        return dataset

    def _gather_arrays(self, signals):
        arrays = []
        boundaries = [0]

        for signal in signals:
            arr = signal.data
            arrays.append(arr)
            increment = arr.sizes[self.dim] if self.dim else 1
            boundaries.append(boundaries[-1] + increment)

        data = xr.concat(arrays, dim=self.dim or "members", combine_attrs="no_conflicts")

        data.attrs["boundaries"] = boundaries
        data.attrs["stack_dim"] = self.dim

        return data

    def _apply(self, signals):
        if isinstance(signals[0].data, xr.Dataset):
            return self._gather_datasets(signals)
        return self._gather_arrays(signals)


class UnstackSignals(Transformer):
    def __init__(self, dim=None):
        self.dim = dim

    def output_schema(self, input_schema):
        stacking_dim = self.dim or 'members'
        if isinstance(input_schema, DatasetSchema):
            return DatasetSchema({
                key: Schema(*(schema.dims - {stacking_dim}), 
                            selectable_dims=schema._selectable_dims)
                for key, schema in input_schema.items()
            })
        dims = set(input_schema.dims)
        dims.discard(stacking_dim)
        return Schema(dims, selectable_dims=input_schema._selectable_dims)

    def resolve_output_class(self):
        from ..sources import Collection
        return Collection

    def __call__(self, signal_stack):
        signals = []
        output_schema = self.output_schema(signal_stack.data_schema)

        for i in range(len(signal_stack.signals)):
            signal_class = signal_stack.transform.signal_class or signal_stack.signal_class
            transform = self._get_transform(i)
            origin = signal_stack.signals[i].origin
            signal = signal_class(
                inputs=[signal_stack],
                transform=transform,
                data_schema=output_schema,
                origin=origin,
                transformer=self,
                source_signal=signal_stack.signals[i],
                start=signal_stack.signals[i].start,
                duration=signal_stack.signals[i].duration,
                coord_map=signal_stack.signals[i].coord_map
            )
            signals.append(signal)

        return self.resolve_output_class()(signals)
    
    def _get_transform(self, idx):
        apply_kwargs = self._get_apply_kwargs(idx)
        return Transform(partial(self._apply, **apply_kwargs))
    
    def _get_apply_kwargs(self, idx):
        return {'idx': idx}

    def _infer_output_class(self, signal):
        if not hasattr(signal, 'transformer'):
            return signal.output_class
        return signal.transformer.resolve_output_class()

    def _apply(self, data, idx):
        attrs = deepcopy(data.attrs)
        boundaries = attrs.pop("boundaries")
        stack_dim = attrs.pop("stack_dim")
      
        dim = self.dim or stack_dim
        start, end = boundaries[idx], boundaries[idx + 1]
        selected_data = data.isel({dim: slice(start, end)})
        selected_data.attrs = attrs
        return selected_data
    