from copy import deepcopy
from functools import partial
import xarray as xr

from .core import Transformer, Transform, KeySpec
from k_onda.central import DatasetSchema, AxisInfo, AxisKind, type_registry


class StackSignals(Transformer):
    name = "stack_signals"

    """Concatenate component signals so downstream calculations can be vectorized."""

    def __init__(self, dim=None):
        self.dim = dim

    def output_schema(self, *input_schemas):
        stacking_dim = self.dim or "members"
        axis = AxisInfo(name=stacking_dim, kind=AxisKind.AXIS)
        
        def stack_schema(schema):
            return schema.with_axis(axis)
       
        if isinstance(input_schemas[0], DatasetSchema):
            return DatasetSchema(
                {
                    key: stack_schema(schema)
                    for key, schema in input_schemas[0].items()
                }
            )
        return stack_schema(input_schemas[0])

    def __call__(self, collection, key=None, key_output_mode=None):
        
        if key is not None or key_output_mode is not None:
            raise NotImplementedError("Key access is not yet implemented for SignalStack")

        key_spec = KeySpec(input_name=key, output_mode=key_output_mode)

        input_schemas = [s.data_schema for s in collection.signals]
        output_schema = self.output_schema(*input_schemas)
        transform = self._get_transform()
        return type_registry.SignalStack(
            collection, 
            data_schema=output_schema, 
            transform=transform, 
            key_spec=key_spec,
            transformer=self
            )

    def _get_transform(self, *args, **kwargs):
        return Transform(self._apply)

    def _gather_datasets(self, data):
        keys = data[0].keys()
        gathered_data = {}
        boundaries = [0]

        for i, key in enumerate(keys):
            arrays = []
            for dataset in data:
                arr = dataset[key]
                arrays.append(arr)
                increment = arr.sizes[self.dim] if self.dim else 1
                if i == 0:
                    boundaries.append(boundaries[-1] + increment)

            gathered_data[key] = xr.concat(
                arrays, dim=self.dim or "members", combine_attrs="no_conflicts"
            )

        dataset = xr.Dataset(gathered_data)
        dataset.attrs["boundaries"] = boundaries
        dataset.attrs["stack_dim"] = self.dim

        return dataset

    def _gather_arrays(self, data):
        arrays = []
        boundaries = [0]

        for arr in data:
            arrays.append(arr)
            increment = arr.sizes[self.dim] if self.dim else 1
            boundaries.append(boundaries[-1] + increment)

        gathered_data = xr.concat(
            arrays, dim=self.dim or "members", combine_attrs="no_conflicts"
        )

        gathered_data.attrs["boundaries"] = boundaries
        gathered_data.attrs["stack_dim"] = self.dim

        return gathered_data

    def _apply(self, *data, **kwargs):
        
        if isinstance(data[0], xr.Dataset):
            return self._gather_datasets(data)
        return self._gather_arrays(data)


class UnstackSignals(Transformer):
    def __init__(self, dim=None):
        self.dim = dim

    def output_schema(self, input_schema):
        stacking_dim = self.dim or "members"
        if isinstance(input_schema, DatasetSchema):
            return DatasetSchema(
                {
                    key: schema.without(stacking_dim)
                    for key, schema in input_schema.items()
                }
            )

        return input_schema.without(stacking_dim)

    def resolve_output_class(self):
        from ..sources import Collection

        return Collection

    def __call__(self, signal_stack):
        signals = []
        output_schema = self.output_schema(signal_stack.data_schema)

        for i in range(len(signal_stack.signals)):
            signal_class = (
                signal_stack.transform.signal_class or signal_stack.signal_class
            )
        
            origin = signal_stack.signals[i].origin
            signal = signal_class(
                inputs=[signal_stack],
                data_schema=output_schema,
                origin=origin,
                transformer=self,
                source_signal=signal_stack.signals[i],
                start=signal_stack.signals[i].start,
                duration=signal_stack.signals[i].duration,
                context=signal_stack.signals[i].context,
                last_stack_index=i
            )
            signals.append(signal)

        return self.resolve_output_class()(signals)
    
    def build_transform_for(self, signal):
        return self._get_transform(signal.last_stack_index)

    def _get_transform(self, idx):
        apply_kwargs = self._get_apply_kwargs(idx)
        return Transform(partial(self._apply, **apply_kwargs))

    def _get_apply_kwargs(self, idx):
        return {"idx": idx}

    def _infer_output_class(self, signal):
        if not hasattr(signal, "transformer"):
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
