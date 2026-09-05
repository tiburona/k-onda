from copy import deepcopy
import xarray as xr

from .core import Transformer, KeySpec
from k_onda.central import DatasetSchema, AxisInfo, AxisKind, type_registry as tr


class StackSignals(Transformer):
    name = "stack_signals"

    """Concatenate component signals so downstream calculations can be vectorized."""

    def __init__(self, dim: str | None = None):
        self.validate_type_hints()
        self._validate_configuration(dim)
        self.adds_member_dim = dim is None
        self.dim = dim or "member"

    def _validate_configuration(self, dim):
        if dim is not None:
            self.validate_parameter("dim", dim, nonempty_string=True)

    def _validate_input(self, collection):
        if not isinstance(collection, tr.Collection):
            raise TypeError(
                f"{self.format_call()}: input must be a Collection, not "
                f"{type(collection).__name__}."
            )
        if not len(collection):
            raise ValueError(f"{self.format_call()}: cannot stack an empty Collection.")

        signals = collection.signals
        reference_schema = signals[0].data_schema
        if any(signal.data_schema != reference_schema for signal in signals[1:]):
            raise ValueError(
                f"{self.format_call()}: all stacked signals must have matching "
                "data schemas."
            )

    def output_schema(self, *input_schemas):
        axis = AxisInfo(name=self.dim, kind=AxisKind.AXIS)

        def stack_schema(schema):
            return schema.with_axis(axis)

        if isinstance(input_schemas[0], DatasetSchema):
            return DatasetSchema(
                {key: stack_schema(schema) for key, schema in input_schemas[0].items()}
            )
        return stack_schema(input_schemas[0])

    def __call__(
        self,
        collection: object,
        *,
        key: str | None = None,
        key_output_mode: str | None = None,
    ):

        if key is not None or key_output_mode is not None:
            raise NotImplementedError(
                "Key access is not yet implemented for SignalStack"
            )

        self._validate_input(collection)

        key_spec = KeySpec(input_name=key, output_mode=key_output_mode)

        input_schemas = [s.data_schema for s in collection.signals]
        output_schema = self.output_schema(*input_schemas)
        transform = self._get_transform(
            *collection.signals,
            key_spec=key_spec,
        )
        return tr.SignalStack(
            collection,
            data_schema=output_schema,
            transform=transform,
            key_spec=key_spec,
            transformer=self,
            stack_dim=self.dim,
            stack_dim_was_added=self.adds_member_dim,
        )

    def _ensure_stack_coord(self, data):
        if self.dim not in data.coords:
            data = data.assign_coords({self.dim: range(data.sizes[self.dim])})
        return data

    def _gather_datasets(self, data):
        keys = data[0].keys()
        gathered_data = {}
        boundaries = [0]

        for i, key in enumerate(keys):
            arrays = []
            for dataset in data:
                arr = dataset[key]
                arrays.append(arr)
                increment = arr.sizes[self.dim] if self.dim in arr.dims else 1
                if i == 0:
                    boundaries.append(boundaries[-1] + increment)

            gathered_data[key] = xr.concat(
                arrays,
                dim=self.dim,
                combine_attrs="no_conflicts",
                join="exact",
            )

        dataset = self._ensure_stack_coord(xr.Dataset(gathered_data))
        dataset.attrs["boundaries"] = boundaries
        dataset.attrs["stack_dim"] = self.dim
        dataset.attrs["stack_dim_was_added"] = self.adds_member_dim

        return dataset

    def _gather_arrays(self, data):
        arrays = []
        boundaries = [0]

        for arr in data:
            arrays.append(arr)
            increment = arr.sizes[self.dim] if self.dim in arr.dims else 1
            boundaries.append(boundaries[-1] + increment)

        gathered_data = xr.concat(
            arrays,
            dim=self.dim,
            combine_attrs="no_conflicts",
            join="exact",
        )
        gathered_data = self._ensure_stack_coord(gathered_data)

        gathered_data.attrs["boundaries"] = boundaries
        gathered_data.attrs["stack_dim"] = self.dim
        gathered_data.attrs["stack_dim_was_added"] = self.adds_member_dim

        return gathered_data

    def _apply(self, *data, **kwargs):
        if isinstance(data[0], xr.Dataset):
            return self._gather_datasets(data)
        return self._gather_arrays(data)


class UnstackSignals(Transformer):
    name = "unstack_signals"

    def _validate_input(self, signal_stack):
        if not isinstance(signal_stack, tr.SignalStack):
            raise TypeError(
                f"{self.format_call()}: input must be a SignalStack, not "
                f"{type(signal_stack).__name__}."
            )

    def output_schema(self, input_schema, stacking_dim, stack_dim_was_added):
        if not stack_dim_was_added:
            return input_schema

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
        self._validate_input(signal_stack)

        signals = []
        schema_kwargs = {
            "stacking_dim": signal_stack.stack_dim,
            "stack_dim_was_added": signal_stack.stack_dim_was_added,
        }
        output_schema = self.make_output_schema(
            signal_stack.data_schema,
            **schema_kwargs,
        )

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
                last_stack_index=i,
                schema_kwargs=schema_kwargs,
            )
            signals.append(signal)

        return self.resolve_output_class()(signals)

    def build_transform_for(self, signal):
        return super()._get_transform(
            *signal.inputs,
            key_spec=signal.key_spec,
            apply_kwargs=self._get_apply_kwargs(signal.last_stack_index),
        )

    def _get_apply_kwargs(self, idx):
        return {"idx": idx}

    def _infer_output_class(self, signal):
        if not hasattr(signal, "transformer"):
            return signal.output_class
        return signal.transformer.resolve_output_class()

    def _apply(self, data, idx):
        attrs = deepcopy(data.attrs)
        boundaries = attrs.pop("boundaries")
        dim = attrs.pop("stack_dim")
        stack_dim_was_added = attrs.pop("stack_dim_was_added")

        start, end = boundaries[idx], boundaries[idx + 1]
        selection = start if stack_dim_was_added else slice(start, end)
        selected_data = data.isel({dim: selection})
        if stack_dim_was_added:
            selected_data = selected_data.drop_vars(dim)
        selected_data.attrs = attrs
        return selected_data
