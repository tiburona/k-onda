import xarray as xr

from .core import Transform, Transformer, KeySpec
from k_onda.central import type_registry, AxisInfo, AxisKind


class Aggregator(Transformer):
    def __init__(self, method="mean", group_by=None):
        self.method = method
        self.group_by = group_by

    def __call__(self, input, key=None, key_output_mode=None):

        if key is not None or key_output_mode is not None:
            raise NotImplementedError("Key access is not yet implemented for Aggregator")

        key_spec = KeySpec(input_name=key, output_mode=key_output_mode)

        if isinstance(input, type_registry.CollectionMap):
            if self.group_by is not None:
                raise ValueError("input of type CollectionMap is already grouped.")
            return self._call_on_collection_map(input, key_spec)

        elif isinstance(input, type_registry.Collection):
            if self.group_by is not None:
                input = input.group_by(self.group_by)
                return self._call_on_collection_map(input, key_spec)
            else:
                return self._call_on_collection(input, key_spec)

        else:
            raise ValueError(
                "Aggregator must be called on a Collection or a Grouped_Collection "
            )

    def _call_on_collection_map(self, collection_map, key_spec):

        return type_registry.SignalMap(
            map={
                k: self._call_on_collection(collection, key_spec)
                for k, collection in collection_map.items()
            }
        )

    def _call_on_collection(self, collection, key_spec):

        inputs = tuple(collection.signals)
        transform = self._get_transform(inputs)
        data_schema = self.output_schema(inputs[0].data_schema)

        return type_registry.AggregatedSignal(
            inputs=inputs,
            transformer=self,
            transform=transform,
            data_schema=data_schema,
            key_spec=key_spec
        )

    def _get_transform(self, *args, **kwargs):

        return Transform(self._apply)

    def output_schema(self, input_schema):
        if self.method == "concat":
            return input_schema.with_axis(AxisInfo("members", kind=AxisKind.AXIS))
        return input_schema

    def _gather_datasets(self, data):
        keys = data[0].keys()
        gathered_data = {}

        for key in keys:
            arrays = []
            for dataset in data:
                arr = dataset[key]
                arrays.append(arr)

            gathered_data[key] = xr.concat(arrays, dim="members", combine_attrs="no_conflicts")

        return xr.Dataset(gathered_data)

    def _gather_arrays(self, data):
        return xr.concat(data, dim="members", combine_attrs="no_conflicts")

    def _gather(self, data):
        if isinstance(data[0], xr.Dataset):
            return self._gather_datasets(data)
        return self._gather_arrays(data)

    def _apply(self, *data, **kwargs):

        gathered_data = self._gather(data)
        if self.method == "concat":
            result = gathered_data
        else:
            result = getattr(gathered_data, self.method)(dim="members")

        return result

