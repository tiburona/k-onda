import numpy as np
import xarray as xr

from .core import Calculator

from k_onda.central import type_registry as tr


class Rate(Calculator):
    name = "rate"
    key_mode = "standalone"

    @property
    def fixed_output_class(self):
        from ..signals import ScalarSignal

        return ScalarSignal

    def _get_extra_apply_kwargs(self, parent):
        from ..signals import BinarySignal

        return {
            "duration": parent.duration,
            "is_binary": isinstance(parent, BinarySignal),
        }
    
    def output_schema(self, input_schema):
        return tr.Schema()   

    def _validate_data_schema(self, input_schema):
        super()._validate_data_schema(input_schema)
        if isinstance(input_schema, tr.DatasetSchema):
            time_key = input_schema.default_variable_for("time")
            if time_key is None:
                raise NotImplementedError(
                    f"{self.format_call()}: the DatasetSchema has no variable "
                    "representing time, and rates over other dimensions are not "
                    "implemented yet."
                )
            input_schema = input_schema[time_key]

        if input_schema.concrete_dim_from("time") is None:
            raise NotImplementedError(
                f"{self.format_call()}: the input schema has no dimension "
                "representing time, and rates over other dimensions are not "
                "implemented yet."
            )

    def _validate_input(self, input, **kwargs):
        from ..signals import BinarySignal, PointProcessSignal

        if not isinstance(input, (PointProcessSignal, BinarySignal)):
            raise TypeError(
                f"{self.format_call()}: input must be a PointProcessSignal or "
                "BinarySignal."
            )

    def _resolve_rate_data(self, data, data_schema):
        if isinstance(data_schema, tr.DatasetSchema):
            time_key = data_schema.default_variable_for("time")
            data = data[time_key]
        return data

    def _validate_duration(self, duration):
        if duration is None:
            raise ValueError(
                f"{self.format_call()}: input duration is required for this rate "
                "calculation."
            )
        self.validate_number(
            "input duration",
            duration,
            minimum=0,
            nonzero=True,
            finite=True,
            allow_quantity=True,
        )

    @staticmethod
    def _count_binary_events(data):
        try:
            mag = data.pint.magnitude
        except Exception:
            mag = data

        arr = np.asarray(mag, dtype=bool).reshape(-1)
        if arr.size == 0:
            return 0

        return int(arr[0]) + np.count_nonzero(arr[1:] & ~arr[:-1])

    def _apply_inner(
        self,
        data,
        *args,
        duration,
        is_binary,
        data_schema,
        **kwargs,
    ):

        self._validate_duration(duration)
        data = self._resolve_rate_data(data, data_schema)
        if is_binary:
            event_count = self._count_binary_events(data)
        else:
            event_count = len(data)
        return event_count / duration

    def _wrap_result(self, result, *args):
        result = xr.DataArray(result)
        result = super()._wrap_result(result)
        return result
