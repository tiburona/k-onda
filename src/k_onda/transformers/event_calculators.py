import numpy as np
import xarray as xr

import pint
from .core import Calculator

from k_onda.central import type_registry


class Rate(Calculator):
    name = "rate"
    key_mode = "standalone"

    def __init__(self, intervals=None, exclude_initial=None):
        self.intervals = intervals
        self.exclude_initial = exclude_initial

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

    def _validate_input(self, input, **kwargs):
        from ..signals import BinarySignal, PointProcessSignal

        if not isinstance(input, (PointProcessSignal, BinarySignal)):
            raise ValueError("Rate can only operate on EventSignal or BinarySignal.")

    def _prepare_rate_inputs(self, data, data_schema, intervals, exclude_initial):
        if isinstance(data_schema, type_registry.DatasetSchema):
            time_key = data_schema.default_variable_for("time")
            data = data[time_key]
            data_schema = data_schema[time_key]

        concrete_dim = data_schema.concrete_dim_from("time")

        if not concrete_dim:
            raise ValueError(
                "Right now Rate only works on time dims and no dim in your DataSchema "
                "represents time."
            )

        intervals = intervals(data) if callable(intervals) else intervals
        exclude_initial = (
            exclude_initial(data) if callable(exclude_initial) else exclude_initial
        )
        return data, data_schema, concrete_dim, intervals, exclude_initial

    @staticmethod
    def _count_binary_events(data):
        if isinstance(data, xr.Dataset):
            data = next(iter(data.data_vars.values()))

        try:
            mag = data.pint.magnitude
        except Exception:
            mag = data

        arr = np.asarray(mag, dtype=bool).reshape(-1)
        if arr.size == 0:
            return 0

        return int(arr[0]) + np.count_nonzero(arr[1:] & ~arr[:-1])

    def _apply_inner(
        self, data, duration=None, is_binary=False, data_schema=None, *args, **kwargs
    ):

        if is_binary:
            if self.intervals is not None or self.exclude_initial is not None:
                raise ValueError(
                    "Cannot select data with `intervals` or `exclude_initial` for BinarySignal."
                )
            return self._count_binary_events(data) / duration

        selected_data, data_schema, concrete_dim, intervals, exclude_initial = (
            self._prepare_rate_inputs(
                data, data_schema, self.intervals, self.exclude_initial
            )
        )

        if exclude_initial:
            index = np.searchsorted(selected_data, exclude_initial)
            selected_data = selected_data[index:]
            num_events = len(selected_data)
            duration = duration - exclude_initial
            return num_events / duration

        if intervals:
            ureg = pint.application_registry
            starts = np.searchsorted(
                selected_data, [interval[0] for interval in intervals]
            )
            stops = np.searchsorted(
                selected_data,
                [interval[1] + 10 ** (-9) * ureg.s for interval in intervals],
            )
            num_events = sum(
                len(selected_data[start:stop]) for start, stop in zip(starts, stops)
            )
            duration = sum([interval[1] - interval[0] for interval in intervals])
            return num_events / duration

        return len(selected_data) / duration

    def _wrap_result(self, result, *args):
        result = xr.DataArray(result)
        result = super()._wrap_result(result)
        return result
