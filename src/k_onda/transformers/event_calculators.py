import numpy as np
import xarray as xr

from ..central import ureg
from .core import Calculator


class Rate(Calculator):

    name = 'rate'
    key_mode = 'standalone'

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
            'duration': parent.duration,
            'is_binary': isinstance(parent, BinarySignal),
            'coord_map': getattr(parent, 'coord_map', None)

        }

    def _validate_input(self, parent):
        from ..signals import BinarySignal, PointProcessSignal

        if not isinstance(parent, (PointProcessSignal, BinarySignal)):
            raise ValueError("Rate can only operate on EventSignal or BinarySignal.")

    def _validate_rate_inputs(self, time_key, intervals=None, exclude_initial=None):
        # todo: add some validation in here: if you got passed a data array,
        # it's name is not time key, and (intervals or exclude_initial), raise
        if not time_key and (intervals or exclude_initial):
            raise ValueError("Cannot select data if `time_key` is not defined.")

    def _prepare_rate_inputs(self, data, coord_map, intervals, exclude_initial):
        time_key = None if coord_map is None else coord_map.get('time')
        intervals = intervals(data) if callable(intervals) else intervals
        exclude_initial = exclude_initial(data) if callable(exclude_initial) else exclude_initial
        return time_key, intervals, exclude_initial

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

    def _apply_inner(self, data, duration=None, is_binary=False, coord_map=None):

        if is_binary:
            if self.intervals is not None or self.exclude_initial is not None:
                raise ValueError(
                    "Cannot select data with `intervals` or `exclude_initial` for BinarySignal."
                )
            return self._count_binary_events(data) / duration

        time_key, intervals, exclude_initial = self._prepare_rate_inputs(
            data, coord_map, self.intervals, self.exclude_initial
        )
        self._validate_rate_inputs(time_key, intervals, exclude_initial)

        if not time_key:
            return len(data[data.keys()[0]]) / duration

        if isinstance(data, xr.Dataset):
            selected_data = data[time_key]
        else:
            selected_data = data

        if exclude_initial:
            index = np.searchsorted(selected_data, exclude_initial)
            selected_data = selected_data[index:]
            num_events = len(selected_data)
            duration = duration - exclude_initial
            return num_events / duration

        if intervals:
            starts = np.searchsorted(selected_data, [interval[0] for interval in intervals])
            stops = np.searchsorted(
                selected_data, [interval[1] + 10 ** (-9) * ureg.s for interval in intervals]
            )
            num_events = sum(
                len(selected_data[start:stop]) for start, stop in zip(starts, stops)
            )
            duration = sum([interval[1] - interval[0] for interval in intervals])
            return num_events / duration

        return len(selected_data) / duration
    
    def _wrap_result(self, result, *args):
        return xr.DataArray(result)
    
