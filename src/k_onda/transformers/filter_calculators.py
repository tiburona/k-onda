from dataclasses import dataclass
from functools import lru_cache
import inspect

import numpy as np
from scipy.signal import iirnotch, medfilt, sosfilt, sosfiltfilt, tf2sos
import xarray as xr
import pint

from k_onda.central import DimBounds, DimPair
from .core import Calculator, PaddingCalculator

from  k_onda.utils import scalar, is_unitful


class FilterRegistry:
    def __init__(self):
        self._filter_types = {}

    def register(self, method):
        def decorator(filter_type):
            self._filter_types[method] = filter_type
            return filter_type

        return decorator

    def create_designer(self, method, **kwargs):
        try:
            filter_type = self._filter_types[method]
        except KeyError:
            known_types = ", ".join(sorted(self._filter_types))
            raise ValueError(
                f"Unknown filter method {method!r}. Registered methods: {known_types}."
            ) from None

        try:
            inspect.signature(filter_type).bind(**kwargs)
        except TypeError as error:
            raise TypeError(
                f"Invalid parameters for filter method {method!r}: {error}"
            ) from None

        return filter_type(**kwargs)


filter_registry = FilterRegistry()


@filter_registry.register("iir_notch")
@dataclass(frozen=True)
class IIRNotchDesigner:
    f_lo: float
    f_hi: float
    notch_Q: float | None = None

    @lru_cache(maxsize=32)
    def design_sos(self, fs):
        f0 = 0.5 * (self.f_lo + self.f_hi)
        bandwidth = max(1e-12, self.f_hi - self.f_lo)
        q_value = float(self.notch_Q) if self.notch_Q is not None else float(f0 / bandwidth)
        b, a = iirnotch(w0=f0, Q=q_value, fs=fs)
        return tf2sos(b, a)


class Filter(PaddingCalculator):
    name = "filter"

    def __init__(self, method, *, dim="time", **kwargs):
        self.method = method
        try:
            self.filter_designer = filter_registry.create_designer(method, **kwargs)
        except (TypeError, ValueError) as error:
            raise type(error)(f"{self.format_call()}: {error}") from None
        self.dim = dim

    def _get_extra_apply_kwargs(self, parent_signal):
        designed_filter = self.design_filter(parent_signal)
        return {"designed_filter": designed_filter}

    def design_filter(self, parent_signal):
        if not is_unitful(parent_signal.sampling_rate):
            raise ValueError("A sampling rate must have units.")
        fs = scalar(parent_signal.sampling_rate)
        return self.filter_designer.design_sos(fs)

    def _compute_padlen(self, parent_signal, apply_kwargs):
        fs = parent_signal.sampling_rate.magnitude
        designed_filter = apply_kwargs["designed_filter"]

        # Generate impulse response
        n_samples = int(fs)  # 1 second worth of samples
        impulse = np.zeros(n_samples)
        impulse[0] = 1.0
        h = sosfilt(designed_filter, impulse)

        # Find where it decays below some threshold.
        threshold = 1e-3  # -60 dB relative to peak
        peak = np.max(np.abs(h))
        settled = np.where(np.abs(h) > threshold * peak)[0]
        pad_needed = settled[-1] if len(settled) > 0 else 0
        pad_seconds = pad_needed / fs * pint.application_registry.s

        return DimBounds({"time": DimPair([-pad_seconds, pad_seconds])})

    def _apply_inner(self, data, designed_filter, data_schema=None, *args, **kwargs):
        concrete_dim = (
            data_schema.concrete_dim_from(self.dim)
            if data_schema is not None
            else self.dim
        )

        if self.dim != "time":
            raise NotImplementedError(
                "You can currently only filter along the time dimension."
            )

        if concrete_dim not in data.dims:
            raise ValueError(
                f"Filter expected a {self.dim} axis, resolved to {concrete_dim}"
                f"but data dims are {data.dims}"
            )
        
        axis = data.get_axis_num(concrete_dim)
        result = sosfiltfilt(designed_filter, data, axis=axis)
        return result

    def _wrap_result(self, result, data):
        result = xr.DataArray(
            result, coords=data.coords, dims=data.dims, attrs=data.attrs
        )
        result = super()._wrap_result(result)
        return result


class MedianFilter(Calculator):
    name = "median_filter"

    def __init__(self, kernel_sizes):
        self.kernel_sizes = kernel_sizes

    def _apply_inner(self, data, *args, **kwargs):
        # kernel_sizes is a dictionary like {'samples': 5}
        kernel_size = tuple(self.kernel_sizes.get(dim, 1) for dim in data.dims)
        result = medfilt(data, kernel_size=kernel_size)
        return result

    def _wrap_result(self, result, data):
        result = xr.DataArray(
            result, coords=data.coords, dims=data.dims, attrs=data.attrs
        )
        result = super()._wrap_result(result)
        return result
