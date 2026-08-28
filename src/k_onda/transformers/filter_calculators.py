from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import inspect
from numbers import Real

import numpy as np
from scipy.signal import iirnotch, medfilt, sosfilt, sosfiltfilt, tf2sos
import xarray as xr
import pint

from k_onda.central import DimBounds, DimPair
from .core import Calculator, PaddingCalculator

from k_onda.utils import scalar, is_unitful, ValidationMixin


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
class IIRNotchDesigner(ValidationMixin):
    f_lo: Real
    f_hi: Real
    notch_Q: Real | None = None

    def __post_init__(self):
        self.validate_type_hints()
        self.validate_number("f_lo", self.f_lo, minimum=0, finite=True)
        self.validate_number(
            "f_hi", self.f_hi, minimum=0, nonzero=True, finite=True
        )
        if self.f_hi <= self.f_lo:
            raise ValueError(
                f"{self.format_call()}: f_hi must be greater than f_lo."
            )
        if self.notch_Q is not None:
            self.validate_number(
                "notch_Q",
                self.notch_Q,
                minimum=0,
                nonzero=True,
                finite=True,
            )

    @lru_cache(maxsize=32)
    def design_sos(self, fs):
        f0 = 0.5 * (self.f_lo + self.f_hi)
        bandwidth = max(1e-12, self.f_hi - self.f_lo)
        q_value = float(self.notch_Q) if self.notch_Q is not None else float(f0 / bandwidth)
        b, a = iirnotch(w0=f0, Q=q_value, fs=fs)
        return tf2sos(b, a)


class Filter(PaddingCalculator):
    name = "filter"
    accepted_data_types = (xr.DataArray,)

    def __init__(
        self,
        method: str,
        *,
        dim: str = "time",
        **kwargs: object,
    ):
        self.validate_type_hints()
        self.validate_parameter("method", method, nonempty_string=True)
        self.validate_parameter("dim", dim, nonempty_string=True)
        if dim != "time":
            raise NotImplementedError(
                f"{self.format_call()}: filtering is currently implemented only "
                "for the time dimension."
            )

        self.method = method
        try:
            self.filter_designer = filter_registry.create_designer(method, **kwargs)
        except (TypeError, ValueError) as error:
            raise type(error)(f"{self.format_call()}: {error}") from None
        self.dim = dim

    def _validate_data_schema(self, input_schema):
        super()._validate_data_schema(input_schema)
        if input_schema.concrete_dim_from(self.dim) is None:
            raise ValueError(
                f"{self.format_call()}: input schema does not contain a dimension "
                f"representing {self.dim!r}."
            )

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

    def _apply_inner(self, data, designed_filter, data_schema, *args, **kwargs):
        concrete_dim = data_schema.concrete_dim_from(self.dim)

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
    require_all_finite = True
    accepted_data_types = (xr.DataArray,)

    def __init__(self, kernel_sizes: Mapping[str, int]):
        self.validate_type_hints()
        self._validate_configuration(kernel_sizes)
        self.kernel_sizes = dict(kernel_sizes)

    def _validate_configuration(self, kernel_sizes):
        self.validate_parameter("kernel_sizes", kernel_sizes, nonempty=True)
        self.validate_string_iterable("kernel dimension", kernel_sizes)

        for dim, size in kernel_sizes.items():
            self.validate_number(
                f"kernel size for {dim!r}",
                size,
                number_type=int,
                minimum=1,
            )
            if size % 2 == 0:
                raise ValueError(
                    f"{self.format_call()}: kernel size for {dim!r} must be a "
                    "positive odd integer."
                )

    def _resolved_kernel_sizes(self, data_schema):
        resolved = {}
        supplied_names = {}
        for dim, size in self.kernel_sizes.items():
            concrete_dim = data_schema.concrete_dim_from(dim)
            if concrete_dim is None:
                raise ValueError(
                    f"{self.format_call()}: kernel dimension {dim!r} was not found "
                    "in the input schema."
                )
            if concrete_dim in resolved:
                raise ValueError(
                    f"{self.format_call()}: kernel dimensions "
                    f"{supplied_names[concrete_dim]!r} and {dim!r} both resolve to "
                    f"{concrete_dim!r}."
                )
            resolved[concrete_dim] = size
            supplied_names[concrete_dim] = dim
        return resolved

    def _validate_data_schema(self, input_schema):
        super()._validate_data_schema(input_schema)
        self._resolved_kernel_sizes(input_schema)

    def _validate_data(self, data, **kwargs):
        units = data.pint.units
        values = np.asarray(data.pint.magnitude if units is not None else data)
        is_real_numeric = np.issubdtype(
            values.dtype, np.number
        ) and not np.issubdtype(values.dtype, np.complexfloating)
        if not is_real_numeric:
            raise TypeError(
                f"{self.format_call()}: input values must be real numbers."
            )

        super()._validate_data(data, **kwargs)

    def _apply_inner(self, data, data_schema, *args, **kwargs):
        resolved_sizes = self._resolved_kernel_sizes(data_schema)
        for dim, size in resolved_sizes.items():
            if size > data.sizes[dim]:
                raise ValueError(
                    f"{self.format_call()}: kernel size {size} for dimension {dim!r} "
                    f"exceeds its length of {data.sizes[dim]}."
                )

        kernel_size = tuple(resolved_sizes.get(dim, 1) for dim in data.dims)
        units = data.pint.units
        values = data.pint.magnitude if units is not None else data.values
        filtered = medfilt(values, kernel_size=kernel_size)
        filtered = filtered * units if units is not None else filtered
        return data.copy(data=filtered)
