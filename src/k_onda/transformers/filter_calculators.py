from functools import lru_cache

import numpy as np
from scipy.signal import iirnotch, medfilt, sosfilt, sosfiltfilt, tf2sos
import xarray as xr
import pint

from ..central import DimBounds, DimPair
from .core import Calculator, PaddingCalculator

from ..utils import scalar


class Filter(PaddingCalculator):
    name = "filter"

    def __init__(self, filter_config, dim="time"):
        self.config = filter_config
        self.dim = dim

    def _get_extra_apply_kwargs(self, parent_signal):
        designed_filter = self.design_filter(parent_signal)
        return {"designed_filter": designed_filter}

    def design_filter(self, parent_signal):
        fs = scalar(parent_signal.sampling_rate)
        # TODO: add in other kinds of filters.
        return self._design_sos(fs=fs, **self.config)

    @staticmethod
    @lru_cache(maxsize=32)
    def _design_sos(method, f_lo, f_hi, fs, notch_Q=None, **_):
        if method == "iir_notch":
            f0 = 0.5 * (f_lo + f_hi)
            bw = max(1e-12, (f_hi - f_lo))

            if notch_Q is not None:
                q_val = float(notch_Q)
            else:
                # Q = center frequency divided by bandwidth
                q_val = float(f0 / bw)
            b, a = iirnotch(w0=f0, Q=q_val, fs=fs)
            sos = tf2sos(b, a)
            return sos

        raise ValueError(f"Unknown filter method: {method}")

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

        return DimBounds({"time": DimPair([pad_seconds, pad_seconds])})

    def _apply_inner(self, data, designed_filter, *args, **kwargs):
        dim = self.dim
        if dim != "time":
            raise NotImplementedError(
                "You can currently only filter along the time dimension."
            )
        axis = data.dims.index(dim)
        result = sosfiltfilt(designed_filter, data, axis=axis)
        return result
    
    def _wrap_result(self, result, data):
        result = xr.DataArray(result, coords=data.coords, dims=data.dims, attrs=data.attrs)
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
        result = xr.DataArray(result, coords=data.coords, dims=data.dims, attrs=data.attrs)
        result = super()._wrap_result(result)
        return result
