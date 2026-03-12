import numpy as np
from scipy.signal import find_peaks, peak_widths
import xarray as xr

from .core import Calculator
from k_onda.central import ureg, Schema


class FWHM(Calculator):

    name = 'fwhm'
    key_mode = 'append'

    def __init__(
        self,
        dim="samples",
        include_valleys=True,
        permissible_distance=50,
        distance_unit=ureg.raw_sample,
    ):
        self.dim = dim
        self.include_valleys = include_valleys
        self.distance_unit = distance_unit
        self.permissible_distance = permissible_distance * self.distance_unit

    def fwhm(self, data):
        def find_max_peak(values):
            peaks, properties = find_peaks(values, height=0)
            if len(peaks) == 0:
                return None, None
            heights = properties["peak_heights"]
            peak_pos = int(np.argmax(heights))
            return int(peaks[peak_pos]), float(heights[peak_pos])

        values = np.asarray(data)
        trim = int(getattr(self.permissible_distance, "magnitude", self.permissible_distance))
        if trim > 0 and values.size > 2 * trim:
            values = values[trim:-trim]

        peak_idx, peak_height = find_max_peak(values)
        signal_for_width = values

        if self.include_valleys:
            valley_idx, valley_height = find_max_peak(-values)
            if valley_idx is not None and (peak_idx is None or valley_height > peak_height):
                peak_idx = valley_idx
                signal_for_width = -values

        if peak_idx is None:
            return np.nan

        widths = peak_widths(signal_for_width, [peak_idx], rel_height=0.5)[0]
        return widths[0]

    def _apply_inner(self, data):
        if data.ndim > 1:
            return xr.apply_ufunc(
                self.fwhm,
                data,
                input_core_dims=[[self.dim]],
                vectorize=True,
            )

        return self.fwhm(data.values)
    
    def _wrap_result(self, result, *args):
        return xr.DataArray(result).pint.quantify(self.distance_unit)
    
    def output_schema(self, input_schema):
        dims = set(input_schema.dims)
        dims.discard(self.dim)
        return Schema(*dims)

