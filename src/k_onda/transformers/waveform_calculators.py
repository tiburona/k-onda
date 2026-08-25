import numpy as np
from scipy.signal import find_peaks, peak_widths
import xarray as xr
import pint

from .core import Calculator


class FWHM(Calculator):
    name = "fwhm"
    key_mode = "append"

    def __init__(
        self,
        *,
        dim="sample",
        include_valleys=True,
        peak_selection="prominence",
    ):
        if peak_selection not in {"prominence", "height"}:
            raise ValueError(
                f"{self.format_call()}: peak_selection must be 'prominence' or "
                "'height'."
            )

        self.dim = dim
        self.include_valleys = include_valleys
        self.peak_selection = peak_selection

    def fwhm(self, data):
        def find_max_peak(values):
            find_kwargs = {"height": 0}
            if self.peak_selection == "prominence":
                find_kwargs["prominence"] = 0

            peaks, properties = find_peaks(values, **find_kwargs)
            if len(peaks) == 0:
                return None, None

            property_name = (
                "prominences"
                if self.peak_selection == "prominence"
                else "peak_heights"
            )
            strengths = properties[property_name]
            peak_pos = int(np.argmax(strengths))
            return int(peaks[peak_pos]), float(strengths[peak_pos])

        values = np.asarray(data)
        peak_idx, peak_strength = find_max_peak(values)
        signal_for_width = values

        if self.include_valleys:
            valley_idx, valley_strength = find_max_peak(-values)
            if valley_idx is not None and (
                peak_idx is None or valley_strength > peak_strength  # pyright: ignore[reportOperatorIssue]
            ):
                peak_idx = valley_idx
                signal_for_width = -values

        if peak_idx is None:
            return np.nan

        widths = peak_widths(signal_for_width, [peak_idx], rel_height=0.5)[0]
        return widths[0]

    def _apply_inner(self, data, *args, **kwargs):

        if data.ndim > 1:
            return xr.apply_ufunc(
                self.fwhm,
                data,
                input_core_dims=[[self.dim]],
                vectorize=True,
            )

        return self.fwhm(data.values)

    def _wrap_result(self, result, *args):
        result = xr.DataArray(result).pint.quantify(
            pint.application_registry.raw_sample
        )
        return super()._wrap_result(result)

    def output_schema(self, input_schema):
        return input_schema.without(self.dim)
