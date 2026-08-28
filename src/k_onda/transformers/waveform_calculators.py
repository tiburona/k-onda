import numpy as np
from scipy.signal import find_peaks, peak_widths
import xarray as xr
import pint

from .core import Calculator


class FWHM(Calculator):
    name = "fwhm"
    key_mode = "append"
    accepted_data_types = (xr.DataArray,)
    peak_selection_modes = ("prominence", "height")

    def __init__(
        self,
        *,
        dim: str = "sample",
        include_valleys: bool = True,
        peak_selection: str = "prominence",
    ):
        self.validate_type_hints()
        self.validate_parameter("dim", dim, nonempty_string=True)
        self.validate_parameter(
            "peak_selection",
            peak_selection,
            choices=self.peak_selection_modes,
        )

        self.dim = dim
        self.include_valleys = include_valleys
        self.peak_selection = peak_selection

    def _validate_data_schema(self, input_schema):
        super()._validate_data_schema(input_schema)
        if input_schema.concrete_dim_from(self.dim) is None:
            raise ValueError(
                f"{self.format_call()}: input schema does not contain dimension "
                f"{self.dim!r}."
            )

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

    def _apply_inner(self, data, data_schema, *args, **kwargs):

        concrete_dim = data_schema.concrete_dim_from(self.dim)

        if data.ndim > 1:
            return xr.apply_ufunc(
                self.fwhm,
                data,
                input_core_dims=[[concrete_dim]],
                vectorize=True,
            )

        return self.fwhm(data.values)

    def _wrap_result(self, result, *args):
        result = xr.DataArray(result).pint.quantify(
            pint.application_registry.raw_sample
        )
        return super()._wrap_result(result)

    def output_schema(self, input_schema):
        return input_schema.without_dim(self.dim)
