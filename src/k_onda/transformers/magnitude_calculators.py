import numpy as np

from .core import Calculator
from k_onda.utils import np_from_xr


class Shift(Calculator):
    name = "shift"

    def __init__(self, shift):
        self.shift = shift

    def _apply_inner(self, data):
        try:
            result = data + self.shift.data
        except AttributeError:
            result = data + self.shift
        return result


class Scale(Calculator):
    name = "scale"

    def __init__(self, factor):
        self.factor = factor

    def _apply_inner(self, data):
        try:
            result = data * self.factor.data
        except AttributeError:
            result = data * self.factor
        return result


class Normalize(Calculator):
    name = "normalize"

    def __init__(self, method="rms", dim=None):
        self.dim = dim
        self.method = method

    def _apply_inner(self, data, *args, **kwargs):
        dim = self.dim
        if self.method == "rms":
            rms = np.sqrt((data**2).mean(dim=dim))
            norm_params = {"rms": rms}
            result = data / rms
        elif self.method == "zscore":
            mean = data.mean(dim=dim)
            std = data.std(dim=dim)
            norm_params = {"mean": mean, "std": std}
            result = (data - mean) / std
        elif self.method == "minmax":
            data_min = data.min(dim=dim)
            data_max = data.max(dim=dim)
            norm_params = {"data_min": data_min, "data_max": data_max}
            result = (data - data.min(dim=dim)) / (
                data.max(dim=dim) - data.min(dim=dim)
            )
        else:
            raise ValueError("Unknown normalize method")

        return result, {"norm_params": norm_params}

    def _wrap_result(self, result, data, norm_params=None):

        if norm_params is not None:
            stripped = {}
            units = {}

            for key, param in norm_params.items():
                stripped[key], units[key] = np_from_xr(param)

            prenorm_units = (
                data.attrs.get(
                    "feature_units"
                )  # e.g. {'fwhm': unit, 'firing_rate': unit}
                or {"data": next(iter(units.values()), None)}  # e.g. {'data': unit}
            )

            result = result.assign_attrs(
                norm_dim=self.dim, norm_params=stripped, prenorm_units=prenorm_units
            )

        result = super()._wrap_result(result)
        return result
