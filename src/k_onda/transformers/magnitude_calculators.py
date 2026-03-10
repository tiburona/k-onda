import numpy as np

from .core import Calculator


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

    def _apply_inner(self, data):
        dim = self.dim
        if self.method == "rms":
            rms = np.sqrt((data**2).mean(dim=dim))
            normalization_params = {'rms': rms}
            result = data / rms
        elif self.method == "zscore":
            mean = data.mean(dim=dim); std = data.std(dim=dim)
            normalization_params = {'mean': mean, 'std': std}
            result = (data - mean) / std
        elif self.method == "minmax":
            data_min = data.min(dim=dim); data_max = data.max(dim=dim)
            normalization_params = {'data_min': data_min, 'data_max': data_max}
            result = (data - data.min(dim=dim)) / (data.max(dim=dim) - data.min(dim=dim))
        else:
            raise ValueError("Unknown normalize method")

        return result, {'normalization_params': normalization_params}
    
    def _wrap_result(self, result, _, normalization_params=None):
        if normalization_params is not None:
            result = result.assign_attrs(normalization_params=normalization_params)
        return result
