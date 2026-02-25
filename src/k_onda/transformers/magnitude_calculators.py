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
            result = data / np.sqrt((data**2).mean(dim=dim))
        elif self.method == "zscore":
            result = (data - data.mean(dim=dim)) / data.std(dim=dim)
        elif self.method == "minmax":
            result = (data - data.min(dim=dim)) / (data.max(dim=dim) - data.min(dim=dim))
        else:
            raise ValueError("Unknown normalize method")

        return result
