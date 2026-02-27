from functools import partial

import numpy as np

from .core import Calculator, Transform
from ..signals import BinarySignal, ValidityMask


class Threshold(Calculator):
    name = "threshold"

    def __init__(self, comparison, threshold):
        self.threshold = threshold
        self.comparison = comparison
        self.operations = {
            "gt": lambda data, value: data > value,
            "lt": lambda data, value: data < value,
            "ge": lambda data, value: data >= value,
            "le": lambda data, value: data <= value,
        }

    def get_child_class(self, _):

        return ValidityMask

    def _apply_inner(self, data):
        return self.operations[self.comparison](data, self.threshold)


class BinaryCalculatorMixin:
    def get_and_validate_sig_overlap(self, parent_data, other_data, dims=None):
        """
        Find overlapping region on shared dimensions.

        Args:
            dims: Dimension(s) to align on. If None, uses all shared dimensions.
                  Can be a string ('time') or list (['time', 'frequency']).
        """

        shared_dims = set(parent_data.dims) & set(other_data.dims)

        # Determine which dims to align on
        if dims is None:
            dims = shared_dims
        else:
            dims = {dims} if isinstance(dims, str) else set(dims)
            dims = dims & shared_dims  # Only use dims that exist in both

        if not dims:
            raise ValueError("No shared dimensions to align on")

        # Build selection slices for each dimension
        slices = {}
        for dim in dims:
            coord_parent = parent_data.coords[dim]
            coord_other = other_data.coords[dim]

            overlap_start = max(coord_parent[0].item(), coord_other[0].item())
            overlap_end = min(coord_parent[-1].item(), coord_other[-1].item())

            if overlap_start >= overlap_end:
                raise ValueError(f"No overlap on dimension '{dim}'")

            slices[dim] = slice(overlap_start, overlap_end)

        # Select overlapping regions
        parent_overlap = parent_data.sel(**slices)
        other_overlap = other_data.sel(**slices)

        # Validate lengths match on aligned dimensions
        for dim in dims:
            if len(parent_overlap.coords[dim]) != len(other_overlap.coords[dim]):
                raise ValueError(f"Signals have different sampling on '{dim}'")

        return parent_overlap, other_overlap

    def validate_sig_types(self, signals):
        

        for signal in signals:
            if not isinstance(signal, BinarySignal):
                raise TypeError(f"{signal} is not of type BinarySignal.")


class Intersection(Calculator, BinaryCalculatorMixin):
    name = "intersection"

    def __init__(self, tolerance_decimals=9):
        self.tolerance = 10 ** (-tolerance_decimals)

    def __call__(self, parent, other):
        self.validate_sig_types([parent, other])

        child_signal_class = self.get_child_class(parent)

        transform = Transform(partial(self._apply, other=other))

        return child_signal_class(
            parent=parent,
            transform=transform,
            origin=(parent.origin, other.origin),
            transformer=self,
        )

    def _apply_inner(self, parent_data, other):
        parent_overlap, other_overlap = self.get_and_validate_sig_overlap(parent_data, other.data)
        return parent_overlap.data & other_overlap.data


class ApplyMask(Calculator, BinaryCalculatorMixin):
    name = "apply_mask"

    def __init__(self, mask=None):
        self.mask = mask

    # TODO: all these calls need to be verified to work with SignalStack.
    def __call__(self, parent_signal, mask=None):
        mask = mask or self.mask
        if mask is None:
            raise ValueError("mask must be provided at init or call time")
        self.validate_sig_types([mask])
        child_signal_class = self.get_child_class(parent_signal)
        transform = Transform(partial(self._apply, mask=mask))

        return child_signal_class(
            parent=parent_signal,
            transform=transform,
            origin=(parent_signal.origin, mask.origin),
            calculator=self,
        )

    def _apply_inner(self, parent_data, mask):
        # Verify xarray alignment behavior - should give masked overlap + NaN outside.
        _, mask_overlap = self.get_and_validate_sig_overlap(parent_data, mask.data)
        result = parent_data.where(mask_overlap, other=np.nan)
        return result
