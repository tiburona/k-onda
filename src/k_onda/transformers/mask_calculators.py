from functools import partial

import numpy as np

from .core import Calculator, Transform, KeySpec


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

    @property
    def fixed_output_class(self):
        from ..signals import ValidityMask

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

        from ..signals import BinarySignal

        for signal in signals:
            if not isinstance(signal, BinarySignal):
                raise TypeError(f"{signal} is not of type BinarySignal.")


# TODO: this and ApplyMasks __call__'s need to be evaluated for how they're
# working with keys, how they apply to stacks, and in general to be brought up
# to date with the code base.  However given I might be refactoring key spec
# related stuff soon why don't I wait till I've done that.
class Intersection(Calculator, BinaryCalculatorMixin):
    name = "intersection"

    def __init__(self, tolerance_decimals=9):
        self.tolerance = 10 ** (-tolerance_decimals)

    def __call__(self, a, b, key=None, key_output_mode=None):

        if key is not None or key_output_mode is not None:
            raise NotImplementedError("Key access is not yet implemented for Intersection")

        key_spec = KeySpec(input_name=key, output_mode=key_output_mode)

        self.validate_sig_types([a, b])

        output_signal_class = self.resolve_output_class(a)

        return output_signal_class(
            inputs=(a, b),
            transform=None,
            data_schema=None,
            key_spec=key_spec,
            origin=(a.origin, b.origin),
            transformer=self,
        )

    def _apply_inner(self, a_data, b_data):
        a_overlap, b_overlap = self.get_and_validate_sig_overlap(a_data, b_data)
        return a_overlap.data & b_overlap.data
    
    def _get_transform(self, *args, **kwargs):
        return Transform(self._apply)


class ApplyMask(Calculator, BinaryCalculatorMixin):
    name = "apply_mask"

    def __init__(self, mask=None):
        self.mask = mask

    # TODO: all these calls need to be verified to work with SignalStack.
    def __call__(self, input, mask=None, key=None, key_output_mode=None):
        
        if key is not None or key_output_mode is not None:
            raise NotImplementedError("Key access is not yet implemented for ApplyMask")

        key_spec = KeySpec(input_name=key, output_mode=key_output_mode)

        mask = mask or self.mask
        if mask is None:
            raise ValueError("mask must be provided at init or call time")
        self.validate_sig_types([mask])
        output_class = self.resolve_output_class(input)

        return output_class(
            inputs=(input, mask),
            transformer=self,
            transform=None,
            data_schema=None,
            key_spec=key_spec,
            origin=(input.origin, mask.origin)
        )
    
    def _get_transform(self, *args, **kwargs):
        return Transform(self._apply)

    def _apply(self, sig_data, mask_data, *args, **kwargs):
        if kwargs.get("key_spec") is not None:
            raise NotImplementedError("ApplyMask can't extract a keyed result from a Dataset" \
            "Signal yet")
        
        result = self._apply_inner(sig_data, mask_data)
        result = self._wrap_result(result, sig_data)

        return result

    def _apply_inner(self, sig_data, mask_data):
        # Verify xarray alignment behavior - should give masked overlap + NaN outside.
        _, mask_overlap = self.get_and_validate_sig_overlap(sig_data, mask_data)
        result = sig_data.where(mask_overlap, other=np.nan)
        return result
