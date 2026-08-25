
from functools import partial
import numpy as np
import pandas as pd
import pint

from .core import Calculator, Transform, KeySpec


class Threshold(Calculator):
    name = "threshold"
    comparisons = {">", "<", ">=", "<="}

    def __init__(self, comparison, threshold):
        self._validate_configuration(comparison, threshold)

        self.threshold = threshold
        self.comparison = comparison
        self.operations = {
            ">": lambda data, value: data > value,
            "<": lambda data, value: data < value,
            ">=": lambda data, value: data >= value,
            "<=": lambda data, value: data <= value,
        }

    def _validate_configuration(self, comparison, threshold):
        if not isinstance(comparison, str):
            raise TypeError(
                f"{self.format_call()}: comparison must be a string."
            )
        if comparison not in self.comparisons:
            known_comparisons = ", ".join(sorted(self.comparisons))
            raise ValueError(
                f"{self.format_call()}: unknown comparison {comparison!r}. "
                f"Available comparisons: {known_comparisons}."
            )
        if threshold is None:
            raise TypeError(
                f"{self.format_call()}: threshold cannot be None."
            )

    @property
    def fixed_output_class(self):
        from ..signals import ValidityMask

        return ValidityMask

    def _apply_inner(self, data):
        return self.operations[self.comparison](data, self.threshold)


class BinaryCalculatorMixin:
    @staticmethod
    def _coord_values(coord):
        units = coord.pint.units
        values = np.asarray(coord.pint.magnitude if units is not None else coord)
        return values, units

    def _mismatched_grid_error(self, dim, detail):
        return NotImplementedError(
            f"{self.format_call()}: matching these coordinate grids is not "
            f"implemented yet on dimension {dim!r}: {detail}."
        )

    def _coordinates_in_common_units(self, parent_coord, other_coord, dim):
        parent_values, parent_units = self._coord_values(parent_coord)
        other_values, other_units = self._coord_values(other_coord)

        if parent_units is None and other_units is None:
            return parent_values, other_values
        if (parent_units is None) != (other_units is None):
            raise ValueError(
                f"{self.format_call()}: coordinates on dimension {dim!r} cannot "
                "be aligned because only one signal has coordinate units."
            )

        try:
            converted_other = (other_values * other_units).to(parent_units)
        except pint.DimensionalityError as error:
            raise ValueError(
                f"{self.format_call()}: coordinates on dimension {dim!r} have "
                f"incompatible units {parent_units!s} and {other_units!s}."
            ) from error

        return parent_values, np.asarray(converted_other.magnitude)

    def _coord_contract(self, schema, dim):
        contract = schema.coord_by_name(dim)
        if contract is None:
            raise ValueError(
                f"{self.format_call()}: the data schema does not describe "
                f"coordinate {dim!r}."
            )
        return contract

    def _ordered_overlap_indices(
        self,
        parent_values,
        other_values,
        dim,
        parent_ordering,
        other_ordering,
    ):
        # searchsorted requires increasing values. Keep the original positions
        # so the selected data can be restored to the parent coordinate's order.
        parent_positions = np.arange(len(parent_values))
        other_positions = np.arange(len(other_values))
        if parent_ordering == "decreasing":
            parent_values = parent_values[::-1]
            parent_positions = parent_positions[::-1]
        if other_ordering == "decreasing":
            other_values = other_values[::-1]
            other_positions = other_positions[::-1]

        try:
            overlap_start = max(parent_values[0], other_values[0])
            overlap_end = min(parent_values[-1], other_values[-1])
        except TypeError as error:
            raise self._mismatched_grid_error(
                dim, "coordinate value types cannot be compared"
            ) from error
        if overlap_start > overlap_end:
            raise ValueError(
                f"{self.format_call()}: coordinates do not overlap on dimension "
                f"{dim!r}."
            )

        parent_slice = slice(
            np.searchsorted(parent_values, overlap_start, side="left"),
            np.searchsorted(parent_values, overlap_end, side="right"),
        )
        other_slice = slice(
            np.searchsorted(other_values, overlap_start, side="left"),
            np.searchsorted(other_values, overlap_end, side="right"),
        )
        if not np.array_equal(
            parent_values[parent_slice], other_values[other_slice]
        ):
            raise self._mismatched_grid_error(
                dim, "coordinates in the overlapping range differ"
            )

        parent_indices = parent_positions[parent_slice]
        other_indices = other_positions[other_slice]
        if parent_ordering == "decreasing":
            parent_indices = parent_indices[::-1]
            other_indices = other_indices[::-1]
        return parent_indices, other_indices

    def _unique_overlap_indices(self, parent_values, other_values, dim):
        # get_indexer returns each parent value's position in the other coordinate,
        # or -1 when that value is absent.
        parent_index = pd.Index(parent_values)
        other_index = pd.Index(other_values)
        if not parent_index.is_unique or not other_index.is_unique:
            raise self._mismatched_grid_error(
                dim,
                "coordinate values are not unique at the selected matching precision",
            )

        try:
            other_positions = other_index.get_indexer(parent_index)
        except pd.errors.InvalidIndexError as error:
            raise self._mismatched_grid_error(
                dim, "coordinate matching is ambiguous"
            ) from error

        parent_indices = np.flatnonzero(other_positions >= 0)
        if not len(parent_indices):
            raise ValueError(
                f"{self.format_call()}: coordinates do not overlap on dimension "
                f"{dim!r}."
            )
        return parent_indices, other_positions[parent_indices]

    def _overlap_indices(
        self,
        parent_values,
        other_values,
        dim,
        parent_contract,
        other_contract,
    ):
        if np.array_equal(parent_values, other_values):
            indices = np.arange(len(parent_values))
            return indices, indices

        if parent_contract.is_monotonic and other_contract.is_monotonic:
            return self._ordered_overlap_indices(
                parent_values,
                other_values,
                dim,
                parent_contract.ordering,
                other_contract.ordering,
            )

        if parent_contract.is_unique and other_contract.is_unique:
            return self._unique_overlap_indices(
                parent_values, other_values, dim
            )

        raise self._mismatched_grid_error(
            dim, "matching coordinates that may contain repeats is ambiguous"
        )

    def _align_overlapping_data(
        self,
        parent_data,
        other_data,
        parent_schema,
        other_schema,
        dims=None,
        tolerance_decimals=None,
    ):
        shared_dims = [dim for dim in parent_data.dims if dim in other_data.dims]
        if dims is None:
            dims = shared_dims
        else:
            dims = [dims] if isinstance(dims, str) else list(dims)
            missing_dims = [dim for dim in dims if dim not in shared_dims]
            if missing_dims:
                raise ValueError(
                    f"{self.format_call()}: dimensions are not shared by both "
                    f"signals: {missing_dims!r}."
                )

        if not dims:
            raise ValueError(
                f"{self.format_call()}: signals have no shared dimensions to align."
            )

        parent_indices = {}
        other_indices = {}
        for dim in dims:
            parent_contract = self._coord_contract(parent_schema, dim)
            other_contract = self._coord_contract(other_schema, dim)
            if parent_contract.ndim != 1 or other_contract.ndim != 1:
                raise self._mismatched_grid_error(
                    dim, "matching multidimensional coordinates is not supported"
                )

            parent_values, other_values = self._coordinates_in_common_units(
                parent_data.coords[dim], other_data.coords[dim], dim
            )

            parent_is_numeric = np.issubdtype(parent_values.dtype, np.number)
            other_is_numeric = np.issubdtype(other_values.dtype, np.number)
            if parent_is_numeric != other_is_numeric:
                raise self._mismatched_grid_error(
                    dim, "one coordinate is numeric and the other is not"
                )
            if parent_is_numeric and tolerance_decimals is not None:
                parent_values = parent_values.round(tolerance_decimals)
                other_values = other_values.round(tolerance_decimals)

            indices = self._overlap_indices(
                parent_values,
                other_values,
                dim,
                parent_contract,
                other_contract,
            )

            parent_indices[dim], other_indices[dim] = indices

        parent_overlap = parent_data.isel(parent_indices)
        other_overlap = other_data.isel(other_indices)
        aligned_coords = {dim: parent_overlap.coords[dim] for dim in dims}
        other_overlap = other_overlap.assign_coords(aligned_coords)
        aligned_units = {
            dim: parent_overlap.coords[dim].pint.units
            for dim in dims
            if parent_overlap.coords[dim].pint.units is not None
        }
        if aligned_units:
            other_overlap = other_overlap.pint.quantify(aligned_units)
        return parent_overlap, other_overlap

    def validate_sig_types(self, signals):

        from ..signals import BinarySignal

        for signal in signals:
            if not isinstance(signal, BinarySignal):
                raise TypeError(f"{signal} is not of type BinarySignal.")


class Intersection(Calculator, BinaryCalculatorMixin):
    name = "intersection"

    def __init__(self, *, tolerance_decimals=9):
        self._validate_configuration(tolerance_decimals)
        self.tolerance_decimals = tolerance_decimals

    def _validate_configuration(self, tolerance_decimals):
        if isinstance(tolerance_decimals, bool) or not isinstance(
            tolerance_decimals, int
        ):
            raise TypeError(
                f"{self.format_call()}: tolerance_decimals must be an integer."
            )
        if tolerance_decimals < 0:
            raise ValueError(
                f"{self.format_call()}: tolerance_decimals cannot be negative."
            )

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
            origin=a.origin,
            transformer=self,
            apply_kwargs={"data_schemas": (a.data_schema, b.data_schema)},
        )

    def _apply_inner(self, a_data, b_data, *, data_schemas):
        a_overlap, b_overlap = self._align_overlapping_data(
            a_data,
            b_data,
            *data_schemas,
            tolerance_decimals=self.tolerance_decimals,
        )
        return a_overlap.copy(data=a_overlap.data & b_overlap.data)
    
    def _get_transform(self, *args, apply_kwargs=None, **kwargs):
        return Transform(partial(self._apply, **(apply_kwargs or {})))


class ApplyMask(Calculator, BinaryCalculatorMixin):
    name = "apply_mask"

    def __call__(self, input, mask, *, key=None, key_output_mode=None):
        
        if key is not None or key_output_mode is not None:
            raise NotImplementedError("Key access is not yet implemented for ApplyMask")

        key_spec = KeySpec(input_name=key, output_mode=key_output_mode)

        self.validate_sig_types([mask])
        output_class = self.resolve_output_class(input)

        return output_class(
            inputs=(input, mask),
            transformer=self,
            transform=None,
            data_schema=None,
            key_spec=key_spec,
            origin=input.origin,
            apply_kwargs={"data_schemas": (input.data_schema, mask.data_schema)},
        )
    
    def _get_transform(self, *args, apply_kwargs=None, **kwargs):
        return Transform(partial(self._apply, **(apply_kwargs or {})))

    def _apply(self, sig_data, mask_data, *args, data_schemas, **kwargs):
        if kwargs.get("key_spec") is not None:
            raise NotImplementedError("ApplyMask can't extract a keyed result from a Dataset" \
            "Signal yet")
        
        result = self._apply_inner(
            sig_data, mask_data, data_schemas=data_schemas
        )
        result = self._wrap_result(result, sig_data)

        return result

    def _apply_inner(self, sig_data, mask_data, *, data_schemas):
        _, mask_overlap = self._align_overlapping_data(
            sig_data, mask_data, *data_schemas
        )
        # PintIndex does not implement reindexing, so temporarily expose its
        # magnitudes while filling the part of the signal outside the mask.
        full_mask = (
            mask_overlap.pint.dequantify()
            .reindex_like(sig_data.pint.dequantify(), fill_value=False)
            .pint.quantify()
        )
        result = sig_data.where(full_mask, other=np.nan)
        return result
