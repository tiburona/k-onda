from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pint
from collections.abc import Sequence
import xarray as xr


from .core import Calculator
from k_onda.central import Schema, DatasetSchema, CoordInfo
from k_onda.central import type_registry as tr
 

XrData = xr.DataArray | xr.Dataset
Indexers = dict[str, np.ndarray]

if TYPE_CHECKING:
    from k_onda.signals import Signal


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
    
    @staticmethod
    def _coord_values(coord: xr.DataArray) -> tuple[np.ndarray, pint.Unit]:
        units = coord.pint.units
        values = np.asarray(coord.pint.magnitude if units is not None else coord)
        return values, units

    def _mismatched_grid_error(self, dim: str, detail: str) -> NotImplementedError:
        return NotImplementedError(
            f"{self.format_call()}: matching these coordinate grids is not "
            f"implemented yet on dimension {dim!r}: {detail}."
        )

    def _coordinates_in_common_units(
            self, parent_coord: xr.DataArray, other_coord: xr.DataArray, dim: str
            ) -> tuple[np.ndarray, np.ndarray]:
        
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

    def _coord_contract(self, schema: Schema, dim: str) -> CoordInfo:
        contract = schema.coord_by_name(dim)
        if contract is None:
            raise ValueError(
                f"{self.format_call()}: the data schema does not describe "
                f"coordinate {dim!r}."
            )
        return contract

    def _ordered_overlap_indices(
        self,
        parent_values: np.ndarray,
        other_values: np.ndarray,
        dim: str,
        parent_ordering: str,
        other_ordering: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        
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

    def _unique_overlap_indices(
        self, reference_values: np.ndarray, other_values: np.ndarray, dim: str
        ) -> tuple[np.ndarray, np.ndarray]:

        # get_indexer returns each reference value's position in the other coordinate,
        # or -1 when that value is absent.
        reference_index = pd.Index(reference_values)
        other_index = pd.Index(other_values)
        if not reference_index.is_unique or not other_index.is_unique:
            raise self._mismatched_grid_error(
                dim,
                "coordinate values are not unique at the selected matching precision",
            )

        try:
            other_positions = other_index.get_indexer(reference_index)
        except pd.errors.InvalidIndexError as error:
            raise self._mismatched_grid_error(
                dim, "coordinate matching is ambiguous"
            ) from error

        reference_indices = np.flatnonzero(other_positions >= 0)
        if not len(reference_indices):
            raise ValueError(
                f"{self.format_call()}: coordinates do not overlap on dimension "
                f"{dim!r}."
            )
        return reference_indices, other_positions[reference_indices]

    def _overlap_indices(
        self,
        reference_values: np.ndarray,
        other_values: np.ndarray,
        dim: str,
        reference_contract: CoordInfo,
        other_contract: CoordInfo,
    ) -> tuple[np.ndarray, np.ndarray]:
        
        if np.array_equal(reference_values, other_values):
            indices = np.arange(len(reference_values))
            return indices, indices

        if reference_contract.is_monotonic and other_contract.is_monotonic:
            return self._ordered_overlap_indices(
                reference_values,
                other_values,
                dim,
                reference_contract.ordering,
                other_contract.ordering,
            )

        if reference_contract.is_unique and other_contract.is_unique:
            return self._unique_overlap_indices(reference_values, other_values, dim)

        raise self._mismatched_grid_error(
            dim, "matching coordinates that may contain repeats is ambiguous"
        )

    def _matching_coordinate_indices(
        self,
        reference_data: XrData,
        other_data: XrData,
        reference_schema: Schema,
        other_schema: Schema,
        dims: str | Sequence[str] | None = None,
        tolerance_decimals: int | None = None 
    ) -> tuple[Indexers, Indexers]:
        
        shared_dims = [dim for dim in reference_data.dims if dim in other_data.dims]
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

        reference_indices = {}
        other_indices = {}
        for dim in dims:
            reference_contract = self._coord_contract(reference_schema, dim)
            other_contract = self._coord_contract(other_schema, dim)
            if reference_contract.ndim != 1 or other_contract.ndim != 1:
                raise self._mismatched_grid_error(
                    dim, "matching multidimensional coordinates is not supported"
                )

            reference_values, other_values = self._coordinates_in_common_units(
                reference_data.coords[dim], other_data.coords[dim], dim
            )

            reference_is_numeric = np.issubdtype(reference_values.dtype, np.number)
            other_is_numeric = np.issubdtype(other_values.dtype, np.number)
            if reference_is_numeric != other_is_numeric:
                raise self._mismatched_grid_error(
                    dim, "one coordinate is numeric and the other is not"
                )
            if reference_is_numeric and tolerance_decimals is not None:
                reference_values = reference_values.round(tolerance_decimals)
                other_values = other_values.round(tolerance_decimals)

            indices = self._overlap_indices(
                reference_values,
                other_values,
                dim,
                reference_contract,
                other_contract,
            )

            reference_indices[dim], other_indices[dim] = indices

        return reference_indices, other_indices

    def _align_overlapping_data(
        self,
        reference_data: XrData,
        other_data: XrData,
        reference_schema: Schema,
        other_schema: Schema,
        dims: str | Sequence[str] | None = None,
        tolerance_decimals: int | None = None
    ) -> tuple[XrData, XrData]:

        reference_indices, other_indices = self._matching_coordinate_indices(
            reference_data, other_data, reference_schema, other_schema, dims, tolerance_decimals
        )

        dims = reference_indices.keys()

        reference_overlap = reference_data.isel(reference_indices)
        other_overlap = other_data.isel(other_indices)
        aligned_coords = {dim: reference_overlap.coords[dim] for dim in dims}
        other_overlap = other_overlap.assign_coords(aligned_coords)
        aligned_units = {
            dim: reference_overlap.coords[dim].pint.units
            for dim in dims
            if reference_overlap.coords[dim].pint.units is not None
        }
        if aligned_units:
            other_overlap = other_overlap.pint.quantify(aligned_units)
        return reference_overlap, other_overlap

    def validate_sig_types(self, signals: Sequence[Signal]) -> None:

        for signal in signals:
            if not isinstance(signal, tr.BinarySignal):
                raise TypeError(f"{signal} is not of type BinarySignal.")

    def _get_extra_apply_kwargs(self, *inputs):
        apply_kwargs = super()._get_extra_apply_kwargs(*inputs)
        apply_kwargs["data_schemas"] = [input.data_schema for input in inputs]
        return apply_kwargs


class Intersection(BinaryCalculatorMixin, Calculator):
    name = "intersection"
    arity = "two_or_more"

    def __init__(self, *, tolerance_decimals=9):
        self._validate_configuration(tolerance_decimals)
        self.tolerance_decimals = tolerance_decimals

    def _validate_input(self, *inputs, **kwargs):
        super()._validate_input(*inputs, **kwargs)

        if any(not isinstance(input, tr.BinarySignal) for input in inputs):
            raise TypeError(f"{self.format_call()}: all inputs must be of type BinarySignal.")
        
        if any(isinstance(input.data_schema, DatasetSchema) for input in inputs):
            raise NotImplementedError(
                f"{self.format_call()}: Datasets are not yet supported for BinaryCalculators."
                )


    def _apply_inner(self, *input_data, data_schemas, data_schema=None):

        a_data = input_data[0]
        a_data_schema = data_schemas[0]

        for b_data, b_data_schema in zip(input_data[1:], data_schemas[1:]):
            a_overlap, b_overlap = self._align_overlapping_data(
                a_data,
                b_data,
                a_data_schema,
                b_data_schema,
                tolerance_decimals=self.tolerance_decimals,
            )

            a_data = a_overlap.copy(data=a_overlap.data & b_overlap.data)

        return a_data


class ApplyMask(BinaryCalculatorMixin, Calculator):
    name = "apply_mask"
    arity = "two"

    def __init__(self, *, tolerance_decimals=9):
        self._validate_configuration(tolerance_decimals)
        self.tolerance_decimals = tolerance_decimals

    def _validate_input(self, *inputs, **kwargs):
        super()._validate_input(*inputs, **kwargs)
        if not isinstance(inputs[1], tr.BinarySignal):
            raise TypeError(
                f"{self.format_call()}: second signal input must be of type BinarySignal. "
                f"Received {type(inputs[1])}"
            )
        if any(isinstance(input.data_schema, DatasetSchema) for input in inputs):
            raise NotImplementedError(
                f"{self.format_call()}: Datasets are not yet supported for BinaryCalculators."
                )

    def _apply_inner(self, sig_data, mask_data, *, data_schemas, data_schema=None):
        _, mask_overlap = self._align_overlapping_data(
            sig_data, mask_data, *data_schemas, tolerance_decimals=self.tolerance_decimals
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
