from collections.abc import Callable, Sequence
from copy import copy
from itertools import product
from math import ceil
from numbers import Real
import pint
import warnings

import numpy as np
import xarray as xr

from .core import Calculator
from k_onda.central import type_registry, AxisInfo, AxisKind, CoordInfo
from k_onda.utils import is_unitful, w_units


DIM_DEFAULT_UNITS = {"time": "s", "frequency": "Hz"}


@type_registry.register
class Histogram(Calculator):
    name = "histogram"
    key_mode = "standalone"
    accepted_data_types = (xr.DataArray,)
    statistics = ("count", "rate")
    range_sources = ("data", "coords", "session")
    bin_coord_modes = ("left", "center")

    def __init__(
        self,
        *,
        bins: (
            int
            | Sequence[Real | pint.Quantity]
            | np.ndarray
            | pint.Quantity
            | Callable[..., object]
            | None
        ) = None,
        bin_size: Real | pint.Quantity | Callable[..., object] | None = None,
        hist_range: (
            Sequence[Real | pint.Quantity]
            | np.ndarray
            | pint.Quantity
            | Callable[..., object]
            | None
        ) = None,
        stat: str = "count",
        density: bool = False,
        dim: str = "time",
        range_source: str = "data",
        bin_coord: str = "left",
    ):
        self.validate_type_hints()
        self._validate_configuration(
            bins,
            bin_size,
            hist_range,
            stat,
            density,
            dim,
            range_source,
            bin_coord,
        )

        if bins is None and bin_size is None:
            bins = 10

        self.bins = bins
        self.hist_range = hist_range
        self.stat = stat
        self.density = density
        self.dim = dim
        self.bin_size = None if bin_size is None else w_units(bin_size, dim=self.dim)
        self.range_source = range_source
        self.bin_coord = bin_coord

    def _validate_configuration(
        self,
        bins,
        bin_size,
        hist_range,
        stat,
        density,
        dim,
        range_source,
        bin_coord,
    ):
        if bins is not None and bin_size is not None:
            raise ValueError(
                f"{self.format_call()}: provide at most one of bins and bin_size."
            )
        if callable(bin_size):
            raise NotImplementedError(
                f"{self.format_call()}: callable bin_size is not implemented yet."
            )

        self.validate_parameter("stat", stat, choices=self.statistics)
        self.validate_parameter("range_source", range_source, choices=self.range_sources)
        self.validate_parameter("bin_coord", bin_coord, choices=self.bin_coord_modes)
        self.validate_parameter("dim", dim, nonempty_string=True)

        if density and stat == "rate":
            raise ValueError(
                f"{self.format_call()}: density=True cannot be combined with "
                "stat='rate'."
            )

        if isinstance(bins, int):
            self.validate_number("bins", bins, number_type=int, minimum=1)
        if bin_size is not None:
            self.validate_number(
                "bin_size",
                bin_size,
                minimum=0,
                nonzero=True,
                finite=True,
                allow_quantity=True,
            )
        if hist_range is not None and not callable(hist_range):
            values = (
                np.asarray(hist_range.magnitude)
                if isinstance(hist_range, pint.Quantity)
                else np.asarray(hist_range)
            )
            if values.ndim != 1 or values.size != 2:
                raise ValueError(
                    f"{self.format_call()}: hist_range must contain exactly two "
                    "values."
                )

    @property
    def fixed_output_class(self):
        return type_registry.DistributionSignal

    def _validate_data_schema(self, input_schema):
        super()._validate_data_schema(input_schema)
        if input_schema.concrete_dim_from(self.dim) is None:
            raise ValueError(
                f"{self.format_call()}: input schema does not contain a dimension "
                f"representing {self.dim!r}."
            )

    def output_schema(self, input_schema):
        schema = input_schema.without_dim(self.dim)
        metadim = input_schema.metadim_from(self.dim) or input_schema.value_metadim
        if isinstance(self.bins, int) or self.bin_size:
            is_regularly_sampled = True
        elif isinstance(self.bins, Callable):
            is_regularly_sampled = None
        else:
            try:
                bin_arr = np.asarray(self.bins.magnitude)
            except AttributeError:
                bin_arr = np.asarray(self.bins)
            diffs = np.diff(bin_arr)
            is_regularly_sampled = np.allclose(diffs, diffs[0], rtol=10**-5)
            
        schema = schema.with_added(
            AxisInfo(
                f"{self.dim}_bins",
                AxisKind.AXIS,
                metadim=metadim or self.dim,
                coords=(
                    CoordInfo(
                        name=f"{self.dim}_bins", 
                        metadim=metadim, 
                        is_regularly_sampled=is_regularly_sampled
                        ),
                    CoordInfo(
                        name=self.dim,
                        metadim=metadim, 
                        is_regularly_sampled=is_regularly_sampled
                        ),
                ),
            )
        )
        if input_schema.value_metadim:
            schema.value_metadim = f"{input_schema.value_metadim}_{self.stat}" 
        return schema

    def _get_extra_apply_kwargs(self, input):

        extra_kwargs = {}
        
        if self.range_source == "session":
            extra_kwargs["hist_range"] = (
                input.origin.session.start,
                input.origin.session.start + input.origin.session.duration,
            )

        return extra_kwargs

    def _prepare_hist_inputs(self, data, hist_range, data_schema):
        dim = data_schema.concrete_dim_from(self.dim)
        axis = data_schema.axis_position_from(dim)

        if callable(self.hist_range):
            lo, hi = self.hist_range(data, dim=dim)
        elif self.hist_range is not None:
            lo, hi = self.hist_range
        elif self.range_source == "coords":
            coord = np.asarray(data.coords[dim])
            lo, hi = coord[0], coord[-1]
        elif self.range_source == "data":
            lo = data.min().item()
            hi = data.max().item()
        elif self.range_source == "session":
            lo, hi = hist_range
        if not (hi > lo):
            raise ValueError(f"Invalid histogram range: ({lo}, {hi})")

        if callable(self.bins):
            bins = self.bins(data, dim=dim)
        elif self.bins is not None:
            bins = self.bins
        else:
            try:
                bins = ceil((hi - lo) / self.bin_size)
            except pint.DimensionalityError as e:
                if not is_unitful(self.bin_size):
                    lo = lo.magnitude
                    hi = hi.magnitude
                    bin_size = self.bin_size
                elif not all([is_unitful(b) for b in (lo, hi)]):
                    bin_size = self.bin_size.magnitude
                else:
                    raise e

                warnings.warn(
                    "One of histogram bin_size or your hist_range did not have units."
                    "Units were stripped to calculate bins."
                )

                bins = ceil((hi - lo) / bin_size)

        return data, axis, bins, (lo, hi)

    def histogram_along_axis(self, data, bins, axis, hist_range):
        """
        Apply np.histogram to every 1D slice along one axis of an N-D array.

        The axis being histogrammed is replaced by a bins axis in the output.
        """
        data = np.asarray(data.pint.magnitude)
        if is_unitful(hist_range[0]):
            hist_range = np.asarray([b.magnitude for b in hist_range])
        data = np.moveaxis(data, axis, -1)
        outer_shape = data.shape[:-1]
        n_bins = bins if isinstance(bins, int) else len(bins) - 1
        dtype = float if self.density else int
        result = np.empty(outer_shape + (n_bins,), dtype=dtype)
        for idx in product(*(range(s) for s in outer_shape)):
            result[idx], bin_edges = np.histogram(
                data[idx],
                bins=bins,
                range=hist_range,
                density=self.density,
            )
        result = np.moveaxis(result, -1, axis)
        return result, bin_edges

    def _wrap_result(self, result, data, axis, bin_edges, transformed_data, input_schema):
        new_dims = []
        new_coords = {}

        for i, dim in enumerate(transformed_data.dims):
            if i != axis:
                new_dims.append(dim)
                new_coords[dim] = data.coords[dim]
            else:
                new_dim = f"{self.dim}_bins"
                new_dims.append(new_dim)

                if self.bin_coord == "left":
                    new_coords[new_dim] = bin_edges[:-1]
                elif self.bin_coord == "center":

                    def center(i, edges):
                        return (edges[i] + edges[i + 1]) / 2

                    new_coords[new_dim] = [
                        center(i, bin_edges) for i in range(len(bin_edges) - 1)
                    ]

        new_attrs = copy(data.attrs)

        # It can make sense to compute a histogram over the stacked dimension
        # but if you do the unstacked signals are no longer recoverable.
        if "stack_dim" in new_attrs and new_attrs["stack_dim"] == self.dim:
            new_attrs.pop("stack_dim")
            new_attrs.pop("boundaries")

        if is_unitful(self.bins):
            bin_unit = self.bins[0].u
        elif is_unitful(self.bin_size):
            bin_unit = self.bin_size.u
        elif self.dim in DIM_DEFAULT_UNITS:
            bin_unit = DIM_DEFAULT_UNITS[self.dim]
        else:
            raise ValueError("Can't put units back on coords")

       

        result = xr.DataArray(result, dims=new_dims, coords=new_coords, attrs=new_attrs)
        # For instance, even though the new dim is time_bins, make sure 'time' is available
        # as an auxiliary coord for later selection
        result = result.assign_coords(
            {self.dim: (new_dim, result.coords[new_dim].data)}
        )
        result = result.pint.quantify({new_dim: bin_unit, self.dim: bin_unit})

        source_axis = input_schema.ax_with_dim(self.dim)
        ureg = pint.get_application_registry()
        count_unit = getattr(source_axis, "item_unit", ureg.dimensionless)

        data_unit = count_unit/bin_unit if self.stat == "rate" else count_unit
        result = result.pint.quantify(data_unit)

     

        result = super()._wrap_result(result)
        return result

    def _apply_inner(
        self, data, *args, hist_range=None, data_schema, **kwargs
    ):
        data, axis, bins, hist_range = self._prepare_hist_inputs(
            data, hist_range, data_schema
            )

        hist, bin_edges = self.histogram_along_axis(
            data, bins, axis, hist_range
        )

        if self.stat == "rate":
            hist = hist/(bin_edges[1]-bin_edges[0])

        extra_args_for_wrap_result = {
            "axis": axis, 
            "bin_edges": bin_edges, 
            "transformed_data": data, 
            "input_schema": data_schema
            }

        return hist, extra_args_for_wrap_result
