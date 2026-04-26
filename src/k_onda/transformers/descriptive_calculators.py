from copy import copy
from itertools import product
from math import ceil
import pint
import warnings

import numpy as np
import xarray as xr

from .core import Calculator
from k_onda.central import types, AxisInfo, AxisKind
from k_onda.utils import is_unitful, w_units


DIM_DEFAULT_UNITS = {'time': 's', 'frequency': 'Hz'}


@types.register
class ReduceDim(Calculator):
    name = "reduce_dim"

    def __init__(self, dim, method="mean", weights=None):
        self.dim = dim
        self.method = method
        self.weights = weights

    def _apply_inner(self, data, *args, **kwargs):
        if self.weights is not None:
            data = data.weighted(self.weights, keep_attrs=True)
        return getattr(data, self.method)(dim=self.dim, keep_attrs=True)
    
    def output_schema(self, input_schema):
        return input_schema.without(self.dim)
    
    def resolve_output_class(self, input):
        # If we're operating on a StackedSignal, preserve the stack type.
        # If this calculator has a fixed output class, return that.
        # Otherwise ask the parent signal what it would produce.
        if getattr(input, "is_stack", False):
            return type(input)
        return self.fixed_output_class or self._infer_output_class(input)

    def _infer_output_class(self, input):

        if isinstance(input.data_schema, types.Schema) and len(input.data_schema.axes) == 1:
            return types.ScalarSignal
        if input.data_schema.is_point_process():
            if not input.data_schema.is_point_process_essential(self.dim):
                return types.PointProcessSignal
            else:
                if input.data_schema.has_dim('time'):
                    return types.TimeSeriesSignal
                else:
                    return types.Signal
        return super()._infer_output_class(input)

        

@types.register
class Histogram(Calculator):
    name = "histogram"

    def __init__(
        self,
        bins=None,
        bin_size=None,
        hist_range=None,
        density=False,
        dim="time",
        range_source="data",
        bin_coord="left",
    ):
        # bins int, None, or callable that operates on parent.data
        # bin_size float, None, or callable that operates on parent.data
        # range tuple or callable that operates on parent.data
        # weights None or callable that operates on parent.data
        self.bins = bins
        self.hist_range = hist_range
        self.weights = None
        self.density = density
        self.dim = dim
        self.bin_size = w_units(bin_size, dim=self.dim)
        self.range_source = range_source
        self.bin_coord = bin_coord

        if self.bins is None and self.bin_size is None:
            raise ValueError("One of `bins` or `bin_size` must not be None.")
        if self.bins and self.bin_size:
            raise ValueError("Provide `bins` or `bin_size`, not both.")

    @property
    def fixed_output_class(self):
        return types.DistributionSignal
    
    def output_schema(self, input_schema):
        schema = input_schema.without(self.dim)
        # TODO: should I write an metadim_from(dim) method on schema?
        axis = input_schema.axis_by_name(self.dim)
        metadim = axis.metadim if axis else input_schema.value_metadim
        schema = schema.with_added(
            AxisInfo(
                f'{self.dim}_bins',
                AxisKind.AXIS,
                metadim=metadim or self.dim
            )
        )
        return schema
        
    def _get_extra_apply_kwargs(self, input):
        
        extra_kwargs = {'is_point_process': isinstance(input, types.PointProcessSignal)}

        # TODO: eventually there should be other string range sources and/or
        # a concept of finding the range from the nearest bound container 
        # along dim, but since this is mostly an issue for point process
        # signals and selection of ragged arrays is deferred on purpose,
        # this is a later problem.
        if self.range_source == 'session':
            extra_kwargs['hist_range'] = (
                input.origin.session.start, 
                input.origin.session.start + input.origin.session.duration
                )

        return extra_kwargs


    def _prepare_hist_inputs(self, data, hist_range, data_schema):
    
        if isinstance(data, xr.Dataset):
            key = data_schema.default_variable_for(self.dim)
            data = data[key]
            data_schema = data_schema[key]

        dim = data_schema.concrete_dim_from(self.dim)
        axis = data_schema.axis_position_from(dim)

        if callable(self.hist_range):
            lo, hi = self.hist_range(data, dim=dim)
        elif self.hist_range is not None:
            lo, hi = self.hist_range
        elif self.range_source == 'coords':
            coord = np.asarray(data.coords[dim])
            lo, hi = coord[0], coord[-1]
        elif self.range_source == 'data':
            lo = data.min().item()
            hi = data.max().item()
        elif self.range_source == 'session':
            lo, hi = hist_range
        else:
            raise ValueError(f"Unknown range_source '{self.range_source}'")
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
                elif not all([is_unitful(b) for b in (lo, hi)]): 
                    bin_size = self.bin_size.magnitude
                else: 
                    raise e
                                               
                warnings.warn("One of histogram bin_size or your hist_range did not have units."
                    "Units were stripped to calculate bins.")
                
                bins = ceil((hi - lo) / bin_size)

        if callable(self.weights):
            weights = self.weights(data, dim=dim)
        else:
            weights = self.weights
        if weights is not None:
            weights = np.asarray(weights)
            if weights.shape not in ((data.shape[axis],), data.shape):
                raise ValueError(
                    f"weights of shape {weights.shape} not compatible with data of shape {data.shape}"
                )

        return data, axis, bins, (lo, hi), weights

    @staticmethod
    def histogram_along_axis(data, bins, axis, hist_range, weights, density):
        """
        Apply np.histogram to every 1D slice along one axis of an N-D array.

        The axis being histogrammed is replaced by a bins axis in the output.
        """
        if weights is not None and weights.shape == data.shape:
            slice_weights = True
            weights = np.moveaxis(weights, axis, -1)
        else:
            slice_weights = False

        data = np.asarray(data)
        if is_unitful(hist_range[0]):
            hist_range = np.asarray([b.magnitude for b in hist_range])
        data = np.moveaxis(data, axis, -1)
        outer_shape = data.shape[:-1]
        n_bins = bins if isinstance(bins, int) else len(bins) - 1
        dtype = float if density or weights is not None else int
        result = np.empty(outer_shape + (n_bins,), dtype=dtype)
        for idx in product(*(range(s) for s in outer_shape)):
            result[idx], bin_edges = np.histogram(
                data[idx],
                bins=bins,
                range=hist_range,
                weights=weights if not slice_weights else weights[idx],
                density=density,
            )
        # TODO should I assign units here like of 1/the bin dimension unit?
        result = np.moveaxis(result, -1, axis)
        return result, bin_edges

    def _wrap_result(self, result, data, axis, bin_edges, transformed_data):
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
                    center = lambda i, edges: (edges[i] + edges[i + 1]) / 2
                    new_coords[new_dim] = [
                        center(i, bin_edges) for i in range(len(bin_edges) - 1)
                    ]
                else:
                    raise ValueError("Unknown bin coord")
                
        if is_unitful(self.bins):
            units = self.bins[0].u
        elif is_unitful(self.bin_size):
            units = self.bin_size.u
        elif self.dim in DIM_DEFAULT_UNITS:
            units = DIM_DEFAULT_UNITS[self.dim]
        else:
            raise ValueError("Can't put units back on coords")

        new_attrs = copy(data.attrs)

        # It can make sense to compute a histogram over the stacked dimension
        # but if you do the unstacked signals are no longer recoverable.
        if "stack_dim" in new_attrs and new_attrs["stack_dim"] == self.dim:
            new_attrs.pop("stack_dim")
            new_attrs.pop("boundaries")

        result = xr.DataArray(result, dims=new_dims, coords=new_coords, attrs=new_attrs)
        # For instance, even though the new dim is time_bins, make sure 'time' is available
        # as an auxiliary coord for later selection
        result = result.assign_coords({self.dim: (new_dim, result.coords[new_dim].data)})
        result = result.pint.quantify({new_dim: units, self.dim: units})

        result = super()._wrap_result(result)
        return result

    def _apply_inner(self, data, *args, **kwargs):
        hist_range = kwargs.get('hist_range')
        data_schema = kwargs.get('data_schema')


        data, axis, bins, hist_range, weights = self._prepare_hist_inputs(
            data, hist_range, data_schema
            )

        hist, bin_edges = self.histogram_along_axis(
            data, bins, axis, hist_range, weights, self.density
        )

        return hist, {'axis': axis, 'bin_edges': bin_edges, 'transformed_data': data}
    
