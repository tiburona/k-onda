from copy import deepcopy
from functools import reduce
import xarray as xr
from operator import and_, or_
import numpy as np
from collections import defaultdict

from k_onda.loci import IntervalSet
from ..core import Transformer, Transform, Calculator
from k_onda.graph import list_nodes, rebuild_tree, walk_graph
from k_onda.central import (
    Schema,
    DatasetSchema,
    type_registry as tr,
    DimBounds,
    DimBoundsArray,
    AxisInfo,
    AxisKind,
    CoordInfo,
)

# Why it requires three different classes to accomplish selection:
# The first, SpecifySelection, marks the user's intention to select with whatever
# configuration they chose.
#
# The second, PlanSelection, executes only when the user requests `.data`,
# or calls `plan_selection()` for debugging purposes, because correctly
# calculating padlen and deciding when to trim depend on knowing the entire
# shape of the graph. Specifically, a selection's bounds can depend on whether
# a later selector had a window. Further, you should only trim padding once,
# after every selection has concluded.
#
# The third, SliceSelection, actually performs select operations on the data array.


@tr.register
class SpecifySelection(Transformer):
    name = "selector"

    def __init__(self, mode="local", locus=None, new_dim=None, window=None):
        self.mode = mode
        self.locus = locus
        self.new_dim = new_dim
        self.window = window

    def _call_on_signal(self, signal, key_spec):
        output = super()._call_on_signal(signal, key_spec)
        if hasattr(self.locus, "conditions"):
            output.conditions.update(self.locus.conditions)
        return output

    def _get_transform(self, *inputs, key_spec=None, **kwargs):
        return Transform(fn=lambda x: x, padlen=self.window, key_spec=key_spec)

    @property
    def fixed_output_class(self):
        return tr.SelectorSignal

    def _validate_input(self, signal, key_spec=None):

        if key_spec and key_spec.input_name is not None:
            raise NotImplementedError(
                "Use signal.payload(key).select(...) or signal[key].select"
            )
        if signal.data_schema.is_point_process() and isinstance(
            self.locus, tr.LocusSet
        ):
            raise NotImplementedError(
                "This operation will result in a ragged array and "
                "support for that is not yet implemented."
            )


class PlanSelection(Transformer):
    @property
    def fixed_output_class(self):
        return tr.SelectorSignal

    def _call_on_signal(self, signal, key_spec=None):

        selector_nodes = [
            node for node in list_nodes(signal) if hasattr(node, "transformer")
            and isinstance(node.transformer, SpecifySelection)
        ]

        leaf = signal

        if len(selector_nodes):
            padlen_accumulator = self._accumulate_padlen(leaf)
            leaf = self._build_slice_plan(leaf, selector_nodes, padlen_accumulator)
        return leaf

    def _accumulate_padlen(self, leaf):

        def merge_state(node, previous_padlen, incoming_padlen):
            merged = deepcopy(previous_padlen)
            return merged.cover_merge(incoming_padlen)

        def step(node, accumulated_padlen, _):
            node_padlen = getattr(node.transform, "padlen", None)
            if node_padlen:
                return accumulated_padlen + node_padlen
            return accumulated_padlen
        
        accumulated_padlen = walk_graph(leaf, DimBounds(), step=step, merge_state=merge_state)

        return accumulated_padlen
    
    def _build_slice_plan(self, leaf, selector_nodes, accumulated_padlen):

        pushdown_selector_nodes = []
        local_selector_nodes = []

        for s in selector_nodes:
            if s.transformer.mode == "pushdown":
                pushdown_selector_nodes.append(s)
            else:
                local_selector_nodes.append(s)

        slicer_plan = defaultdict(list)
       
        if len(pushdown_selector_nodes):
            self._make_pushdown_slicers(
                pushdown_selector_nodes, accumulated_padlen, slicer_plan
            )

        if len(local_selector_nodes):
            self._make_local_slicers(
                local_selector_nodes, accumulated_padlen, slicer_plan 
            )

        self._make_trim_slicers(
            leaf, pushdown_selector_nodes, local_selector_nodes, slicer_plan
            )

        leaf = self._rebuild_tree(leaf, slicer_plan) 

        return leaf

    def _make_local_slicers(
        self, local_selector_nodes, accumulated_padlen, slicer_plan
        ):

        for ls_node in local_selector_nodes:
            padlen = accumulated_padlen[id(ls_node)]
            self._make_slicer(ls_node, ls_node, slicer_plan, padlen=padlen)

    def _make_pushdown_slicers(
            self, selector_nodes, accumulated_padlen, slicer_plan
            ):
        # for each pushdown selector, walk the graph from the selector node
        # carrying the set of dims still looking for placement

        for ps_node in selector_nodes:

            def _check_and_make(node, value, _):
                return self._check_and_make(
                    node,
                    value,
                    ps_node,
                    accumulated_padlen,
                    slicer_plan,
                )

            live_dims = set(ps_node.transformer.locus.dim_bounds)

            walk_graph(ps_node, live_dims, step=_check_and_make)

    def _check_and_make(
        self,
        node,
        live_dims,
        ps_node,
        accumulated_padlen,
        slicer_plan,
    ):

        selectable_dims = node.data_schema.selectable

        def dim_in_inputs(node, dim):
            return any(dim in inp.data_schema.selectable for inp in node.inputs)

        if not live_dims:
            return live_dims
        next_live_dims = set(live_dims)
        for dim in selectable_dims:
            if dim not in next_live_dims:
                continue
            elif not node.is_source and dim_in_inputs(node, dim):
                continue
            elif next_live_dims:
                self._make_slicer(node, ps_node, slicer_plan, accumulated_padlen[id(node)])
                next_live_dims.remove(dim)
            else:
                break

        return next_live_dims

    def _make_trim_slicers(
            self, 
            leaf, 
            pushdown_selector_nodes, 
            local_selector_nodes, 
            slicer_plan
            ):
        # For every pushdown selector place a slicer with the selector's original
        # bounds

        window_nodes = [
            node for node in pushdown_selector_nodes + local_selector_nodes
            if node.transformer.window is not None
            ]
        
        for ps_node in pushdown_selector_nodes:
            trim_bounds = ps_node.transformer.locus.dim_bounds
            for window_node in window_nodes:
                # Detect if the two nodes are attempting to select on different coords
                # on the same dim, where one has a window and one is a pushdown node.  
                # That is a tricky situation we can't yet handle.
                if ps_node.transformer.locus.dim != window_node.transformer.locus.dim:
                    common_metadim = ps_node.data_schema.get_common_metadim(
                        ps_node.transformer.locus.dim, 
                        window_node.data_schema, 
                        window_node.transformer.locus.dim
                        )
                    if common_metadim:
                        raise NotImplementedError(
                            f"You can't yet select on two coords over dim {common_metadim}" 
                            " if they're not the same coord.")
                # Make sure we don't trim more narrowly than the window.
                else:
                    trim_bounds = deepcopy(trim_bounds)
                    trim_bounds.cover(window_node.transformer.locus.dim_bounds)

            self._make_slicer(
                leaf, ps_node, slicer_plan, is_trim=True, trim_bounds=trim_bounds
            )
        
    def _make_slicer(
        self,
        node,
        selector_signal,
        slicer_plan,
        padlen=None,
        is_trim=False,
        trim_bounds=None
    ):
        selector = selector_signal.transformer
        new_dim = selector.new_dim
        window = selector.window if not is_trim else None
        slicer = SliceSelection(
            selector.mode,
            selector.locus,
            new_dim,
            window,
            padlen=padlen,
            is_trim=is_trim,
            trim_bounds=trim_bounds
        )

        slicer_plan[id(node)].append(slicer)
    
        return slicer
    
    def _rebuild_tree(self, leaf, slicer_plan):

        def insert_slicers(original, rebuilt):
            slicers = slicer_plan.get(id(original))
            if not slicers:
                return rebuilt
            for slicer in slicers:
                rebuilt = slicer(rebuilt)
            return rebuilt
        
        new_leaf = rebuild_tree(leaf, rebuild_node=insert_slicers)

        return new_leaf


class SliceSelection(Calculator):
    def __init__(
            self, 
            mode, 
            locus, 
            new_dim, 
            window, 
            padlen=None, 
            is_trim=False,
            trim_bounds=None
            ):
        self.mode = mode
        self.locus = locus
        self.new_dim = new_dim
        self.window = window
        self.padlen=padlen
        self.is_trim = is_trim
        self.trim_bounds = trim_bounds
        self.multi_select = isinstance(locus, IntervalSet)
        self.selection_bounds = self.compute_selection_bounds()

    def _call_on_signal(self, signal, key_spec=None):
        output_signal = super()._call_on_signal(signal, key_spec=key_spec)
        output_signal = self.start_and_duration(output_signal)
        return output_signal

    def make_output_schema(self, data_schema, key_spec):
        if not self.new_dim or self.is_trim:
            return data_schema

        def update_schema(arr_schema):

            metadim = arr_schema.metadim_from(self.locus.dim)

            # update coords on the old dim
            metadim_axis = arr_schema.axes_by_metadim(metadim)[0]
            coords = (
                CoordInfo(
                    name=self._new_dim_coord(), metadim=metadim, is_relative=True
                ),
            )
            if f"relative_{metadim}" not in arr_schema.coord_names:
                coords += (
                    CoordInfo(
                        name=f"relative_{metadim}", metadim=metadim, is_relative=True
                    ),
                )
            arr_schema = arr_schema.update_axis_coords(metadim_axis, coords=coords)

            # add the new dim
            new_axis = AxisInfo(
                name=self.new_dim,
                kind=AxisKind.ORDINAL_INDEX,
                created_from_dim=self.locus.dim,
                created_from_metadim=metadim,
                coords=(
                    CoordInfo(name=self.new_dim), 
                    CoordInfo(f"{self.new_dim}_start_{metadim}", metadim=metadim),
                    CoordInfo(f"{self.new_dim}_stop_{metadim}", metadim=metadim),
                    *[CoordInfo(name=condition) for condition in self.locus.member_condition_names]
                    )
            )
            arr_schema = arr_schema.with_axis(new_axis, if_exists="error")
            arr_schema = arr_schema.rename_axis(
                arr_schema.concrete_dim_from(self.locus.dim), self._new_dim_coord()
                )
            
            dim_order = self.output_dim_order(arr_schema.dim_names, arr_schema)
            arr_schema = arr_schema.reorder_axes(dim_order)

            return arr_schema

        if isinstance(data_schema, Schema):
            return update_schema(data_schema)
        else:
            return DatasetSchema(
                {key: update_schema(val) for key, val in data_schema.items()}
            )


    def start_and_duration(self, signal):

        def _start_and_duration(bounds):
            duration = bounds[1] - bounds[0]
            start = bounds[0]
            return start, duration

        time_dim = next(
            (
                dim
                for dim in self.selection_bounds
                if signal.data_schema.metadim_from(dim) == "time"
            ),
            None,
        )

        if time_dim:
            start, duration = list(
                zip(
                    *[
                        _start_and_duration(bounds)
                        for bounds in self.selection_bounds[time_dim]
                    ]
                )
            )
            signal.duration = duration
            signal.start = start

        return signal

    def _validate_input(self, signal, key_spec=None):
        for dim in set(self.selection_bounds):
            if not signal.data_schema.is_selectable(dim):
                raise ValueError(f"Signal data can not be selected on dimension {dim}.")

    def _get_apply_kwargs(self, input, **kwargs):

        return {"data_schema": getattr(input, "data_schema")}

    def _apply(self, data, data_schema=None):

        if isinstance(data_schema, tr.DatasetSchema):
            if data_schema.is_point_process(require_all=True):
                return self.select_point_process(data, data_schema)
            else:
                if data_schema.is_point_process(require_all=False):
                    raise ValueError(
                        "Can't select over datasets with both "
                        "continuous and point process data.  You need to extract "
                        "the keys. "
                    )
                else:
                    raise NotImplementedError(
                        "Selection over a continuous dataset is not yet implemented."
                    )
        else:
            if data_schema.is_point_process():
                return self.select_point_process(data, data_schema)
            else:
                if self.is_trim:
                    return self.trim_continuous(data)
                return self.select_continuous(data, data_schema)
            
    def trim_continuous(self, data):
        selection_bounds = self.selection_bounds.to_array_of_dicts()

        mask = reduce(
            or_,
            [self.make_bounds_mask_over_dims(data, bounds) 
             for bounds in selection_bounds]
             )
        
        trimmed = data.where(mask, drop=True)
        return trimmed
    
    def compute_selection_bounds(self):
        if self.is_trim:
            return deepcopy(self.trim_bounds)
        
        bounds = deepcopy(self.locus.dim_bounds)
        if self.padlen:
            bounds += self.padlen
        
        return bounds

    def coord_correction(self):
        coord_correction = self.locus.w_units(0.0)
        if self.window:
            coord_correction += self.window[self.locus.dim][0]
        if self.padlen:
            coord_correction += self.padlen[self.locus.dim][0]
        return coord_correction
    
    def mask_by_parent_intervals(self, data_schema):
        if not self.new_dim:
            return False
        
        coord = data_schema.coord_by_name(self.locus.dim)
        if coord and coord.is_relative:
            return False
        
        if not data_schema.ordinal_axes_created_from(self.locus.metadim):
            return False
        
        return True

    def get_parent_interval_masks(self, data, data_schema, selection_bounds):
    
        parent_axes = data_schema.ordinal_axes_created_from(self.locus.metadim)

        ordinal_axis_masks = []

        for ord_ax in parent_axes:
           
            start_coord = data.coords[f"{ord_ax.name}_start_{self.locus.metadim}"]
            stop_coord = data.coords[f"{ord_ax.name}_stop_{self.locus.metadim}"]
            ordinal_bounds = DimBoundsArray.from_coords(
                start=start_coord, stop=stop_coord, dim=self.locus.dim
            )

            # a list of len selection_bounds
            if getattr(self.locus[0], "anchor", None):
                indices_of_parent_intervals = ordinal_bounds.containing_indices(
                    [interval.anchor.value for interval in self.locus], self.locus.dim
                    )
            else:
                indices_of_parent_intervals = ordinal_bounds.containing_indices(
                    selection_bounds, self.locus.dim
                    )
            
            # filter

            empty_mask = xr.full_like(data.coords[ord_ax.name], False, dtype=bool)

            ordinal_bounds_mask = [
                data.coords[ord_ax.name] == data.coords[ord_ax.name][ind]
                if ind is not None
                else empty_mask
                for ind in indices_of_parent_intervals
            ]

            ordinal_axis_masks.append(ordinal_bounds_mask)

        return list(zip(*ordinal_axis_masks))
        
    @staticmethod   
    def make_bounds_mask_over_dims(data, bounds):
        return reduce(
                and_,
                [
                    (data.coords[dim] >= bounds[dim][0])
                    & (data.coords[dim] < bounds[dim][1])
                    for dim in bounds
                ],
            )

    def select_continuous(self, data, data_schema):

        original_dim = data_schema.concrete_dim_from(self.locus.dim)

        selection_bounds = self.selection_bounds.to_array()

        mask_by_parent_intervals = self.mask_by_parent_intervals(data_schema)

        if mask_by_parent_intervals:
            ordinal_bounds = self.get_parent_interval_masks(data, data_schema, selection_bounds)

        selected = []
        kept_indices = []

        for i, bounds in enumerate(selection_bounds):

            mask = self.make_bounds_mask_over_dims(data, bounds)
            if mask_by_parent_intervals:
                for ob_mask in ordinal_bounds[i]:
                    mask = mask & ob_mask

            sliced = data.where(mask, drop=True)

            if any(size == 0 for size in sliced.sizes.values()):
                continue

            for d in list(sliced.dims):
                if d != original_dim and sliced.sizes[d] == 1:
                    sliced = sliced.squeeze(d)
            selected.append(sliced)
            kept_indices.append(i)

        if not selected:
            raise ValueError(
                "Selection produced no data in the current `selection_bounds`."
            )

        if not self.new_dim:
            selected = self.concat_or_extract(selected, kept_indices)
            return selected
        
        parent_coords = self.parent_metadata_coords(data, data_schema)

        selected = self.attach_continuous_relative_coords(selected)
        selected = self.swap_coords(selected)
        selected = self.concat_or_extract(selected, kept_indices)
        selected = self.restore_ordinal_dims(selected, data_schema, parent_coords)
        selected = self.attach_condition_coords(selected, kept_indices)
        selected = self.transpose(selected, data_schema)

        return selected
    
    def attach_condition_coords(self, selected, kept_indices):

        locs = [loc for i, loc in enumerate(self.locus) if i in kept_indices]

        conditions = reduce(and_, [set(loc.conditions.keys()) for loc in locs])

        selected = selected.assign_coords(
            {
                condition: (
                    self.new_dim, 
                    [loc.conditions.get(condition) for loc in locs]
                    ) for condition in conditions
                }
            )

        return selected
    
    def parent_metadata_coords(self, arr, data_schema):
        ordinal_dims = set(data_schema.names_by_axis_kind(AxisKind.ORDINAL_INDEX))
        current_condition_names = self.locus.member_condition_names
        return {
            name: coord 
            for name, coord in arr.coords.items()
            if name not in arr.dims
            and name not in current_condition_names
            and set(coord.dims).issubset(ordinal_dims)
        }

    def restore_ordinal_dims(self, arr, data_schema, parent_coords=None):
        # If child selections were made inside parent ordinal rows, reconstruct 
        # that hierarchy as real dimensions instead of leaving parent identity 
        # as a coordinate on the child dimension.

        parent_coords = parent_coords or {}
        new_dim = self.new_dim

        # any previously created ordinal dims are at this point coords on the new dim
        parent_ordinal_coords = [
            name
            for name, c in arr.coords.items()
            if c.dims == (new_dim,) 
            and name != new_dim
            and name in data_schema.names_by_axis_kind(AxisKind.ORDINAL_INDEX)
        ]

        if not parent_ordinal_coords:
            arr = arr.assign_coords(parent_coords)
            return arr

        # inner index: position within each unique tuple of parent ordinal coords
        keys = list(zip(*[arr.coords[poc].data for poc in parent_ordinal_coords]))
        counts = {}
        index_within_parent_ordinal_coords = np.empty(len(keys), dtype=int)
        for i, k in enumerate(keys):
            counts[k] = counts.get(k, -1) + 1
            index_within_parent_ordinal_coords[i] = counts[k]

        inner_name = f"__{new_dim}_index_within_parent_ordinal_coords"  # unique placeholder

        arr = (
            arr.drop_vars(new_dim)  
            .assign_coords({inner_name: (new_dim, index_within_parent_ordinal_coords)})
            .set_index({new_dim: parent_ordinal_coords + [inner_name]})
            .unstack(new_dim)
            .rename({inner_name: new_dim})
        )

        arr = arr.assign_coords(parent_coords)

        return arr

    def _new_dim_coord(self):
        return f"{self.new_dim}_{self.locus.metadim}"  # e.g., epoch_time

    def attach_continuous_relative_coords(self, selected_data):
        result = []
        for i, arr in enumerate(selected_data):
            if i == 0:
                relative_coord = (
                    arr.coords[self.locus.dim] - 
                    arr.coords[self.locus.dim][0] + 
                    self.coord_correction()
                )
          
            # every arr will have coord relative_foo
            arr = arr.assign_coords(
                {
                    f"relative_{self.locus.metadim}": (
                        relative_coord.dims,
                        relative_coord.data,
                    )
                }
            )

            # for example, if dim is time new_dim is block, now every block will have coord block_time
            if self.new_dim:
                arr = arr.assign_coords(
                    {self._new_dim_coord(): (relative_coord.dims, relative_coord.data)}
                )

            result.append(arr)

        return result

    def swap_coords(self, selected_data):
        # If you've created a new_dim 'block', the main time dim becomes 'block_time', the
        # time relative to the block, and 'time', the absolute time relative to the session
        # becomes an auxiliary coordinate.

        new_dim_coord = self._new_dim_coord()
        swapped = []

        for arr in selected_data:
            old_dim = arr.coords[new_dim_coord].dims[0]
            units = arr.coords[new_dim_coord].pint.units

            arr = arr.swap_dims({old_dim: new_dim_coord})

            if units is not None:
                arr = (
                    arr.drop_indexes(new_dim_coord)
                    .set_xindex(new_dim_coord)
                    .pint.quantify({new_dim_coord: units})
                )

            swapped.append(arr)

        return swapped
    
    def assign_ordinal_coordinates(self, data, kept_indices):
        new_dim = self.new_dim or "interval"

        def make_unitful_ord_coord(ind):
            coord = [bounds[ind] for i, bounds in enumerate(self.locus.dim_bounds[self.locus.dim])
                     if i in kept_indices]
            unit = coord[0].units
            return np.array([q.to(unit).magnitude for q in coord]) * unit

        data = data.assign_coords(
            {
                new_dim: np.arange(data.sizes[new_dim]),
                f"{new_dim}_start_{self.locus.metadim}": (new_dim, make_unitful_ord_coord(0)),
                f"{new_dim}_stop_{self.locus.metadim}": (new_dim, make_unitful_ord_coord(1))
            }
        )

        return data

    def concat_or_extract(self, data, kept_indices):

        for i, arr in enumerate(data):
            if i == 0:
                canonical_shape = arr.shape
            if arr.shape != canonical_shape:
                raise ValueError(
                    "You are calling a method on a ragged array "
                    "that is designed for a uniform one."
                )

        if self.multi_select:

            selected_data = xr.concat(
                data,
                dim=self.new_dim or "interval",
                combine_attrs="no_conflicts",
                join="exact",
                coords="different",
                compat="equals"
            )

            selected_data = self.assign_ordinal_coordinates(selected_data, kept_indices)

        else:
            selected_data = data[0]
        return selected_data

    def select_point_process(self, data, data_schema):

        dim_source_map = {}

        for dim in self.selection_bounds:
            if data_schema.is_value_metadim(dim):
                if isinstance(data_schema, tr.DatasetSchema):
                    source = data[data_schema.variable_for_metadim(dim)]
                else:
                    source = data

            else:
                source = data.coords[data_schema.concrete_dim_from(dim)]
            dim_source_map[dim] = source

        selection_bounds = self.selection_bounds.to_array_of_dicts()

        selected = []

        for bounds in selection_bounds:
            mask = reduce(
                and_,
                [
                    (source >= bounds[dim][0]) & (source < bounds[dim][1])
                    for dim, source in dim_source_map.items()
                ],
            )
            selected.append(data.where(mask, drop=True))

        selected = self.concat_or_extract(selected)

        if self.new_dim:
            selected = self.attach_point_process_relative_coords(
                selected, selection_bounds
            )

        return selected
    
    def output_dim_order(self, dims, data_schema):
        ordinal_dims =  data_schema.names_by_axis_kind(AxisKind.ORDINAL_INDEX)
        
        feature_dims = [
            dim for dim in dims
            if dim not in ordinal_dims
            and dim != self._new_dim_coord()
        ]

        ordered = (
            feature_dims
            + [dim for dim in ordinal_dims if dim in dims]
            + [self._new_dim_coord()]
        )

        return ordered

    def transpose(self, arr, data_schema):
        ordered = self.output_dim_order(arr.dims, data_schema)
        arr = arr.transpose(*[dim for dim in ordered if dim in arr.dims])
        return arr

    def attach_point_process_relative_coords(self, data, selection_bounds):

        # say we've selected on time, and created dim epoch.
        # the arrays now need dim epoch_spikes, epochs
        # there needs to be a key spike_times, and a key epoch_spike_times

        # this can probably be the same logic as attach_continuous_relative_coords
        # except if you've selected on `time` you effectively need a dataset with
        # a new key: `relative_spike_times`
        # in point of fact select point process isn't going to work at all until
        # I implement some kind of solution for ragged arrays so maybe I should
        # just hold off implementing this until then

        pass

