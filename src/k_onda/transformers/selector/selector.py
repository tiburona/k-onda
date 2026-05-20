from copy import deepcopy
from functools import reduce
import xarray as xr
from operator import and_, or_
import numpy as np
from collections import defaultdict

from k_onda.loci import IntervalSet
from ..core import Transformer, Transform, Calculator
from k_onda.graph import list_nodes, walk_tree, rebuild_tree
from k_onda.central import (
    Schema,
    DatasetSchema,
    type_registry,
    DimBounds,
    AxisInfo,
    AxisKind,
    CoordInfo,
)

# Why it requires three different classes to accomplish selection:
# The first, Selector, marks the user's intention to select with whatever
# configuration they chose.
#
# The second, SelectionPlanner, executes only when the user requests `.data`,
# or calls `plan_selection()` for debugging purposes, because correctly
# calculating padlen and deciding when to trim depend on knowing the entire
# shape of the graph. Specifically, a selection's bounds can depend on whether
# a later selector had a window. Further, you should only trim padding once,
# after every selection has concluded.
#
# The third, Slicer, actually performs select operations on the data array.


@type_registry.register
class Selector(Transformer):
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
        return type_registry.SelectorSignal

    def _validate_input(self, signal, key_spec=None):

        if key_spec and key_spec.input_name is not None:
            raise NotImplementedError(
                "Use signal.payload(key).select(...) or signal[key].select"
            )
        if signal.data_schema.is_point_process() and isinstance(
            self.locus, type_registry.LocusSet
        ):
            raise NotImplementedError(
                "This operation will result in a ragged array and "
                "support for that is not yet implemented."
            )


class SelectionPlanner(Transformer):
    @property
    def fixed_output_class(self):
        return type_registry.SelectorSignal

    def _call_on_signal(self, signal, key_spec=None):

        all_nodes, selector_nodes = self._gather_selectors(signal)
        leaf = signal
        if len(selector_nodes):
            padlen_accumulators = self._accumulate_padlen(all_nodes)
            leaf = self._build_slice_plan(
                all_nodes, selector_nodes, padlen_accumulators
                )
        return leaf

    def _gather_selectors(self, signal):

        all_nodes = [
            node for node in list_nodes(signal) if hasattr(node, "transformer")
        ]

        selector_nodes = [
            node for node in all_nodes if isinstance(node.transformer, Selector)
        ]
        return all_nodes, selector_nodes

    def _accumulate_padlen(self, all_nodes):
        # Imagine 5 (pseudocode) nodes, listed downstream->upstream
        # [select_event(window=w), select_epoch, filter, scale, source]

        # In our example: because window and epoch share a metadim, time, an events
        # window can be outside the range of the epoch. So every slot upstream of
        # window must be padded.

        # If the selector is going to get pushed upstream of filter,
        # which needs padding, it must add padlen to accommodate the filter.  If it
        # stays downstream of filter, it doesn't need to pad.

        padlen_accumulators = [
            DimBounds(metadim_of=node.data_schema.metadim_from) for node in all_nodes
        ]

        for i, node in enumerate(all_nodes):
            if hasattr(node, "transform") and getattr(node.transform, "padlen", None):
                for pa in padlen_accumulators[i + 1 :]:
                    pa += node.transform.padlen

        return padlen_accumulators
    
    def _build_slice_plan(self, all_nodes, selector_signals, padlen_accumulators):

        pushdown_selector_nodes = []
        local_selector_nodes = []

        for s in selector_signals:
            if s.transformer.mode == "pushdown":
                pushdown_selector_nodes.append(s)
            else:
                local_selector_nodes.append(s)

        pushdown_selector_nodes.reverse()
        local_selector_nodes.reverse()

        slicer_plan = defaultdict(list)

        leaf = all_nodes[0]

        node_index = {id(node): i for i, node in enumerate(all_nodes)}
       
        if len(pushdown_selector_nodes):
            self._make_pushdown_slicers(
                pushdown_selector_nodes, padlen_accumulators, node_index, slicer_plan
            )

        if len(local_selector_nodes):
            self._make_local_slicers(
                local_selector_nodes, padlen_accumulators, node_index, slicer_plan 
            )

        self._make_trim_slicers(leaf, pushdown_selector_nodes, local_selector_nodes, slicer_plan)

        leaf = self._rebuild_tree(leaf, slicer_plan) 

        return leaf

    def _make_local_slicers(
        self, local_selector_nodes, padlen_accumulators, node_index, slicer_plan
    ):

        for ls_node in local_selector_nodes:
            padlen = padlen_accumulators[node_index[id(ls_node)]]
            self._make_slicer(ls_node, ls_node, slicer_plan, padlen=padlen)

    def _make_pushdown_slicers(
        self,
        pushdown_selector_nodes,
        padlen_accumulators,
        node_index,
        slicer_plan
    ):
        # for each pushdown selector, walk the graph from the selector node
        # carrying the set of dims still looking for placement

        for ps_node in pushdown_selector_nodes:

            def _check_and_make(node, last_node, value):
                return self._check_and_make(
                    node,
                    last_node,
                    value,
                    ps_node,
                    padlen_accumulators,
                    node_index,
                    slicer_plan,
                )

            live_dims = set(ps_node.transformer.locus.dim_bounds)

            for _, live_dims, _ in walk_tree(
                ps_node,
                func=_check_and_make,  # signature: (node, downstream, live_dims)
                starting_val=live_dims,
            ):
                pass

    def _check_and_make(
        self,
        node,
        _,
        live_dims,
        ps_node,
        padlen_accumulators,
        node_index,
        slicer_plan,
    ):

        selectable_dims = node.data_schema.selectable

        def dim_in_inputs(node, dim):
            return any(dim in inp.data_schema.selectable for inp in node.inputs)

        if not live_dims:
            return live_dims
        for dim in selectable_dims:
            if dim not in live_dims:
                continue
            elif not node.is_source and dim_in_inputs(node, dim):
                continue
            elif live_dims:
                padlen = padlen_accumulators[node_index[id(node)]]
                self._make_slicer(node, ps_node, slicer_plan, padlen=padlen)

                live_dims.remove(dim)
            else:
                break

        return live_dims

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
        slicer = Slicer(
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


class Slicer(Calculator):
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
                coords=(
                    CoordInfo(name=self.new_dim), 
                    *[CoordInfo(name=condition) for condition in self.locus.member_condition_names]
                    )
            )
            arr_schema = arr_schema.with_added(new_axis)
            arr_schema = arr_schema.rename_axis(
                arr_schema.concrete_dim_from(self.locus.dim), self._new_dim_coord()
                )

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

        if isinstance(data_schema, type_registry.DatasetSchema):
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

    def select_continuous(self, data, data_schema):

        original_dim = data_schema.concrete_dim_from(self.locus.dim)

        selection_bounds = self.selection_bounds.to_array_of_dicts()

        selected = []

        for bounds in selection_bounds:

            mask = self.make_bounds_mask_over_dims(data, bounds)

            sliced = data.where(mask, drop=True)

            if any(size == 0 for size in sliced.sizes.values()):
                continue

            for d in list(sliced.dims):
                if d != original_dim and sliced.sizes[d] == 1:
                    sliced = sliced.squeeze(d)
            selected.append(sliced)

        if not selected:
            raise ValueError(
                "Selection produced no data in the current `selection_bounds`."
            )

        if not self.new_dim:
            selected = self.concat_or_extract(selected)
            return selected
        
        parent_coords = self.parent_metadata_coords(data, data_schema)

        selected = self.attach_continuous_relative_coords(selected)
        selected = self.swap_coords(selected)
        selected = self.concat_or_extract(selected)
        selected = self.unfold_new_dim(selected, data_schema, parent_coords)
        selected = self.attach_condition_coords(selected)

        return selected
    
    def attach_condition_coords(self, selected):
        conditions = reduce(and_, [set(l.conditions.keys()) for l in self.locus])

        selected = selected.assign_coords(
            {
                condition: (
                    self.new_dim, 
                    [l.conditions.get(condition) for l in self.locus]
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

    def unfold_new_dim(self, arr, data_schema, parent_coords=None):

        parent_coords = parent_coords or {}

        new_dim = self.new_dim
        structure = [
            name
            for name, c in arr.coords.items()
            if c.dims == (new_dim,) 
            and name != new_dim
            and name in data_schema.names_by_axis_kind(AxisKind.ORDINAL_INDEX)
        ]

        if not structure:
            arr = arr.assign_coords(parent_coords)
            return arr

        # inner index: position within each unique tuple of structure-values
        keys = list(zip(*[arr.coords[s].data for s in structure]))
        counts = {}
        inner = np.empty(len(keys), dtype=int)
        for i, k in enumerate(keys):
            counts[k] = counts.get(k, -1) + 1
            inner[i] = counts[k]

        inner_name = f"__{new_dim}_inner"  # unique placeholder

        arr = (
            arr.drop_vars(new_dim)
            .assign_coords({inner_name: (new_dim, inner)})
            .set_index({new_dim: structure + [inner_name]})
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

    def concat_or_extract(self, data):

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
                dim=self.new_dim or "intervals",
                combine_attrs="no_conflicts",
                join="exact",
                coords="different",
            )
            selected_data = selected_data.assign_coords(
                {
                    self.new_dim or "intervals": np.arange(
                        selected_data.sizes[self.new_dim or "intervals"]
                    )
                }
            )

        else:
            selected_data = data[0]
        return selected_data

    def select_point_process(self, data, data_schema):

        dim_source_map = {}

        for dim in self.selection_bounds:
            if data_schema.is_value_metadim(dim):
                if isinstance(data_schema, type_registry.DatasetSchema):
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

