from copy import deepcopy, copy
from functools import partial
from functools import reduce
import xarray as xr
from collections import defaultdict
from operator import and_
import numpy as np

from k_onda.loci import  IntervalSet
from ..core import Transformer, Transform, Calculator
from k_onda.graph import list_nodes, walk_tree, build_consumers_map
from k_onda.central import Schema, DatasetSchema, types, DimBounds, AxisInfo, AxisKind, CoordInfo


# TODO: Selector was working for multiple dims, but work on Interval selection
# has resulted in regression.  To think about later: support for ragged arrays.   

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

@types.register
class Selector(Transformer):

    name = 'selector'

    def __init__(self, mode='local', locus=None, new_dim=None, window=None):
        self.mode = mode
        self.locus = locus
        self.new_dim = new_dim
        self.window = window
      
    def _call_on_signal(self, signal, key_spec):
        output = super()._call_on_signal(signal, key_spec)
        if hasattr(self.locus, 'conditions'):
            output.conditions.update(self.locus.conditions)
        return output
      
    def _get_transform(self, signal, key_spec):
        return Transform(fn = lambda x: x, padlen=self.window)
    
    @property
    def fixed_output_class(self):
        return types.SelectorSignal
   
    def _validate_input(self, signal, key_spec=None):

        if key_spec and key_spec.input_name is not None:
            raise NotImplementedError(
                "Use signal.payload(key).select(...) or signal[key].select"
                )
        if (signal.data_schema.is_point_process() and 
            isinstance(self.locus, types.LocusSet)):
            raise NotImplementedError("This operation will result in a ragged array and " \
            "support for that is not yet implemented.")


class SelectionPlanner(Transformer):

    @property
    def fixed_output_class(self):
        return types.SelectorSignal

    def _call_on_signal(self, signal, key_spec=None):

        all_nodes, selector_nodes = self._gather_selectors(signal)
        leaf = signal
        if len(selector_nodes):
            padlen_accumulators = self._accumulate_padlen(signal, all_nodes)
            lower_bound_offsets = self._lower_bound_correction(selector_nodes)
            leaf = self._place_slicers(
                all_nodes, selector_nodes, padlen_accumulators, lower_bound_offsets
                )
        return leaf
            
    def _gather_selectors(self, signal):

        all_nodes = [node for node in list_nodes(signal) if hasattr(node, 'transformer')]
        
        selector_nodes = [
            node for node in all_nodes if isinstance(node.transformer, Selector)
            ]
        return  all_nodes, selector_nodes

    # TODO: I'm not sure that signal as the source of the metadim_of callable 
    # makes sense forever, at least in Schema's current state.
    # It's possible to have a pushdown selector that destroys dims downstream 
    # that are selectable upstream.  This needs more thought, but I need to check
    # that a data schema can do metadim translation on all the dims a signal's 
    # ancestors had.  Deferred for now.

    def _accumulate_padlen(self, signal, all_nodes):
        # Imagine 5 (pseudocode) nodes, listed downstream->upstream
        # [select_event(window=w), select_epoch, filter, scale, source]

        # In our example: because window and epoch share a metadim, time, an events 
        # window can be outside the range of the epoch. So every slot upstream of 
        # window must be padded.

        # If the selector is going to get pushed upstream of filter, 
        # which needs padding, it must add padlen to accommodate the filter.  If it 
        # stays downstream of filter, it doesn't need to pad.

        padlen_accumulators = [
            DimBounds(metadim_of=signal.data_schema.metadim_from) 
            for _ in range(len(all_nodes))
            ]
    
        for i, node in enumerate(all_nodes):
            if hasattr(node, 'transform') and getattr(node.transform, 'padlen', None): 
                for pa in padlen_accumulators[i+1:]: 
                    pa += node.transform.padlen

        return padlen_accumulators

    
    def _lower_bound_correction(self, selector_nodes):
        offsets = {}

        for selector_node in selector_nodes:
            if selector_node.transformer.window:
                metadim = selector_node.transformer.locus.metadim  
                if metadim in offsets:
                    continue
                offsets[metadim] = selector_node.transformer.window[metadim][0]
        return offsets
    
    def _place_slicers(
            self, 
            all_nodes,
            selector_signals, 
            padlen_accumulators,
            lower_bound_offsets
            ):
        
        pushdown_selector_nodes = []
        local_selector_nodes = []

        for s in selector_signals:
            if s.transformer.mode == 'pushdown':
                pushdown_selector_nodes.append(s)
            else:
                local_selector_nodes.append(s)
        
        pushdown_selector_nodes.reverse()
        local_selector_nodes.reverse()

        leaf = all_nodes[0]
        node_index = {id(node): i for i, node in enumerate(all_nodes)}
        consumers_map = build_consumers_map(leaf)

        if len(pushdown_selector_nodes):
            leaf = self._place_pushdown_slicers(
                leaf,
                pushdown_selector_nodes, 
                padlen_accumulators,
                node_index,
                consumers_map
                )
            
        if len(local_selector_nodes):
           leaf = self._place_local_slicers(
                leaf,
                local_selector_nodes,
                padlen_accumulators,
                node_index, 
                consumers_map
            )

        return self._place_trim_slicers(leaf, pushdown_selector_nodes, lower_bound_offsets)

    def _place_local_slicers(self, leaf, local_selector_nodes, padlen_accumulators, 
                             node_index, consumers_map):
      
        for ls_node in local_selector_nodes:
            padlen = padlen_accumulators[node_index[id(ls_node)]]
            dim_bounds = deepcopy(ls_node.transformer.locus.dim_bounds)
            dim_bounds += padlen
            slicer_signal = self._place_slicer(
                ls_node, ls_node, dim_bounds, consumers=consumers_map[id(ls_node)]
                )
            leaf = slicer_signal
        
        return leaf
    
    def _place_pushdown_slicers(self, leaf, pushdown_selector_nodes, padlen_accumulators, 
                                node_index, consumers_map):
        # for each pushdown selector, walk the graph from the selector node
        # carrying the set of dims still looking for placement

        for ps_node in pushdown_selector_nodes:
          
            selector_ancestors = {id(node) for node in list_nodes(ps_node)}

            _check_and_place = lambda node, last_node, value: self._check_and_place(
                node, last_node, value, ps_node, padlen_accumulators, node_index, 
                selector_ancestors, consumers_map
            )

            live_dims = set(ps_node.transformer.locus.dim_bounds)

            for _, live_dims, _ in walk_tree(
                ps_node, 
                func=_check_and_place,  # signature: (node, downstream, live_dims)
                starting_val=live_dims
                ):
                    pass
            
        return leaf
                        
    def _check_and_place(self, node, _, live_dims, ps_node, padlen_accumulators, 
                         node_index, selector_ancestors, consumers_map):
        
       
        selectable_dims = node.data_schema.selectable
        dim_in_inputs = lambda node, dim: any(dim in inp.data_schema.selectable 
                                              for inp in node.inputs)

        if not live_dims:
            return live_dims
        for dim in selectable_dims:
            if dim not in live_dims:
                continue
            elif not node.is_source and dim_in_inputs(node, dim):
                continue
            elif live_dims and not node.is_source:
                padlen = padlen_accumulators[node_index[id(node)]]
                dim_bounds = deepcopy(ps_node.transformer.locus.dim_bounds)
                dim_bounds += padlen
            
                self._place_slicer(
                    node, ps_node, dim_bounds, selector_ancestors=selector_ancestors, 
                    consumers=consumers_map[id(node)], check_ancestors=True
                    )
                
                live_dims.remove(dim)
            elif live_dims and node.is_source:
                raise ValueError(f"dims {live_dims} were not found.")
            else:
                break
                
        return live_dims

    def _place_trim_slicers(self, leaf, pushdown_selector_nodes, lower_bound_offsets):
        # For every pushdown selector place a slicer with the selector's original 
        # bounds
        for ps_node in pushdown_selector_nodes:
            slicer_signal = self._place_slicer(
                leaf, ps_node, ps_node.transformer.locus.dim_bounds, is_trim=True
                )
            leaf = slicer_signal

        for metadim, offset in lower_bound_offsets.items():
            leaf = CoordTranslator(metadim=metadim, offset=offset)(leaf)
           
        return leaf
    
    def _place_slicer(
            self, 
            node, 
            selector_signal,
            selection_bounds, 
            selector_ancestors=None, 
            consumers=None, 
            check_ancestors=False,
            is_trim=False
            ):
        selector = selector_signal.transformer
        new_dim = selector.new_dim if not is_trim else None
        window = selector.window if not is_trim else None
        slicer_signal = Slicer(
            selection_bounds, selector.mode, selector.locus, new_dim, window, 
            is_trim=is_trim)(node)
        slicer_signal._selection_planned = True
        for consumer in consumers or []:
            if not check_ancestors or id(consumer) in selector_ancestors:
                consumer.inputs = [slicer_signal if inp is node else inp for inp in consumer.inputs]
        return slicer_signal
     

class Slicer(Calculator):

    def __init__(self, selection_bounds, mode, locus, new_dim, window, is_trim=False):
        self.mode = mode
        self.locus = locus
        self.new_dim = new_dim
        self.window = window
        self.is_trim = is_trim
        self.selection_bounds = selection_bounds
        self.multi_select = isinstance(locus, IntervalSet)
        
    def _call_on_signal(self, signal, key_spec=None):
        output_signal = super()._call_on_signal(signal, key_spec=key_spec)
        output_signal = self.start_and_duration(output_signal)
        return output_signal
       
    def make_output_schema(self, data_schema, key_spec):
        if not self.new_dim or self.is_trim:
            return data_schema

        # TODO is this reflecting the actual order of the dims in the xarray data array?
        # I think it's not, and I have code above that assumes the append is in this order
        # I need to check this, and maybe have an index property that's the reverse
        # of when the axes appear in the list self.axes
        def update_schema(arr_schema):
        
            metadim = arr_schema.metadim_from(self.locus.dim)

            # update coords on the old dim
            metadim_axis = arr_schema.axes_by_metadim(metadim)[0]
            coords=(CoordInfo(name=self._new_dim_coord, metadim=metadim, is_relative=True),)
            if f'relative_{metadim}' not in arr_schema.coord_names:
                coords += (CoordInfo(name=f'relative_{metadim}', metadim=metadim, is_relative=True),)
            arr_schema = arr_schema.update_axis_coords(metadim_axis, coords=coords)

            # add the new dim
            new_axis = AxisInfo(name=self.new_dim, metadim=metadim, kind=AxisKind.AXIS)
            arr_schema = arr_schema.with_added(new_axis)

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

        time_dim = next((dim for dim in self.selection_bounds
                         if signal.data_schema.metadim_from(dim) == 'time'
                        ), None)
        
        # TODO: Check the order that dims are added to the schema.  This is only
        # gonna work right if epoch, pip, etc. get added later
        if time_dim:
            start, duration = list(zip(*[
                _start_and_duration(bounds) for bounds in self.selection_bounds[time_dim]
                ]))
            signal.duration = duration
            signal.start = start
        
        return signal

    def _validate_input(self, signal, key_spec=None):
        for dim in set(self.selection_bounds):
            if not signal.data_schema.is_selectable(dim):
                raise ValueError(f"Signal data can not be selected on dimension {dim}.")
            
    def _get_apply_kwargs(self, input, *args):
     
        return {
            'data_schema': getattr(input, 'data_schema')
            }

    def _apply(self, data, data_schema=None):
        # TODO: do I want to add validation of the the window boundaries versus
        # the data boundaries or will the natural error be informative enough?
        
        if isinstance(data_schema, types.DatasetSchema):
        
            if data_schema.is_point_process(require_all=True):
                return self.select_point_process(data, data_schema)
            else:
                if data_schema.is_point_process(require_all=False):
                    raise ValueError("Can't select over datasets with both " \
                    "continuous and point process data.  You need to extract " \
                    "the keys. ")
                else:
                    raise NotImplementedError("Selection over a continuous dataset "
                    "is not yet implemented.")
                    # self.select_continuous(data, data_schema)
        else:
            if data_schema.is_point_process():
                return self.select_point_process(data, data_schema)
            else:
                return self.select_continuous(data, data_schema)

    def select_continuous(self, data, data_schema):

        original_dim = data_schema.concrete_dim_from(self.locus.dim)

        selection_bounds = self.selection_bounds.to_array_of_dicts()

        selected = []

        for bounds in selection_bounds:
            mask = reduce(and_, [
                (data.coords[dim] >= bounds[dim][0]) & (data.coords[dim] < bounds[dim][1])
                for dim in bounds
            ])
            sliced = data.where(mask, drop=True)
            for d in list(sliced.dims):
                if d != original_dim and sliced.sizes[d] == 1:
                    sliced = sliced.squeeze(d)
            selected.append(sliced)

        if self.new_dim:
            
            selected = self.attach_continuous_relative_coords(selected)
            selected = self.swap_coords(selected)

        selected = self.concat_or_extract(selected)
        selected = self.unfold_new_dim(selected) 

        return selected

    def unfold_new_dim(self, arr):
        new_dim = self.new_dim
        structure = [
            name for name, c in arr.coords.items()
            if c.dims == (new_dim,) and name != new_dim
        ]

        if not structure:
            return arr
        
        # inner index: position within each unique tuple of structure-values
        keys = list(zip(*[arr.coords[s].data for s in structure]))
        counts = {}
        inner = np.empty(len(keys), dtype=int)
        for i, k in enumerate(keys):
            counts[k] = counts.get(k, -1) + 1
            inner[i] = counts[k]

        inner_name = f'__{new_dim}_inner' # unique placeholder

        return (
            arr
            .drop_vars(new_dim)
            .assign_coords({inner_name: (new_dim, inner)})
            .set_index({new_dim: structure + [inner_name]})
            .unstack(new_dim)
            .rename({inner_name: new_dim})
        )

    def _new_dim_coord(self):
        return f'{self.new_dim}_{self.locus.metadim}' # e.g., epoch_time
    
    def attach_continuous_relative_coords(self, selected_data):
        result = []
        for i, arr in enumerate(selected_data):
            if i == 0:
                relative_coord = arr.coords[self.locus.dim] - arr.coords[self.locus.dim][0]
        
            # every arr will have coord relative_foo
            arr = arr.assign_coords(
                {f'relative_{self.locus.metadim}':(relative_coord.dims, relative_coord.data)}) 
            
            # for example, if dim is time new_dim is block, now every block will have coord block_time
            if self.new_dim: 
                arr = arr.assign_coords({self._new_dim_coord(): (relative_coord.dims, relative_coord.data)})
            
            result.append(arr)
        
        return result
    
    def swap_coords(self, selected_data):
        # If you've created a new_dim 'block', the main time dim becomes 'block_time', the 
        # time relative to the block, and 'time', the absolute time relative to the session
        # becomes an auxiliary coordinate.

        new_dim_coord = self._new_dim_coord()
        return [(
            arr
            .swap_dims({arr.coords[new_dim_coord].dims[0]: new_dim_coord}) # after swap, dim is epoch_time
            ) for arr in selected_data]
    
    def concat_or_extract(self, data):

        for i, arr in enumerate(data):
            if i == 0:
                canonical_shape = arr.shape
            if arr.shape != canonical_shape:
                raise ValueError("You are calling a method on a ragged array " \
                "that is designed for a uniform one.")

        if self.multi_select:
            selected_data = xr.concat(
                data, 
                dim=self.new_dim or 'intervals', 
                combine_attrs='no_conflicts',
                join='exact',
                coords='different'
            )
            selected_data = selected_data.assign_coords({
                self.new_dim or 'intervals': np.arange(
                    selected_data.sizes[self.new_dim or 'intervals'])
            })
        else:
            selected_data = data[0]
        return selected_data
    

    def select_point_process(self, data, data_schema):

        dim_source_map = {}

        for dim in self.selection_bounds:
            if data_schema.is_value_metadim(dim):
                if isinstance(data_schema, types.DatasetSchema):
                    source = data[data_schema.variable_for_metadim(dim)]
                else:
                    source = data

            else:
                source = data.coords[data_schema.concrete_dim_from(dim)]
            dim_source_map[dim] = source
     
        selection_bounds = self.selection_bounds.to_array_of_dicts()

        selected = []

        for bounds in selection_bounds:
            mask = reduce(and_, [
                (source >= bounds[dim][0]) & (source < bounds[dim][1]) 
                for dim, source in dim_source_map.items()]
                )
            selected.append(data.where(mask, drop=True))

        selected = self.concat_or_extract(selected)

        if self.new_dim:
            selected = self.attach_point_process_relative_coords(selected, selection_bounds)

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


class CoordTranslator(Calculator):

    def __init__(self, metadim, offset):
        self.metadim = metadim
        self.offset = offset

    def _call_on_signal(self, signal, key_spec=None):
        output_signal = super()._call_on_signal(signal, key_spec=key_spec)
        for attr in ('start', 'duration'):
            val = getattr(signal, attr, None)
            if val is not None:
                setattr(output_signal, attr, val)

        return output_signal
    

    def _get_apply_kwargs(self, input, *args):
     
        return {
            'data_schema': getattr(input, 'data_schema')
            }
    
    def _apply(self, data, *, data_schema=None, **_):
       
        ax = next(ax for ax in data_schema.axes if ax.metadim == self.metadim)
        updated = {
            name: data.coords[name] + self.offset
            for name in data.coords
            if data_schema.coord_by_name(name).is_relative 
        }
        
        return data.assign_coords(updated) if updated else data 


      
        
