from copy import deepcopy, copy
from functools import partial
from functools import reduce
import xarray as xr
from collections import defaultdict
from operator import and_

from k_onda.loci import  IntervalSet
from .core import Transformer, Transform, PaddingCalculator
from k_onda.graph import list_nodes
from k_onda.central import Schema, DatasetSchema
from k_onda.loci.core import DimBounds



# TODO: Selector was working for multiple dims, but work on Interval selection
# has resulted in regression.  To think about later: support for ragged arrays.   


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
        from k_onda.signals import SelectorSignal
        return SelectorSignal
   
    def _validate_input(self, signal, key_spec=None):

        if key_spec and key_spec.input_name is not None:
            raise NotImplementedError(
                "Use signal.payload(key).select(...) or signal[key].select"
                )
    

class SelectionPlanner(Transformer):

    @property
    def fixed_output_class(self):
        from k_onda.signals import SelectorSignal
        return SelectorSignal

    def _call_on_signal(self, signal, key_spec=None):

        all_nodes, padlen_accumulators, selectors = self._gather_selectors(signal)

        self._accumulate_window_padlen(all_nodes, padlen_accumulators)
        self._accumulate_calculator_padlen(all_nodes, padlen_accumulators)


        self._place_slicers(all_nodes, selectors, padlen_accumulators)
    

    def _gather_selectors(self, signal):

        all_nodes = list_nodes(signal)
        padlen_accumulators = [DimBounds() for _ in range(len(all_nodes))]
        selectors = [node for node in all_nodes if isinstance(node.transformer, Selector)]
        return all_nodes, padlen_accumulators, selectors

    def _accumulate_window_padlen(self, all_nodes, padlen_accumulators):
        for i, node in enumerate(reversed(all_nodes)):
            if isinstance(node.transformer, Selector) and node.transformer.window:
                for pa in padlen_accumulators[i+1:]:
                    pa += node.transformer.window

    def _accumulate_calculator_padlen(self, all_nodes, padlen_accumulators):
        for i, node in enumerate(all_nodes):
            if node.transform.padlen: # e.g time
                for pa in padlen_accumulators[i+1:]: # 
                    pa += node.transform.padlen
    
    def _place_slicers(
            self, 
            all_nodes,
            selectors, 
            padlen_accumulators
            ):

        pushdown_selectors = [s for s in selectors if s.mode == 'pushdown']

        all_nodes = self._place_pushdown_slicers(
            all_nodes, 
            pushdown_selectors, 
            padlen_accumulators
            )
        
        self._place_local_slicers(
            all_nodes, 
            padlen_accumulators
        )

        self._place_trim_slicers(all_nodes, pushdown_selectors)

    def _place_local_slicers(self, all_nodes, padlen_accumulators):
        slicers= []
        for i, node in enumerate(all_nodes):
            if isinstance(node.transformer, Selector) and node.transformer.mode == 'local':
                dim_bounds = deepcopy(node.transformer.locus.metadim_bounds)
                dim_bounds += padlen_accumulators[i]
                slicer = self.place_slicer(
                    node, 
                    all_nodes[i+1], 
                    node.transformer, 
                    dim_bounds
                    )
                slicers.append(slicer)
        all_nodes += slicers
        return all_nodes
    
    def _place_pushdown_slicers(self, all_nodes, pushdown_selectors, padlen_accumulators):
        
        dim_ps_map = defaultdict(list)
        
        for ps in pushdown_selectors:
            for dim in ps.dim_map:
                dim_ps_map[dim].append(ps)

        done_dims_map = defaultdict(list)

        slicers = []

        for i, node in enumerate(all_nodes):
            selectable_dims = node.data_schema.selectable_dims
            for dim in selectable_dims:
                pushdown_selectors = dim_ps_map[dim]
                for ps in pushdown_selectors:
                    if dim in done_dims_map[ps]:
                        continue
                    padlen = padlen_accumulators[i]
                    dim_bounds = deepcopy(ps.locus.metadim_bounds)
                    dim_bounds += padlen
                    next_node = all_nodes[i + 1] if len(all_nodes) > i + 1 else None

                    concrete_dim_bounds = DimBounds(
                        {ps.locus.metadim_to_dim(metadim): bounds 
                         for metadim, bounds in dim_bounds}
                         )

                    slicer = self.place_slicer(node, next_node, ps, concrete_dim_bounds)
                    slicers.append(slicer)
                    
                    done_dims_map[ps].append(dim)

        all_nodes = all_nodes + slicers


        return all_nodes

    def _place_trim_slicers(self, all_nodes, pushdown_selectors):
        # for every pushdown selector place a slicer with the selector's original bounds
        node = all_nodes[-1]
        for ps in pushdown_selectors:
            slicer = self.place_slicer(node, None, ps, ps.locus.dim_bounds)
            node = slicer

    def place_slicer(self, node, next_node, selector, selection_bounds):
        slicer = Slicer(selection_bounds, selector.mode, selector.locus, selector.new_dim)
        slicer.inputs = [node]
        if next_node:
            next_node.inputs = [slicer]
        return slicer
        

class Slicer(PaddingCalculator):

    def __init__(self, selection_bounds, mode, locus, new_dim):
        self.mode = mode
        self.locus = locus
        self.new_dim = new_dim
        self.selection_bounds = selection_bounds
        self.multi_select = isinstance(locus, IntervalSet)

    def _call_on_signal(self, signal, key_spec=None):

        from k_onda.signals import PointProcessSignal
      
        output_signal = super()._call_on_signal(signal, key_spec=key_spec)

        if 'time' in self.selection_bounds:
            start, duration = list(zip(*[
                self.start_and_duration(bounds) for bounds in self.selection_bounds
                ]))
            output_signal.duration = duration
            output_signal.start = start

        if isinstance(signal, PointProcessSignal) and self.new_dim:
                    
            output_signal.coord_map.update({
                f"{self.new_dim}_{self.locus.metadim}": 
                f"{self.new_dim}_{output_signal.coord_map[dim]}"
                for dim in set(self.selection_bounds) & set(output_signal.coord_map)
            })

        return output_signal
    
    def make_output_schema(self, data_schema, key_spec):
        if not self.new_dim:
            return data_schema

        def update_schema(arr_schema):
            metadim = self.locus.metadim
            dims = copy(arr_schema.dims)
            dims = dims.remove(metadim)
            dims.add(self.new_dim)
            dims.add(f'{self.new_dim}_{metadim}')
            selectable_dims = copy(arr_schema._selectable_dims).add(metadim)
            return Schema(dims, selectable_dims=selectable_dims)

        if isinstance(data_schema, Schema):
            return update_schema(data_schema)
        else:
            return DatasetSchema(
                {key: update_schema(val) for key, val in data_schema}
                )
    
    def start_and_duration(self, bounds):
        duration = bounds['time'].hi - bounds['time'].lo
        start = bounds['time'].lo
        return start, duration

    def _validate_input(self, signal):
        for dim in set(self.selection_bounds):
            if dim not in signal.data_schema.selectable_dims:
                raise ValueError(f"Signal data does not have dimension {dim}.")

    def _get_transform(self, input, key_spec):
        apply_kwargs = self._get_apply_kwargs(input)
   
        return Transform(partial(self._apply, **apply_kwargs))

    def _get_apply_kwargs(self, input):
     
        return {
            'is_point_process': bool(getattr(input, 'coord_map', None)),
            'coord_map': getattr(input, 'coord_map', None)
            }

    def _apply(self, data, is_point_process=False, coord_map=None):
        # TODO: do I want to add validation of the the window boundaries versus
        # the data boundaries or will the natural error be informative enough?s
        if is_point_process:
            return self.select_point_process(data, coord_map)
        else:
            return self.select_continuous(data)

    def select_continuous(self, data):

        selection_bounds = self.selection_bounds.to_array_of_dicts()

        selected = []

        for bounds in selection_bounds:
            mask = reduce(and_, [
                (data.coords[dim] >= bounds[dim].lo) & (data.coords[dim] < bounds[dim].hi)
                for dim in bounds
            ])
            selected.append(data.where(mask, drop=True))

        if self.new_dim:
            selected = self.attach_continuous_relative_coords(selected)
            selected = self.swap_coords(selected)

        return self.concat_or_extract(selected)

    def _new_dim_coord(self):
        return f'{self.new_dim}_{self.locus.metadim}'
    
    def attach_continuous_relative_coords(self, selected_data):
        result = []
        for arr in selected_data:
            relative_coord = arr.coords[self.locus.dim] - arr.coords[self.locus.dim][0]

            # every arr will have coord relative_foo
            arr = arr.assign_coords(
                {f'relative_{self.locus.metadim}':(self.locus.dim, relative_coord)}) 
            
            # for example, if dim is time new_dim is block, now every block will have coord block_time
            if self.new_dim: 
                arr = arr.assign_coords({self._new_dim_coord(): (self.locus.dim, relative_coord)})
            
            result.append(arr)
        
        return result
    
    def swap_coords(self, selected_data):
        # If you've created a new_dim 'block', the main time dim becomes 'block_time', the 
        # time relative to the block, and 'time', the absolute time relative to the session
        # becomes an auxiliary coordinate.
        return [(
            arr
            .swap_dims({self.locus.dim: self._new_dim_coord()})
            .assign_coords(
                {self.locus.dim: (self._new_dim_coord(), arr[self.locus.dim].values())}
                )
            ) for arr in selected_data]
    
    def concat_or_extract(self, data):
        if self.multi_select:
            selected_data = xr.concat(
            data, 
            dim=self.new_dim or 'intervals', 
            combine_attrs='no_conflicts'
            )
        else:
            selected_data = data[0]
        return selected_data
    

    def select_point_process(self, data, coord_map):

        map_dims = set(self.selection_bounds) & set(coord_map)
        other_dims = set(self.selection_bounds) - map_dims
        
        # I want to select time,  My dims are spikes.  coord_map tells 
        # me that I can find time in spike_times
        # if I selected epoch I'd have spikes, epochs, and epoch_time/time 
     

        selection_bounds = self.selection_bounds.to_array_of_dicts()

        map_selection_bounds = [
            {dim: bounds for dim, bounds in dim_bounds.items() if dim in map_dims} 
            for dim_bounds in selection_bounds
            ]

        selected = []

        for bounds in map_selection_bounds:
            map_dim_mask = reduce(and_, [
                (data[coord_map[dim]] >= bounds[dim].lo) & 
                (data[coord_map[dim]] < bounds[dim].hi)
                for dim in map_dims
            ])

            if len(other_dims):
                other_dim_mask = reduce(and_, [
                    (data.coords[dim] >= bounds[dim].lo) & 
                    (data.coords[dim] < bounds[dim].hi)
                    for dim in other_dims
                ])

            mask = map_dim_mask

            if len(other_dims):
                mask = mask & other_dim_mask

            selected.append(data.where(mask, drop=True))

        selected = self.concat_or_extract(selected)

        data = self.attach_point_process_relative_coords(selected, map_selection_bounds, coord_map)

        return data
    

    def attach_point_process_relative_coords(self, data, selection_bounds, coord_map):

        # say we've selected on time, and created dim epoch.
        # the arrays now need dim epoch_spikes, epochs
        # there needs to be a key spike_times, and a key epoch_spike_times
        # we need to be know the epoch boundaries from the locus in order to 


        for orig_dim, key in coord_map.items():
            arr = data[key]  # e.g dims epochs, spike_times
            new_arr = deepcopy(arr)
            for i in arr[self.new_dim]:
                new_arr[i] -= selection_bounds[i][orig_dim].lo
            data[f"{self.new_dim}_{orig_dim}"] = new_arr

        return data
    
        