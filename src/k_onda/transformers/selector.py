from copy import deepcopy
from functools import partial
from functools import reduce
import xarray as xr

from k_onda.time import Epoch
from .core import Transformer, Transform
from k_onda.central import ureg
from k_onda.graph import walk_tree, new_tree


class FrequencyBand:

    def __init__(self, f_lo, f_hi, units="Hz", mode='pushdown'):
        self.f_lo = ureg.Quantity(f_lo, units=units)
        self.f_hi = ureg.Quantity(f_hi, units=units)
        self.mode = mode
    


    
 


class Selector(Transformer):

    name = 'selector'

    def __init__(self, mode='pushdown', selection=None, units=None, **dim_endpoints):
        self.mode = mode # pushdown | local
        self.selection = selection
        self.units = units
        self.dim_endpoints = dim_endpoints
        self.old_tree = None
    
    def _call_on_signal(self, signal, key_spec=None):

        self._validate_input(signal, key_spec=key_spec)

        endpoints = self._process_endpoints(signal)

        if self.mode == 'pushdown':
            new_leaf = self._place_pushdown_windows(signal, endpoints)
        else:
            new_leaf = self._place_local_window(signal, endpoints)

        self.old_tree = signal
        new_leaf.optimizers.append(self)

        return new_leaf
    
    def _validate_input(self, signal, key_spec=None):

        if key_spec and key_spec.input_name is not None:
            raise NotImplementedError(
                "Use signal.payload(key).select(...) or signal[key].select"
                )
        
        for node, _, _ in walk_tree(signal):
            if isinstance(node, Window) and node.selector_mode == self.mode:
                common_dims = set(node.selection_endpoints) & set(self.dim_endpoints)
                if common_dims:
                    raise ValueError(
                        f"Two selectors with mode {self.mode} for dims {common_dims}"
                        )
                
    def _process_endpoints(self, input_signal):
        
        # Expand selection objects into endpoints
        if isinstance(self.selection, Epoch):
            dim_endpoints = {
                'time': [self.selection.t0, self.selection.t1]
                }
        elif isinstance(self.selection, FrequencyBand):
            dim_endpoints = {
                'frequency': (self.selection.f_lo, self.selection.f_hi)
                }
        else:
            dim_endpoints = self.dim_endpoints

        # Assign units to endpoints from self.units
        selection_endpoints = deepcopy(dim_endpoints)

        for dim, eps in selection_endpoints.items():
            if len(eps) == 3:
                ep_units = getattr(ureg, eps[2])
                eps = eps[:2]
            elif isinstance(self.units, str):
                if len(list(dim_endpoints.keys())) > 1:
                    raise ValueError("units type str is ambiguous when selecting" \
                    "on more than one dimension.")
                ep_units = getattr(ureg, self.units)
            elif isinstance(self.units, dict) and dim in self.units:
                ep_units = getattr(ureg, self.units[dim])
            else:
                ep_units = 1
            
            selection_endpoints[dim] = [ep * ep_units for ep in eps]

        # If that didn't yield units, try to get them from the signal
        # First collect every default dim leading up to this selection
        dim_defaults = {}
        for node, _, _ in walk_tree(input_signal):
            defaults = getattr(node, 'dim_defaults', {})
            dim_defaults.update(defaults)
        
        # Then assign them to every non-unit-aware endpoint
        for dim in dim_endpoints:
            eps = selection_endpoints[dim]
            is_unit_aware = all([isinstance(ep, ureg.Quantity) for ep in eps])
            if not is_unit_aware: 
                if dim_defaults:
                    selection_endpoints[dim] = [
                        ureg.Quantity(float(p), dim_defaults[dim]) 
                        for p in selection_endpoints[dim]
                        ]
            else: 
                selection_endpoints[dim] = list(selection_endpoints[dim])

        return selection_endpoints

    def _place_pushdown_windows(self, signal, endpoints):
        
        def get_units(dim, index): 
            try: 
                units = endpoints[dim][index].units
            except:
                units = 1
            finally:
                return units

        starting_val = {
            dim: [0.0 * get_units(dim, 0), 0.0 * get_units(dim, 1)] 
            for dim in endpoints
            }
        
        def accumulate_padlen(node, padlen):
           
            node_padlen = getattr(getattr(node, 'transform', None), 'padlen', {})
       
            dims = set(padlen) & set(node_padlen)
            for dim in dims:
                padlen[dim][0] += node_padlen[dim][0]
                padlen[dim][1] += node_padlen[dim][1]
            return padlen

        new_leaf = new_tree(signal)
            
        done_dims = set()

        def is_insertion_point(dim):
            if not hasattr(node, 'inputs'):
                return True
            for inp in node.inputs:
                for d in inp.data_dims:
                    if d == dim:
                        return False
            return True

        for node, padlen, last_node in walk_tree(
            new_leaf, 
            func=accumulate_padlen, 
            starting_val=starting_val
            ):

            ours = set(endpoints); theirs = set(node.data_schema.selectable_dims)
            dims = ours & theirs - done_dims
            insertion_dims = set([dim for dim in dims if is_insertion_point(dim)])
            if insertion_dims:
                done_dims.update(insertion_dims)
                eps = {dim: [p * 1.0 for p in points] 
                       for dim, points in endpoints.items() 
                       if dim in insertion_dims}
                for dim in eps:
                    eps[dim][0] -= padlen[dim][0]
                    eps[dim][1] += padlen[dim][1]
                new_node = Window(eps, self.mode)(node)
                last_node.inputs = (new_node,)

        new_window_leaf = Window(endpoints, self.mode)(new_leaf)

        return new_window_leaf
    
    def _place_local_window(self, signal, endpoints):

        new_leaf = new_tree(signal)
        new_window_leaf = Window(endpoints, self.mode)(new_leaf)
        return new_window_leaf


class Window(Transformer):

    name = 'window'
    
    def __init__(self, selection_endpoints, selector_mode):
        self.selection_endpoints = selection_endpoints
        self.selector_mode = selector_mode

    def _call_on_signal(self, signal, key_spec=None):
        # figure out what super does
        output_signal = super()._call_on_signal(signal, key_spec=key_spec)

        if 'time' in self.selection_endpoints:
            duration = self.selection_endpoints['time'][1] - self.selection_endpoints['time'][0]
            start = self.selection_endpoints['time'][0]
            output_signal.duration = duration
            output_signal.start = start

        return output_signal

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
            return data.sel(**self._create_dim_slices(self.selection_endpoints))
        
    @staticmethod
    def _create_dim_slices(dim_endpoints):
        return {dim: slice(*eps) for dim, eps in dim_endpoints.items()}
    
    def get_points(self, data, coord_map, map_dims):
        
        array_keys = list(coord_map.values())
        first_array_key = array_keys[0]
        first_coord_arr = data[first_array_key]  # example coord_array: spike_times
        point_dim = first_coord_arr.dims[0]  # example event_dim: spikes

        if len(map_dims) == 0:
            return point_dim, set(range(len(first_coord_arr[point_dim])))

        selected_points = []

        for dim in map_dims:
            array_key = coord_map[dim]  # dim is 'time' array_key is 'spike_times'
            coord_array = data[array_key]  # the array of spike_times
            entries = set()
            for i, entry in enumerate(coord_array):  # i is the index of the spike. entry is the value of the spike time
                if (self.selection_endpoints[dim][0] <= entry and 
                    entry < self.selection_endpoints[dim][1]):
                    entries.add(i)
            selected_points.append(entries)

        if len(selected_points) == 0:
            selected_points = set()
        else:
            selected_points = reduce(set.intersection, selected_points)

        return point_dim, selected_points 
    
    def select_point_process(self, data, coord_map):

        endpoints = self.selection_endpoints
        coord_map_dims = set(endpoints) & set(coord_map)
        other_dims = set(endpoints) - coord_map_dims

        points_args = (data, coord_map, coord_map_dims)
        point_dim, selected_points = self.get_points(*points_args)
       
        filtered = {}

        for key in data:
            da = data[key]
            da = da.isel({point_dim: sorted(selected_points)})

            dims = other_dims & set(da.dims)
            eps = {d: ps for d, ps in endpoints.items() if d in dims}
            da = da.sel(**self._create_dim_slices(eps))

            filtered[key] = da

        result = xr.Dataset(
            filtered,
            attrs=data.attrs
        )

        return result
        