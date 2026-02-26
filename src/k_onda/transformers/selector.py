from copy import deepcopy, copy
from functools import partial
from collections import defaultdict

from k_onda.time import Epoch
from .core import Transformer
from k_onda.central import ureg
from ..utils import scalar




class FrequencyBand:

    def __init__(self, f_lo, f_hi, units="Hz", mode='pushdown'):
        self.f_lo = ureg.Quantity(f_lo, units=units)
        self.f_hi = ureg.Quantity(f_hi, units=units)
        self.mode = mode


class SelectTransform:

    def __init__(self, fn, selection, selection_endpoints=None, 
                 padded_endpoints=None, padlen=None):
        self.fn = fn
        self.selection = selection
        self.padlen = padlen
        self.selection_endpoints = selection_endpoints
        self.padded_endpoints = padded_endpoints
            

    def __call__(self, data):
        return self.fn(data)
    

class SelectMixin:

    def band(self, freq_band):
        return self.select(selection=freq_band, mode=freq_band.mode)

    def window(self, epoch):
        return self.select(selection=epoch, mode=epoch.mode)
    
    def select(self, mode='pushdown', selection=None, units=None, **dim_endpoints):
        return Selector(mode, selection, units, **dim_endpoints)(self)


class Selector(Transformer):

    def __init__(self, mode='pushdown', selection=None, units=None, **dim_endpoints):
        self.mode = mode # pushdown | local
        self.selection = selection
        self.units = units
        self.dim_endpoints = dim_endpoints
    
    def _call_on_signal(self, signal, key=None):
        
        child_class = self.get_output_class(signal)
        transform = self._get_transform(signal, key=key)
        eps = transform.selection_endpoints
        if 'time' in eps:
            duration = eps['time'][1] - eps['time'][0]
        else:
            duration = None
        child_signal = child_class(
            parent=signal, 
            transform=transform, 
            transformer=self, 
            duration=duration)
        return child_signal
    
    def _get_transform(self, parent, key=None):

        plan = {
            'selection_endpoints': ...,
            'padded_endpoints': ...,
            'padlen_by_dim': ...,
            'ancestors_with_selectors': ...,
            'when_to_select': ...,
        }

        plan= self._process_endpoints(plan, parent.lineage)
          
        plan = self.accumulate_padlen(plan, parent.lineage)
        
        apply_kwargs = self._get_apply_kwargs(parent, plan)
        
        transform_kwargs = self._get_transform_kwargs(plan)

        return SelectTransform(
            partial(self._apply, **apply_kwargs), **transform_kwargs
            )
    
    def _get_apply_kwargs(self, parent, plan):
        return {
            'plan': plan,
            'lineage': parent.lineage
            }
    
    def _get_transform_kwargs(self, plan):
        return {
            'selection_endpoints': plan['selection_endpoints'],
            'selection': self.selection,
            'padlen': plan['padlen_by_dim'],
            'padded_endpoints': plan['padded_endpoints']
        }
    
    def _apply(self, _, plan, lineage):
        
        plan = self._get_ancestors_with_selectors(plan, lineage)

        self.validate_selection(plan, lineage)
        
        if self.mode == 'pushdown':
            plan = self._plan_pushdown_select_timing(plan)
            return self.pushdown_selection(plan, lineage)
    
        elif self.mode == 'local':
            return self.local_selection(plan, lineage)
        
        else:
            raise ValueError(f"Unknown selection placement {self.mode}")

        
    @staticmethod
    def _create_dim_slices(dim_endpoints):
        return {dim: slice(*eps) for dim, eps in dim_endpoints.items()}
    
    def _process_endpoints(self, plan, lineage):
        
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
        for ancestor in lineage:
            defaults = getattr(ancestor, 'dim_defaults', {})
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

        plan['selection_endpoints'] = selection_endpoints
        return plan
    
    def _get_ancestors_with_selectors(self, plan, lineage):
        plan['ancestors_with_selectors'] = [
            (i, ancestor) for i, ancestor in enumerate(lineage)
            if isinstance(getattr(ancestor, 'transform', None), SelectTransform)
            ]
        return plan
    
    def accumulate_padlen(self, plan, lineage):

        selection_endpoints = plan['selection_endpoints']
     
        def get_units(dim, index): 
            try: 
                units = selection_endpoints[dim][index].units
            except:
                units = 1
            finally:
                return units

        def reset_padlen_accumulator(dim):
            return [0.0 * get_units(dim, 0), 0.0 * get_units(dim, 1)] 

        our_padlen = {dim: reset_padlen_accumulator(dim) for dim in selection_endpoints}

        padded_endpoints = deepcopy(selection_endpoints)
        # assuming lineage is root-leaf
        for ancestor in lineage:
            
            transform = getattr(ancestor, 'transform', None)
            
            if transform is None:
                continue

            elif isinstance(transform, SelectTransform):
                their_selection_endpoints = transform.selection_endpoints
                for dim in their_selection_endpoints:
                    our_padlen[dim] = reset_padlen_accumulator(dim) # reset

            else:
                their_padlen = transform.padlen
                for dim in selection_endpoints:
                    if dim in their_padlen:
                        our_padlen[dim][0] += their_padlen[dim][0]
                        our_padlen[dim][1] += their_padlen[dim][1]
        
        padded_endpoints = {
            dim: [selection_endpoints[dim][0] - our_padlen[dim][0], 
                  selection_endpoints[dim][1] + our_padlen[dim][1]] 
                  for dim in selection_endpoints}
        
        plan['padlen_by_dim'] = our_padlen
        plan['padded_endpoints'] = padded_endpoints
        return plan
    
    def validate_selection(self, plan, lineage):

        def get_data_endpoints(ancestor):
            return {
                dim: 
                [ancestor.data.coords[dim][0].item(), ancestor.data.coords[dim][-1].item()] 
                for dim in ancestor.data.dims
                }

        def compare_endpoints(theirs, ours):
            for dim in ours:
                if dim in theirs:
                    if theirs[dim][0] > ours[dim][0] or theirs[dim][1] < ours[dim][1]:
                        raise ValueError(f"Available data on dim {dim} is {theirs[dim]}.  " \
                        "Requested selection is {ours[dim]}") 
                    
        ours = plan['selection_endpoints']
        theirs = get_data_endpoints(lineage[0])

        compare_endpoints(theirs, ours)

        if self.mode == 'pushdown':
            for _, ancestor in plan['ancestors_with_selectors']:
                theirs = ancestor.transform.selection_endpoints 
                compare_endpoints(theirs, ours)
    
    def _plan_pushdown_select_timing(self, plan):
        when_to_select = defaultdict(lambda: 0)
        ours = plan['selection_endpoints']

        for i, ancestor in plan['ancestors_with_selectors']:
            theirs = ancestor.transform.selection_endpoints

            if self.mode == 'pushdown':
                for dim in ours:
                    if dim in theirs:
                        when_to_select[dim] = i

        plan['when_to_select'] = when_to_select
        return plan

    def local_selection(self, plan, lineage):
        data = lineage[0].data

        for ancestor in lineage[1:]:
            transform = getattr(ancestor, 'transform', None)
            if transform is not None:
                data = transform(data)

        data = data.sel(**self._create_dim_slices(plan['selection_endpoints']))

        return data

    def pushdown_selection(self, plan, lineage):

        when_to_select = plan['when_to_select']
        selection_endpoints = plan['selection_endpoints']
        padded_endpoints = plan['padded_endpoints']

        index_dims_dict = defaultdict(list)

        for dim, index in when_to_select.items():
            index_dims_dict[index].append(dim)

        def select_dims_for_index(idx):
            selection = {dim: padded_endpoints[dim] 
                         for dim in index_dims_dict[idx]}
            return data.sel(**self._create_dim_slices(selection))

        for i, ancestor in enumerate(lineage):

            if i == 0:
                data = ancestor.data
                data = select_dims_for_index(0)
            
            else:
                transform = getattr(ancestor, 'transform', None)
                if transform is not None:
                    data = transform(data)

                if i in index_dims_dict:
                    # do select
                    data = select_dims_for_index(i)

        return data.sel(**self._create_dim_slices(selection_endpoints))



    
    
  