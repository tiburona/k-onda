from copy import deepcopy, copy
from functools import partial
import pint

from k_onda.time import Epoch
from .core import Transformer
from k_onda.central import ureg




class FrequencyBand:

    def __init__(self, f_lo, f_hi, units="Hz"):
        self.f_lo = pint.Quantity(f_lo, units=units)
        self.f_hi = pint.Quantity(f_hi, units=units)


class SelectTransform:

    def __init__(self, fn, selection, padlen=None):
        self.fn = fn
        self.selection = selection
        self.padlen = padlen

    def __call__(self, data):
        return self.fn(data)
    

class SelectMixin:

    def band(self, freq_band):
        return self.select(frequency=(freq_band.f_lo, freq_band.f_hi))

    def window(self, epoch):
        return self.select(time=(epoch.t0, epoch.t1))
    
    def select(self, placement='pushdown', selection=None, units=None, **dim_endpoints):
        return Selector(placement, selection, units, **dim_endpoints)(self)


class Selector(Transformer):

    def __init__(self, placement='pushdown', selection=None, units=None, **dim_endpoints):
        self.placement = placement # pushdown | local
        self.selection = selection
        if isinstance(selection, Epoch):
            self.dim_endpoints = {'time': [selection.t0, selection.t1]}
        elif isinstance(selection, FrequencyBand):
            self.dim_endpoints = {
                'frequency': (selection.f_lo, selection.f_hi)}
        else:
            self.dim_endpoints = dim_endpoints
        self.units = units
        self._process_endpoints_and_units()

    def _process_endpoints_and_units(self):
        units = self.units; dim_endpoints = self.dim_endpoints
        for dim, endpoints in dim_endpoints.items():
            if len(endpoints) == 3:
                ep_units = getattr(ureg, endpoints[2])
            elif isinstance(self.units, str):
                if len(list(dim_endpoints.keys())) > 1:
                    raise ValueError("units type str is ambiguous when selecting" \
                    "on more than one dimension.")
                else:
                    ep_units = getattr(ureg, units)
            elif isinstance(units, dict) and dim in units:
                ep_units = getattr(ureg, units[dim])
            else:
                ep_units = 1
            
            for ep in endpoints:
                ep *= ep_units
       
    def _call_on_signal(self, signal, key=None):
        
        child_class = self.get_output_class(signal)
        transform = self._get_transform(signal, key=key)
        if 'time' in self.dim_endpoints:
            duration = self.dim_endpoints['time'][1] - self.dim_endpoints['time'][0]
        else:
            duration = None
        child_signal = child_class(
            parent=signal, 
            transform=transform, 
            transformer=self, 
            duration=duration)
        return child_signal
    
    def _get_transform(self, parent, key=None):
        apply_kwargs = self._get_apply_kwargs(parent)
        transform_kwargs = self._get_transform_kwargs(apply_kwargs)
        return SelectTransform(partial(self._apply, **apply_kwargs), **transform_kwargs)
    
    def _get_apply_kwargs(self, parent):
        return {
            'lineage': parent.lineage,
            'padlen': self._get_padlen(parent.lineage, self.dim_endpoints),
            'dim_defaults': getattr(parent, 'dim_defaults', None)
            }
    
    def _get_transform_kwargs(self, apply_kwargs):
        return {
            'selection': self.selection,
            'padlen': apply_kwargs['padlen']
        }

    def _get_padlen(self, lineage, dim_endpoints):
        
        def get_units(dim, index): 
            try: 
                units = dim_endpoints[dim][index].units
            except:
                units = 1
            finally:
                return units

        accumulator = {dim: [0.0 * get_units(dim, 0), 0.0 * get_units(dim, 1)] 
                       for dim in dim_endpoints}

        for signal in lineage:
            transform = getattr(signal, 'transform', None)
            if transform is not None and not isinstance(transform, SelectTransform):
                padlen = getattr(signal.transform, 'padlen', {})
                dims = set(padlen.keys()) & set(dim_endpoints.keys())
                for dim in dims:
                    accumulator[dim][0] += padlen[dim][0]
                    accumulator[dim][1] += padlen[dim][1]

        return accumulator
    
    def _apply(self, data, lineage, padlen, dim_defaults):
        if self.placement == 'pushdown':
            return self._apply_pushdown(data, lineage, padlen, dim_defaults)
        elif self.placement == 'local':
            return self._apply_local(data, dim_defaults)
        else:
            raise ValueError(f"Unknown placement {self.placement}")
        
    @staticmethod
    def _create_dim_slices(dim_endpoints):
        return {dim: slice(*eps) for dim, eps in dim_endpoints.items()}
        
    def _apply_default_units(self, dim_defaults):

        dim_endpoints_local = deepcopy(self.dim_endpoints)

        for dim, endpoints in self.dim_endpoints.items():
            is_unit_aware = all([isinstance(ep, pint.Quantity) for ep in endpoints])
            if not is_unit_aware: 
                if dim_defaults:
                    dim_endpoints_local[dim] = [
                        pint.Quantity(float(p), dim_defaults[dim]) 
                        for p in self.dim_endpoints[dim]
                        ]
            else: 
                dim_endpoints_local[dim] = list(dim_endpoints_local[dim])

        return dim_endpoints_local
    
    def _apply_pushdown(self, data, lineage, padlen, dim_defaults):
        endpoints = self._apply_default_units(dim_defaults)
        return self._select_in_stages(data, lineage, endpoints, padlen)
    
    def _apply_local(self, data, dim_defaults):
        dim_endpoints = self._apply_default_units(dim_defaults)
        return data.sel(**self._create_dim_slices(**dim_endpoints))
    
    def _select_in_stages(self, data, lineage, dim_endpoints, padlen):

        original_endpoints = deepcopy(dim_endpoints)

        for dim in dim_endpoints:
            if dim in padlen:
                dim_endpoints[dim][0] -= padlen[dim][0]
                dim_endpoints[dim][1] += padlen[dim][1]

        done_dims = set()

        def compare_endpoints(selection, existing):
            for dim in selection:
                if dim in existing:
                    if (existing[dim][0] > selection[dim][0] or
                    existing[dim][1] < selection[dim][1]):
                        raise ValueError(
                        f"selection {selection[dim]} outside the bounds of" 
                        f" available data {existing[dim]}.")
                    
        for ancestor in lineage:
            transform = getattr(ancestor, 'transform', None)
            if isinstance(transform, SelectTransform):
                dims = (set(ancestor.transformer.dim_endpoints.keys()) & 
                        set(dim_endpoints.keys())) - done_dims
                for dim in dims:
                    compare_endpoints(
                        dim_endpoints, ancestor.selector.dim_endpoints
                        )
                    done_dims.add(dim)

            if (done_dims == set(dim_endpoints.keys()) or 
                getattr(ancestor, 'parent', None) is None):
                replay_from_ancestor = ancestor
                break
        
        data = self._replay_transforms(replay_from_ancestor, lineage, dim_endpoints)
                # the signal that needs to sel here is the 

        trimmed_data = data.sel(**self._create_dim_slices(original_endpoints))

        return trimmed_data
    
    def _replay_transforms(self, replay_from_ancestor, lineage, dim_endpoints):
        dim_endpoints = deepcopy(dim_endpoints)
        # todo: this would be a little nicer if it reversed after it identified
        # the right index -- fewer objects to move around.
        history = list(reversed(lineage))
        start_index = history.index(replay_from_ancestor)
        recent_history = history[start_index:]
        data = recent_history[0].data
        for ancestor in recent_history[1:]:
            data = ancestor.transform(data)
            dims = list(dim_endpoints.keys())
            for dim in dims:
                if dim in data.dims and dim in dim_endpoints:
                    data = data.sel(**self._create_dim_slices(dim_endpoints))
                    dim_endpoints.pop(dim)
        return data
    
  