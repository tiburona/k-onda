from collections import defaultdict
from copy import deepcopy
import pint


class FrequencyBand:

    def __init__(self, f_lo, f_hi, units="Hz"):
        self.f_lo = pint.Quantity(f_lo, units=units)
        self.f_hi = pint.Quantity(f_hi, units=units)


class SelectionContext:
    
    def __init__(self):
        self.padlen = {}

    def add_padding(self, calculator_padlen):
        if not calculator_padlen:
            return
        
        for dim in calculator_padlen:
            existing_padlen = self.padlen.get(dim)
            if existing_padlen is None:
                self.padlen[dim] = calculator_padlen[dim]
            else:
                self.padlen[dim] = (
                    self.padlen[dim][0] + calculator_padlen[dim][0],
                    self.padlen[dim][1] + calculator_padlen[dim][1]
                    )


class SelectMixin:

    def band(self, freq_band):
        return self.select(frequency=(freq_band.f_lo, freq_band.f_hi))

    def window(self, epoch):
        return self.select(time=(epoch.t0, epoch.t1))

    def select(self, **dim_endpoints):
        """Select a subset of data along specified dimensions.

        Returns a NEW signal containing only the selected data. This breaks
        the parent chain since we materialize the computed result.
        """

        sc = SelectionContext()

        data = self._select_recursive(sc, **dim_endpoints)

        dim_endpoints = self._apply_default_units(dim_endpoints)
        dim_slices = self._create_slices(dim_endpoints)
        trimmed_data = data.sel(**dim_slices)
        
        signal_class = getattr(self, 'output_signal_class', type(self))
        new_signal = signal_class.from_data(trimmed_data, **self._get_inheritable_attrs())
        return new_signal
    
    def _apply_default_units(self, dim_endpoints):

        dim_endpoints_local = deepcopy(dim_endpoints)

        for dim, endpoints in dim_endpoints.items():
            is_unit_aware = all([isinstance(ep, pint.Quantity) for ep in endpoints])
            if not is_unit_aware: 
                dim_default = getattr(self, 'dim_defaults', {}).get(dim)
                if dim_default:
                    dim_endpoints_local[dim] = [
                        pint.Quantity(p, units=dim_default) 
                        for p in dim_endpoints[dim]
                        ]
            else: 
                dim_endpoints_local[dim] = list(dim_endpoints_local[dim])

        return dim_endpoints_local
    
    def _create_slices(self, dim_endpoints):
        return {dim: slice(*endpoints) for dim, endpoints in dim_endpoints.items()}
    
    def _select_recursive(self, selection_context, **dim_endpoints):
        """Compute selected data using recursive optimization. Returns DataArray."""
        
        dim_endpoints_local = self._apply_default_units(dim_endpoints)

        if hasattr(self, 'transform'):
            padlen = getattr(self.transform, 'padlen', {})
            selection_context.add_padding(padlen)

        if getattr(self, "parent", None) is None:
            for dim in selection_context.padlen:
                if dim in dim_endpoints_local:
                    dim_endpoints_local[dim][0] -= selection_context.padlen[dim][0]
                    dim_endpoints_local[dim][1] += selection_context.padlen[dim][1]

            dim_slices = self._create_slices(dim_endpoints_local)

            return self.data.sel(**dim_slices)
        
        # Check which dims exist in parent
        parent_dims = self.parent.data.dims if hasattr(self.parent, 'data') else ()
        parent_slices = {k: v for k, v in dim_endpoints_local.items() if k in parent_dims}
        transform_slices = {k: v for k, v in dim_endpoints_local.items() if k not in parent_dims}

        # Recurse for dims that exist in parent
        if parent_slices:
            parent_data = self.parent._select_recursive(
                selection_context, **parent_slices)
        else:
            parent_data = self.parent.data
        
        # Apply this signal's transform
        if self.transform is None:
            data = parent_data
        else:
            data = self.transform(parent_data)

        # Select on dims created by transform (e.g. frequency from spectrogram)
        if transform_slices:
            data = data.sel(**transform_slices)

        return data
    
    def _get_inheritable_attrs(self):
        """Get attributes that should be inherited by selected signals."""
        attrs = {}
        for attr in ['sampling_rate', 'origin']:
            if hasattr(self, attr):
                attrs[attr] = getattr(self, attr)
        return attrs