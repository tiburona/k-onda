import pint


class FrequencyBand:

    def __init__(self, f_lo, f_hi, units="Hz"):
        self.f_lo = pint.Quantity(f_lo, units=units)
        self.f_hi = pint.Quantity(f_hi, units=units)


class SelectMixin:

    def band(self, freq_band):
        return self.select(frequency=slice(freq_band.f_lo, freq_band.f_hi))

    def window(self, epoch):
        return self.select(time=slice(epoch.t0, epoch.t1))
    
    def select(self, **dim_slices):
        data = self._select_recursive(**dim_slices)
        signal_class = getattr(self, 'output_signal_class', type(self))
        return signal_class.from_data(data, **self._get_inheritable_attrs())
    
    def _select_recursive(self, **dim_slices):
        """Compute selected data using recursive optimization. Returns DataArray."""
        if getattr(self, 'parent', None) is None:
            return self.data.sel(**dim_slices)
        
        # Check which dims exist in parent
        parent_dims = self.parent.data.dims if hasattr(self.parent, 'data') else ()
        parent_slices = {k: v for k, v in dim_slices.items() if k in parent_dims}
        transform_slices = {k: v for k, v in dim_slices.items() if k not in parent_dims}

        # Recurse for dims that exist in parent
        if parent_slices:
            parent_data = self.parent._select_recursive(**parent_slices)
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
        if hasattr(self, 'sampling_rate'):
            attrs['sampling_rate'] = self.sampling_rate
        return attrs