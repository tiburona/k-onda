import xarray as xr

from k_onda.model.data import TransformRegistryMixin

from k_onda.math import (msc_from_spectra, fisher_z_from_coherence, fisher_z_from_msc, 
                         back_transform_fisher_z, psd, cross_spectral_density)


class LFPMethods(TransformRegistryMixin):

    TRANSFORMS = {
        'coherence': (fisher_z_from_msc, back_transform_fisher_z),
        # 'amp_xcorr': (amp_xcorr_transform, amp_xcorr_back), ...
    }


    def get_weights(self):
        if not hasattr(self, 'children') or len(self.children) == 0:
            return None
        if 'segment' in self.children[0].name:
            weights = self.segment_weights
        else:
            weights = [1 for _ in range(len(self.children))]
        return weights
        
    def frequency_selector(self, da):
                                      
        tol = 0.3                          

        # boolean mask along the *frequency* coord
        fmask = ((da['frequency'] >= self.freq_range[0] - tol) &
                (da['frequency'] <= self.freq_range[1] + tol))         

        return da.sel(frequency=fmask)
       
 
    def get_power(self):
        return self.get_average('get_power', stop_at=self.calc_opts.get('base', 'event'))
    
    def get_amp_xcorr(self):
        return self.get_average('get_amp_xcorr', 
                                stop_at=self.calc_opts.get('base', 'amp_xcorr_calculator'))
    
    def get_lag_of_max_corr(self):
        return self.get_average('get_lag_of_max_corr', 
                                stop_at=self.calc_opts.get('base', 'amp_xcorr_calculator'))

    def resolve_calc_fun(self, calc_type, stop_at=None):

        if not hasattr(self, 'children') or len(self.children) == 0 or stop_at in self.name:
            return getattr(self, f"get_{calc_type}_")()
        else:
            return self.get_average(f'get_{calc_type}', weights=self.get_weights(), stop_at=stop_at)
      
    def get_psd(self):
        base = None
        if self.calc_type == 'psd':
            base = self.calc_opts.get('base')
            if base is None:
                base = 'segment' if self.calc_opts.get('validate_events') else 'period'
        elif self.calc_type == 'coherence':
            if isinstance(self.calc_opts.get('base'), dict):
                base = self.calc_opts['base'].get('psd')
            if base is None:
                base = 'segment' if self.calc_opts.get('validate_events') else 'calculator'
        else:
            raise ValueError("Why are you trying to calculate PSD?")
        
        return self.resolve_calc_fun('psd', stop_at=base)

    def get_csd(self):
        base = None
        if self.calc_type == 'csd':
            base = self.calc_opts.get('base')
            if base is None:
                base = 'segment' if self.calc_opts.get('validate_events') else 'calculator'
        elif self.calc_type == 'coherence':
            if isinstance(self.calc_opts.get('base'), dict):
                base = self.calc_opts['base'].get('csd')
            if base is None:
                base = 'segment' if self.calc_opts.get('validate_events') else 'calculator'
        else:
            raise ValueError("Why are you trying to calculate CSD?")
        
        return self.resolve_calc_fun('csd', stop_at=base)

    def get_coherence(self):
        stop_at = self.calc_opts.get('base', 'coherence_calculator')

        coherence = self.resolve_calc_fun('coherence', stop_at=stop_at)
        
        return coherence


class PSDMethods:

    def psd_from_contiguous_data(self, data):

        args = self.welch_and_coherence_args('psd')

        f, return_val = psd(data, self.lfp_sampling_rate, **args)

        da = xr.DataArray(
            data=return_val, 
            dims=["frequency"],
            coords=dict(frequency=f))
        
        da = self.frequency_selector(da)

        if self.calc_type == 'psd' and self.calc_opts.get('frequency_type') == 'block':
            da = da.mean(dim='frequency', keep_attrs=True) 
        
        return da
    

class CoherenceMethods:

    def get_psd_(self):

        regions_data = dict(zip(self.regions, self.regions_data))
        out = {region: self.psd_from_contiguous_data(data) 
               for region, data in regions_data.items()}
        ds = xr.Dataset(out)
        return ds
    
    def get_csd_(self):

        args = self.welch_and_coherence_args('csd')

        f, val = cross_spectral_density(*self.regions_data, self.lfp_sampling_rate, **args)
        da = xr.DataArray(val, dims=['frequency'], coords={'frequency': f})
        da = self.frequency_selector(da)

        if self.calc_type == 'csd' and self.calc_opts.get('frequency_type') == 'block':
            da = da.mean(dim='frequency', keep_attrs=True)

        return da
    
    def get_coherence_(self):
        Sxx, Syy = list(self.get_psd().data_vars.values())
        Sxy = self.get_csd()
        return_val = msc_from_spectra(Sxx=Sxx, Syy=Syy, Sxy=Sxy)
        transform_attrs = {'space': 'raw', 'transform_key': 'coherence'}

        da = xr.DataArray(
                return_val,
                attrs=transform_attrs
            )
        
        # TODO: eventually it could be nice to read an optional override of 
        # transforms from calc opts; that could happen right here.
        if da.attrs['space'] == 'raw':
            da = fisher_z_from_msc(da)
            da.attrs['space'] = 'z'

        if self.calc_opts.get('frequency_type', 'continuous') == 'block':
            da = da.mean(dim='frequency', keep_attrs=True) 
          
        return da
           
    def welch_and_coherence_args(self, analysis):

        nperseg = 1000                
        noverlap = 500

        args = {
            'nperseg': nperseg,
            'noverlap': noverlap,
            'window': 'hann',
            'detrend': 'constant',
        }
        args.update(self.calc_opts.get(f'{analysis}_params', {}).get('args', {}))
        args.update(self.calc_opts.get('coherence_params', {}).get('args', {}))
        return args

 

    
   
    
    
   


    
    
