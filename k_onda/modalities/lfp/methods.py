import mne
import numpy as np
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
    
    def frequency_and_freq_bin(self, f):
        n_freq = len(f)
        freq_bin = np.arange(n_freq)

        frequency = self.quantity(
            f,
            units="Hz",
            dims=("freq_bin",),
            name="frequency",
        )
        return frequency, freq_bin
    
    def frequency_selector(self, da):
        """
        Select a frequency band using pint-aware, unitful comparisons.

        Expects:
        - da['frequency'] as a pint-xarray coord in Hz
        - self.freq_range as a 2-element pint-xarray in Hz with edge=('low','high')
        - self.tolerance as a pint-xarray scalar in Hz
        """
        low = (self.freq_range.sel(edge="low") - self.tolerance).pint.to("Hz")
        high = (self.freq_range.sel(edge="high") + self.tolerance).pint.to("Hz")
        f = da["frequency"].pint.to("Hz")
        fmask = (f >= low) & (f <= high)
        return da.sel(freq_bin=fmask)
    
        """
        Select a frequency band using pint-aware, unitful comparisons.

        Expects:
        - da['frequency'] as a pint-xarray coord in Hz
        - self.freq_range as a 2-element pint-xarray in Hz with edge=('low','high')
        - self.tolerance as a pint-xarray scalar in Hz
        """
        low = (self.freq_range.sel(edge="low") - self.tolerance).pint.to("Hz")
        high = (self.freq_range.sel(edge="high") + self.tolerance).pint.to("Hz")
        f = da["frequency"].pint.to("Hz")
        fmask = (f >= low) & (f <= high)
        return da.sel(freq_bin=fmask)
    
    def resample(data, fs, new_fs, axis=-1):
        data_rs = mne.filter.resample(
            data,
            down=fs / new_fs,
            npad='auto',
            axis=axis
        )
        return data_rs, new_fs
 
    def get_power(self):
        return self.resolve_calc_fun('power', stop_at=self.calc_opts.get('base', 'event'))
        return self.resolve_calc_fun('power', stop_at=self.calc_opts.get('base', 'event'))
    
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
    
    def get_base_of_coherence_constituent(self, calc_type):
        base = None
        if self.calc_type == calc_type:
            base = self.calc_opts.get('base')
            if base is None:
                base = 'segment' if self.calc_opts.get('validate_events') else 'period'
        elif self.calc_type == 'coherence':
            if isinstance(self.calc_opts.get('base'), dict):
                base = self.calc_opts['base'].get(calc_type)
                base = self.calc_opts['base'].get(calc_type)
            if base is None:
                base = 'segment' if self.calc_opts.get('validate_events') else 'calculator'
        else:
            raise ValueError(f"Why are you trying to calculate {calc_type}?")
            raise ValueError(f"Why are you trying to calculate {calc_type}?")
        
        return base

    def get_psd(self):
        base = self.get_base_of_coherence_constituent('psd')
        return base

    def get_psd(self):
        base = self.get_base_of_coherence_constituent('psd')
        return self.resolve_calc_fun('psd', stop_at=base)

    def get_csd(self):
        base = self.get_base_of_coherence_constituent('csd')  
        base = self.get_base_of_coherence_constituent('csd')  
        return self.resolve_calc_fun('csd', stop_at=base)

    def get_coherence(self):
        stop_at = self.calc_opts.get('base', 'coherence_calculator')
        return self.resolve_calc_fun('coherence', stop_at=stop_at)

class SpectralDensityMethods:

    def spectral_density_calc(self, data, calc, func):

        args = self.welch_and_coherence_args(calc)

        fs = self.to_float(self.lfp_sampling_rate, unit="Hz")

        f, return_val = func(*data, fs, **args)

        frequency, freq_bin = self.frequency_and_freq_bin(f)

        da = xr.DataArray(
            data=return_val,
            dims=("freq_bin",),
            coords={
                "freq_bin": freq_bin,   # unitless bin index
                "frequency": frequency, # unitful Hz coord along freq_bin
            },
            attrs={"units": "V^2/Hz"},
        )

        da = self.frequency_selector(da)

        if self.calc_type == calc and self.calc_opts.get("frequency_type") == "block":
            da = da.mean(dim="frequency", keep_attrs=True)

        return da
    

class CoherenceMethods(SpectralDensityMethods):

    def get_psd_(self):

        regions_data = dict(zip(self.regions, self.regions_data))
        out = {region: self.spectral_density_calc([data], "psd", psd) 
               for region, data in regions_data.items()}
        ds = xr.Dataset(out)
        return ds
    
    def get_csd_(self):

        return self.spectral_density_calc(
            self.regions_data, "csd", cross_spectral_density)
    
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
            da = da.mean(dim='freq_bin', keep_attrs=True) 
          
        return da
           
    def welch_and_coherence_args(self, analysis):

        nperseg = 2000                
        noverlap = 1000

        args = {
            'nperseg': nperseg,
            'noverlap': noverlap,
            'window': 'hann',
            'detrend': 'constant',
        }
        args.update(self.calc_opts.get(f'{analysis}_params', {}).get('args', {}))
        args.update(self.calc_opts.get('coherence_params', {}).get('args', {}))
        return args
