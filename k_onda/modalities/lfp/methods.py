import mne
import numpy as np
import xarray as xr

from k_onda.model.data import TransformRegistryMixin

from k_onda.math import (
    msc_from_spectra,
    fisher_z_from_msc,
    back_transform_fisher_z,
    welch_psd,
    welch_csd,
    multitaper_psd,
    multitaper_csd
)


class LFPMethods(TransformRegistryMixin):
    TRANSFORMS = {
        "coherence": (fisher_z_from_msc, back_transform_fisher_z),
        # 'amp_xcorr': (amp_xcorr_transform, amp_xcorr_back), ...
    }

    def get_weights(self):
        if not hasattr(self, "children") or len(self.children) == 0:
            return None
        if "segment" in self.children[0].name:
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

    @staticmethod
    def resample(data, fs, new_fs, axis=-1):
        data_rs = mne.filter.resample(
            data,
            down=fs / new_fs,
            npad="auto",
            axis=axis,
        )
        return data_rs, new_fs

    def get_power(self):
        return self.resolve_calc_fun("power", stop_at=self.calc_opts.get("base", "event"))

    def get_amp_xcorr(self):
        return self.get_average(
            "get_amp_xcorr", stop_at=self.calc_opts.get("base", "amp_xcorr_calculator")
        )

    def get_lag_of_max_corr(self):
        return self.get_average(
            "get_lag_of_max_corr", stop_at=self.calc_opts.get("base", "amp_xcorr_calculator")
        )
    
    def get_granger_causality(self):
        return self.get_average(
            "get_granger_causality", stop_at=self.calc_opts.get("base", "granger_causalityzs")
        )

    def resolve_calc_fun(self, calc_type, stop_at=None):
        if not hasattr(self, "children") or len(self.children) == 0 or stop_at in self.name:
            return getattr(self, f"get_{calc_type}_")()
        else:
            return self.get_average(
                f"get_{calc_type}", weights=self.get_weights(), stop_at=stop_at
            )

    def get_base_of_coherence_constituent(self, calc_type):
        base = self.calc_opts.get("base")
        if isinstance(base, dict):
            base = self.calc_opts["base"].get(calc_type)
        
        if base is None:
            base = "segment" if self.calc_opts.get("validate_events") else "calculator"
            
        return base

    def get_psd(self):
        base = self.get_base_of_coherence_constituent("psd")
        return self.resolve_calc_fun("psd", stop_at=base)

    def get_csd(self):
        base = self.get_base_of_coherence_constituent("csd")
        return self.resolve_calc_fun("csd", stop_at=base)

    def get_coherence(self):
        base = self.get_base_of_coherence_constituent("coherence")
        return self.resolve_calc_fun("coherence", stop_at=base)
    
    def get_psd_and_csd(self):
        stop_at = self.get_base_of_coherence_constituent("psd_and_csd")
        return self.resolve_calc_fun("psd_and_csd", stop_at=stop_at)


class SpectralDensityMethods:

    def spectral_density_calc(self, data, calc):
    
        method, func, kwargs, return_val_names = self.get_spectral_func_and_args(calc)

        fs = self.to_float(self.lfp_sampling_rate, unit="Hz")

        f, *return_vals = func(*data, fs, **kwargs)

        das = []
        for return_val in return_vals:
            da = self.wrap_calc_in_da(f, return_val, method, calc, units="V^2/Hz")
            da = self.trim_da(da, calc)
            das.append(da)

        if len(return_vals) > 1:
            return xr.Dataset({name: val for name, val in zip(return_val_names, das)})
        else:
            return das[0]

    def wrap_calc_in_da(self, f, val, method, calc, units=None):
        frequency, freq_bin = self.frequency_and_freq_bin(f)

        da = xr.DataArray(
            data=val,
            dims=("freq_bin",),
            coords={
                "freq_bin": freq_bin,    # unitless bin index
                "frequency": frequency,  # unitful Hz coord
            },
            attrs={"units": units, "method": method, "calc": calc},
        )
        return da
    
    def trim_da(self, da, calc):
        da = self.frequency_selector(da)
        if self.calc_type == calc and self.calc_opts.get("frequency_type") == "block":
            da = da.mean(dim="freq_bin", keep_attrs=True)
        return da
    
    def welch_args(self, analysis):
        nperseg = 2000
        noverlap = 1000

        args = {
            "nperseg": nperseg,
            "noverlap": noverlap,
            "window": "hann",
            "detrend": "constant",
        }
        args.update(self.calc_opts.get(f"{analysis}_params", {}).get("args", {}))
        args.update(self.calc_opts.get("coherence_params", {}).get("args", {}))
        return args

    def multitaper_args(self, analysis):
        """
        Args for multitaper calculations.

        You control smoothing via:
        - bandwidth (Hz): full bandwidth (2W) in MNE psd_array_multitaper
        - adaptive, low_bias, normalization

        You can override via:
          calc_opts[f'{analysis}_params']['args']
        and/or
          calc_opts['coherence_params']['args']
        """
        args = {
            "bandwidth": None,          # full bandwidth in Hz (2W). None => MNE heuristic
            "adaptive": False,
            "low_bias": True,
            "normalization": "full"
        }
        args.update(self.calc_opts.get(f"{analysis}_params", {}).get("args", {}))
        args.update(self.calc_opts.get("coherence_params", {}).get("args", {}))
        return args
    
    def get_spectral_func_and_args(self, calc):
        func_table = {
            ("psd", "welch"): welch_psd,
            ("psd", "multitaper"): multitaper_psd,
            ("csd", "welch"): welch_csd,
            ("csd", "multitaper"): multitaper_csd,
        }
        method = self.calc_opts.get("coherence_params", {}).get("method", "welch").lower()
        args = getattr(self, f"{method}_args")(calc)

        if calc == "psd_and_csd":
            return method, multitaper_csd, args, ["Sxy", "Sxx", "Syy"]

        try:
            func = func_table[(calc, method)]
        except KeyError:
            raise ValueError(f"Unsupported calc/method: {calc}, {method}")

        return method, func, args, []


class CoherenceMethods(SpectralDensityMethods):

    def get_psd_(self):
        regions_data = dict(zip(self.regions, self.regions_data))
        out = {
            region: self.spectral_density_calc([data], "psd")
            for region, data in regions_data.items()
        }
        return xr.Dataset(out)

    def get_csd_(self):
        return self.spectral_density_calc(self.regions_data, "csd")

    def get_psd_and_csd_(self):
        # This eventually calls a function that saves time by calculating the complexly valued 
        # PSD as the input to CSD so you don't calculate PSD twice.
        return self.spectral_density_calc(self.regions_data, "psd_and_csd")

    def get_coherence_(self):

        method = self.calc_opts.get("coherence_params", {}).get("method", "welch").lower()

        if method == 'welch':
            Sxx, Syy = [self.get_psd()[region] for region in self.regions]
            Sxy = self.get_csd()
            
        else:
            ds = self.get_psd_and_csd()
            Sxy = ds["Sxy"]
            Sxx = ds["Sxx"]
            Syy = ds["Syy"] 

        coherence = msc_from_spectra(Sxx=Sxx, Syy=Syy, Sxy=Sxy)

        f = Sxy.coords["frequency"].pint.to("Hz").data.magnitude
        da = self.wrap_calc_in_da(f, coherence, method, "coherence", units=1)

        da = self.trim_da(da, "coherence")

        da.attrs.update({"space": "raw", "transform_key": "coherence"})
    
        if da.attrs["space"] == "raw":
            da = fisher_z_from_msc(da)
            da.attrs["space"] = "z"

        return da
   