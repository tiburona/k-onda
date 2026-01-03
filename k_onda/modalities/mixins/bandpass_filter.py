from copy import deepcopy

from k_onda.math.filtering import FilterMixin

DEFAULTS = {
    "method": "iir_butter",
    "iir_order": 8
}

class BandPassFilterMixin(FilterMixin):

    @property
    def band_pass_filter_cfg(self):
        cfg = deepcopy(DEFAULTS)
        fs = self.to_float(self.lfp_sampling_rate, unit="Hz")
        low = self.to_float(self.freq_range[0], unit="Hz")
        high = self.to_float(self.freq_range[1], unit="Hz")
        cfg.update(
            {
                "fs": fs,
                "low": low,
                "high": high,
        }
        )
        return cfg
        
    @property
    def filter(self):
        flt, _ = self.make_filter(default_cfg=self.band_pass_filter_cfg, section=self.calc_type)
        return flt
    
    
