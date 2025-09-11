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
        cfg.update(
            {"fs": self.lfp_sampling_rate,
             "low": self.freq_range[0],
             "high": self.freq_range[1]
        }
        )
        return cfg
        
    @property
    def filter(self):
        flt, _ = self.make_filter(default_cfg=self.band_pass_filter_cfg, section=self.calc_type)
        return flt
    
    