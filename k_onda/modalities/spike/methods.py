import numpy as np
import xarray as xr

from k_onda.math import normalized_xcorr, calc_hist
from k_onda.utils import correlogram


class SpikeMethods:

    def get_psth(self):
        return self.get_average('get_psth', stop_at=self.calc_opts.get('base', 'event'))
    
    def get_firing_rates(self):
        return self.get_average('get_firing_rates', stop_at=self.calc_opts.get('base', 'event'))
    
    def get_spike_counts(self):
        return self.get_average('get_spike_counts', stop_at=self.calc_opts.get('base', 'event'))
    
    def get_proportion(self):
        return self.get_average('get_proportion', stop_at=self.calc_opts.get('base', 'event'))
    
    def get_spike_train(self):
        return self.get_average('get_spike_train', stop_at=self.calc_opts.get('base', 'event'))

    
class RateMethods:

    # Methods shared by SpikePeriod and SpikeEvent

    @property
    def spikes(self):
        if self._spikes is None:
            spikes = self.unit.find_spikes(*self.spike_range)  # quantity, dim 'spike'

            spikes_abs = spikes.pint.to('second')
            spikes_rel = (spikes - self.start).pint.to('second')

            self._spikes = spikes.assign_coords(
                absolute_time=('spike', spikes_abs.data),
                relative_time=('spike', spikes_rel.data),
            )

        return self._spikes
    
    @property
    def spike_range(self):
        return (self.start, self.stop)
    
    def resolve_calc_fun(self, calc_type, stop_at=None):
        if stop_at is None:
            stop_at=self.calc_opts.get('base', 'event')
        if self.name == stop_at:
            return getattr(self, f"get_{calc_type}_")()
        else:
            return self.get_average(f"get_{calc_type}", stop_at=stop_at)
    
    def get_psth(self):
        return self.resolve_calc_fun('psth')
    
    def get_firing_rates(self):
        return self.resolve_calc_fun('firing_rates')
    
    def get_spike_counts(self):
        return self.resolve_calc_fun('spike_counts')
    
    def get_spike_train(self):
        return self.resolve_calc_fun('spike_train')
    
    def get_proportion(self):
        return self.resolve_calc_fun('proportion')
        
    def get_spike_counts_(self):
        if 'counts' in self.private_cache:
            counts = self.private_cache['counts']
        else:
            # 1) histogram over spikes in SECONDS (plain floats)
            spikes_sec = self.to_float(self.spikes, unit="second")
            spike_range = tuple(self.to_float(b, unit="second") for b in self.spike_range)
            raw_counts = calc_hist(spikes_sec, self.num_bins_per, spike_range)[0]

            # 2) bin index
            index = xr.DataArray(
                np.arange(len(raw_counts)),
                dims=["time_bin"],
                name="time",
            )

            bin_size = self.bin_size.pint.to("second")
            start = self.start.pint.to("second")
            absolute_time = self.standardize_time(index * bin_size + start)  

            rel_start = self._start.pint.to("second")
            relative_time = self.standardize_time(absolute_time - rel_start)

            if self.name == "event":
                period_start = self.parent._start.pint.to("second")
                period_time = self.standardize_time(absolute_time - period_start)
            else:
                period_time = relative_time

            coord_dict = {
                "time_bin": index,
                "absolute_time": absolute_time,
                "relative_time": relative_time,
                "period_time": period_time,
            }

            if self.name == "event":
                coord_dict["event_time"] = relative_time

            counts = xr.DataArray(
                raw_counts,
                dims=["time_bin"],
                coords=coord_dict,
                name="spike_counts",
            )

            if self.calc_type == "psth":
                self.private_cache['counts'] = counts

        return counts

    def get_spike_train_(self):
        return xr.where(self.get_spike_counts_() != 0, 1, 0)
        
    def get_psth_(self):
        firing_rates = self.get_firing_rates() 
        reference_rates = self.reference.get_firing_rates()
        corrected_rates = (firing_rates - reference_rates)/self.unit.get_firing_std_dev() 
        corrected_rates = corrected_rates.assign_coords(**firing_rates.coords)
        self.private_cache = {}
        return corrected_rates

    def get_firing_rates_(self):
        return self.get_spike_counts_()/self.bin_size

    def get_proportion_(self):
        return xr.where(self.get_psth() > 0, 1, 0)
    
    def get_cross_correlations(self, pair=None):
        other = self.get_other(pair)
        raw_cross_corr, _ = normalized_xcorr(self.get_unadjusted_rates(),
                                            other.get_firing_rates_(),
                                            mode='full')
        n = raw_cross_corr.shape[0]
        midpoint = n // 2
        # Create lag coordinates so that 0 is at the midpoint.
        lag_coords = np.arange(-midpoint, n - midpoint)
        cross_corr = xr.DataArray(raw_cross_corr, dims=['lags'], coords={'lags': lag_coords})
        
        boundary = round(self.calc_opts['max_lag'] / self.calc_opts['bin_size'])
        # Now select lags from -boundary to boundary using .sel (coordinate-based slicing)
        return cross_corr.sel(lags=slice(-boundary, boundary))

    def get_correlogram(self, pair=None, num_pairs=None):
        max_lag, bin_size = (self.calc_opts[opt] for opt in ['max_lag', 'bin_size'])
        lags = round(max_lag/bin_size)
        return correlogram(lags, bin_size, self.spikes, pair.spikes, num_pairs)

    def get_autocorrelogram(self):
        max_lag, bin_size = (self.calc_opts[opt] for opt in ['max_lag', 'bin_size'])
        lags = round(max_lag / bin_size)
        return correlogram(lags, bin_size, self.spikes, self.spikes, 1)
