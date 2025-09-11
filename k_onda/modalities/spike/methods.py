import numpy as np
import xarray as xr

from k_onda.math import normalized_crosscorr
from k_onda.utils import standardize, calc_hist, correlogram


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
            self._spikes = self.unit.find_spikes(*self.spike_range)
        if self.period_type == 'prestim':
            a = 'foo'
        return self._spikes
 
    @property
    def spikes_in_seconds_from_start(self):
        return self.unit.find_spikes(*self.spike_range) - self.start
    
    @property
    def spike_range(self):
        return (self.start, self.stop)
    
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
    
    def get_coords(self, length):
        index = np.arange(length)
        absolute_time = standardize(index * self.calc_opts['bin_size'] + self.start)
        relative_time = standardize(absolute_time - self._start)
        period_time = standardize(absolute_time - self.parent._start 
                                  if self.name == 'event' else relative_time)

        coord_dict = {
            'time': index,
            'absolute_time': ('time', absolute_time),
            'relative_time': ('time', relative_time),
            'period_time': ('time', period_time)
        }

        if self.name == 'event':
            coord_dict['event_time'] = ('time', relative_time)

        return coord_dict
        
    def get_spike_counts_(self):
        if 'counts' in self.private_cache:
            counts = self.private_cache['counts']
        else:
            raw_counts = calc_hist(self.spikes, self.num_bins_per, self.spike_range)[0]
            coords = self.get_coords(len(raw_counts))
            # Wrap the raw counts in a DataArray with 'time' as the dimension.
            counts = xr.DataArray(
                raw_counts,
                dims=['time'],  
                coords=coords
            )
        if self.calc_type == 'psth':
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
        return self.get_spike_counts_()/self.calc_opts.get('bin_size', .01)

    def get_proportion_(self):
        return xr.where(self.get_psth() > 0, 1, 0)
    
    def get_cross_correlations(self, pair=None):
        other = self.get_other(pair)
        raw_cross_corr, _ = normalized_crosscorr(self.get_unadjusted_rates(),
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
