
from bisect import bisect_left as bs_left, bisect_right as bs_right
from collections import defaultdict

import numpy as np
import xarray as xr

from k_onda.utils import calc_hist, cross_correlation, correlogram, standardize
from k_onda.interfaces import PhyInterface
from k_onda.data import Data
from k_onda.data.period_constructor import PeriodConstructor
from k_onda.data.bins import BinMethods
from k_onda.data.period_event import Period, Event



class SpikeMethods:

    def get_psth(self, exclude=True):
        return self.get_average('get_psth', stop_at=self.calc_opts.get('base', 'event'), exclude=exclude)
    
    def get_firing_rates(self, exclude=True):
        return self.get_average('get_firing_rates', stop_at=self.calc_opts.get('base', 'event'), 
                                exclude=exclude)
    
    def get_spike_counts(self):
        return self.get_average('get_spike_counts', stop_at=self.calc_opts.get('base', 'event'))
    
    def get_proportion(self):
        return self.get_average('get_proportion', stop_at=self.calc_opts.get('base', 'event'))
    
    def get_spike_train(self, exclude=True):
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
    
    def get_firing_rates(self, exclude=True):
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
        raw_cross_corr = cross_correlation(self.get_unadjusted_rates(),
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


class Unit(Data, PeriodConstructor, SpikeMethods):

    _name = 'unit'
    
    def __init__(self, animal, category, spike_times, cluster_id, waveform=None, 
                 experiment=None, neuron_type=None, quality=None, firing_rate=None, 
                 fwhm_seconds=None, **kwargs):
        super().__init__(**kwargs)
        self.animal = animal
        self.category = category
        self.spike_times = np.array(spike_times)
        self.cluster_id = cluster_id
        self.waveform = waveform
        self.experiment = experiment
        self.neuron_type = neuron_type
        self.quality = quality
        self.firing_rate = firing_rate
        self.fwhm_seconds = fwhm_seconds
        self.fwhm_microseconds = fwhm_seconds * 1e6 if fwhm_seconds else None
        self.animal.units[category].append(self)
        self.identifier = '_'.join([self.animal.identifier, self.category, 
                                    str(self.animal.units[category].index(self) + 1)])
        self.spike_periods = defaultdict(list)
        self.mrl_calculators = defaultdict(list)
        self.parent = animal
        self.kind_of_data_to_period_type = {
            'spike': SpikePeriod
        }

    @property
    def children(self):
        if self.kind_of_data == 'mrl':
            return self.select_children('mrl_calculators')   
        else:
            return self.select_children('spike_periods')
        
    @property
    def all_periods(self):
        return [period for key in self.spike_periods for period in self.spike_periods[key]]

    @property
    def unit_pairs(self):
        all_unit_pairs = self.get_pairs()
        pairs_to_select = self.calc_opts.get('unit_pair')
        if pairs_to_select is None:
            return all_unit_pairs
        else:
            return [unit_pair for unit_pair in all_unit_pairs if ','.join(
                [unit_pair.unit.neuron_type, unit_pair.pair.neuron_type]) == pairs_to_select]
        
    def get_coords(self, length):
        index = np.arange(length)
        absolute_time = standardize(index * self.calc_opts['bin_size'] + self.start)
        relative_time = standardize(absolute_time - self._start)
    

        coord_dict = {
            'time': index,
            'absolute_time': ('time', absolute_time),
        }

        if self.name == 'event':
            coord_dict['event_time'] = ('time', relative_time)

        return coord_dict
 
    def spike_prep(self):
        self.prepare_periods()

    def get_pairs(self):
        return [UnitPair(self, other) for other in [unit for unit in self.animal if unit.identifier != self.identifier]]

    def find_spikes(self, start, stop):
        return np.array(self.spike_times[bs_left(self.spike_times, start): bs_right(self.spike_times, stop)])

    def get_spikes_by_events(self):
        return [event.spikes for period in self.children for event in period.children]

    def get_firing_std_dev(self):
        concatenated = self.calc_opts.get('base', 'event')
        return np.std([self.concatenate(concatenator='unit', concatenated=concatenated, attrs=['get_firing_rates'])])

    def get_cross_correlations(self, axis=0):
        cross_corrs = [pair.get_cross_correlations(axis=axis, stop_at=self.calc_opts.get('base', 'period'))
                    for pair in self.unit_pairs]
        combined = xr.concat(cross_corrs, dim='_neutral_dim')
        return combined.mean(dim='_neutral_dim', skipna=True)

    def get_correlogram(self, axis=0):
        correlograms = [pair.get_correlogram(axis=axis, stop_at=self.calc_opts.get('base', 'period'))
                        for pair in self.unit_pairs]
        combined = xr.concat(correlograms, dim='_neutral_dim')
        return combined.mean(dim='_neutral_dim', skipna=True)

    def get_cross_correlations(self, axis=0):
        return np.mean([pair.get_cross_correlations(axis=axis, stop_at=self.calc_opts.get('base', 'period'))
                        for pair in self.unit_pairs], axis=axis)
    
    def get_waveform(self, exclude=True):
        if self.waveform is not None:
            return self.waveform
        else:
            phy = PhyInterface(self.calc_opts['data_path'], self.parent.identifier)
            electrodes = phy.cluster_dict[self.cluster_id]['electrodes']
            wf = phy.get_mean_waveforms(self.cluster_id, electrodes)
            self.waveform = wf
            return wf
    
    def get_raster(self, exclude=True):
        base = self.calc_opts.get('base', 'event')
        # raster type can be spike_train for binarized data or spike_counts for gradations
        raster_type = self.calc_opts.get('raster_type', 'spike_train')
        depth = 1 if base == 'period' else 2
        raster = self.get_stack(method=f"get_{raster_type}", depth=depth)        
        return raster
    
    def get_mrl(self):
        return self.get_average('get_mrl', stop_at='mrl_calculator')


class SpikePeriod(Period, RateMethods):

    name = 'period'

    def __init__(self, unit, index, period_type, period_info, onset, 
                 events=None, target_period=None, is_relative=False, 
                 experiment=None):
        super().__init__(index, period_type, period_info, onset, events=events, 
                         experiment=experiment, target_period=target_period, 
                         is_relative=is_relative)
        self.unit = unit
        self.animal = self.unit.animal
        self.parent = unit
        self.private_cache = {}
        self._spikes = None 
        self.neuron_type = self.unit.neuron_type
  
        
    def get_events(self):
        self._events = [SpikeEvent(self, self.unit, start, i) 
                        for i, start in enumerate(self.event_starts)]
        
    def get_other(self, pair):
        return pair.spike_periods[self.period_type][self.identifier]
    
    def index_transformation_function(self, concatenator):
        if concatenator == 'unit':
            return lambda calc: calc.assign_coords(
                time=calc.coords['time'] + self.onset
                ) if isinstance(calc, xr.DataArray) else calc
        else:
            raise NotImplementedError("Period concatenation is only supported by unit.")

        
class SpikeEvent(Event, RateMethods, BinMethods):

    def __init__(self, period, unit, onset,  index):
        super().__init__(period, onset, index)
        self.unit = unit
        self.private_cache = {}
        self._spikes = None
        self.neuron_type = self.unit.neuron_type

    @property
    def start(self):
       return self._start - self.pre_event

    @property
    def stop(self):
        return self._start + self.post_event

    @property
    def spike_range(self):
        return (self.start, self.stop)
    
    def get_other(self, pair):
        return pair.spike_periods[self.period_type][self.period.identifier].events[self.identifier]
    
    def index_transformation_function(self, concatenator):
        if concatenator == 'period':
            return lambda calc: calc.assign_coords(
                time=calc.coords['time'] + self.identifier * self.duration
                ) if isinstance(calc, xr.DataArray) else calc
        elif concatenator == 'unit':
            return lambda calc: calc.assign_coords(
                time=calc.coords['time'] + self.identifier * self.duration + self.period.onset
                ) if isinstance(calc, xr.DataArray) else calc
        else:
            raise NotImplementedError("Event concatenation is only supported by period and unit.")



class UnitPair(Data):
    """A pair of two units for the purpose of calculating cross-correlations or correlograms."""

    name = 'unit_pair'

    def __init__(self, unit, pair):
        self.parent = unit.parent
        self.unit = unit
        self.pair = pair
        self.identifier = str((unit.identifier, pair.identifier))
        self.pair_category = ','.join([unit.neuron_type, pair.neuron_type])
        self.children = self.unit.children

    def get_cross_correlations(self, **kwargs):
        for kwarg, default in zip(['axis', 'stop_at'], [0, self.data_opts.get('base', 'period')]):
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs else default
        return self.get_average('get_cross_correlations', pair=self.pair, **kwargs)

    def get_correlogram(self, **kwargs):
        for kwarg, default in zip(['axis', 'stop_at'], [0, self.data_opts.get('base', 'period')]):
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs else default
        return self.get_average('get_correlogram', pair=self.pair, num_pairs=len(self.unit.unit_pairs), **kwargs)





            


            
                