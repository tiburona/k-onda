
import numpy as np
from collections import defaultdict
from bisect import bisect_left as bs_left, bisect_right as bs_right
from base_data import Data
from period_event import Period, Event
from period_constructor import PeriodConstructor
from spike_methods import SpikeMethods
from math_functions import calc_rates, calc_hist, cross_correlation, correlogram
from utils import cache_method

  
class Unit(Data, PeriodConstructor, SpikeMethods):

    _name = 'unit'
    
    def __init__(self, animal, category, spike_times, cluster_id, waveform, experiment=None, 
                 neuron_type=None, quality=None):
        super().__init__()
        self.animal = animal
        self.category = category
        self.spike_times = np.array(spike_times)
        self.cluster_id = cluster_id
        self.waveform = waveform
        self.experiment = experiment
        self.neuron_type = neuron_type
        self.quality = quality
        self.animal.units[category].append(self)
        self.identifier = str(self.animal.units[category].index(self) + 1)
        self.spike_periods = defaultdict(list)
        self.parent = animal
        
    @property
    def all_periods(self):
        return [period for key in self.spike_periods for period in self.spike_periods[key]]

    @property
    def firing_rate(self):
        return self.animal.sampling_rate * len(self.spike_times) / \
            float(self.spike_times[-1] - self.spike_times[0])

    @property
    def unit_pairs(self):
        all_unit_pairs = self.get_pairs()
        pairs_to_select = self.calc_opts.get('unit_pair')
        if pairs_to_select is None:
            return all_unit_pairs
        else:
            return [unit_pair for unit_pair in all_unit_pairs if ','.join(
                [unit_pair.unit.neuron_type, unit_pair.pair.neuron_type]) == pairs_to_select]
 
    def spike_prep(self):
        self.prepare_periods()
        self.children = self.get_all('spike_periods')

    def get_pairs(self):
        return [UnitPair(self, other) for other in [unit for unit in self.animal if unit.identifier != self.identifier]]

    def find_spikes(self, start, stop):
        return np.array(self.spike_times[bs_left(self.spike_times, start): bs_right(self.spike_times, stop)])

    def get_spikes_by_events(self):
        return [event.spikes for period in self.children for event in period.children]

    def get_spontaneous_firing(self):
        spontaneous_period = self.calc_opts.get('spontaneous', 120)
        if not isinstance(spontaneous_period, tuple):
            start = self.earliest_period.onset - spontaneous_period * self.sampling_rate - 1
            stop = self.earliest_period.onset - 1
        else:
            start = spontaneous_period[0] * self.sampling_rate
            stop = spontaneous_period[1] * self.sampling_rate
        num_bins = round((stop-start) / (self.sampling_rate * self.calc_opts['bin_size']))
        return calc_rates(self.find_spikes(start, stop), num_bins, (start, stop), self.calc_opts['bin_size'])

    def get_firing_std_dev(self, period_types=None):
        if period_types is None:  # default: take all period_types
            period_types = [period_type for period_type in self.spike_periods]
        return np.std([rate for period_type, periods in self.spike_periods.items() for period in periods
                       for rate in period.get_all_firing_rates() if period_type in period_types])

    def get_cross_correlations(self, axis=0):
        return np.mean([pair.get_cross_correlations(axis=axis, stop_at=self.calc_opts.get('base', 'period'))
                        for pair in self.unit_pairs], axis=axis)

    def get_correlogram(self, axis=0):
        return np.mean([pair.get_correlogram(axis=axis, stop_at=self.calc_opts.get('base', 'period'))
                        for pair in self.unit_pairs], axis=axis)


class UnitPair:
    pass


class SpikePeriod(Period, SpikeMethods):

    name = 'period'

    def __init__(self, unit, index, period_type, period_info, onset, events=None, 
                 target_period=None, is_relative=False, experiment=None):
        super().__init__(index, period_type, period_info, onset, experiment=experiment, 
                         target_period=target_period, is_relative=is_relative)
        self.unit = unit
        self.animal = self.unit.animal
        self.parent = unit
        self.cls = 'spike'

    def get_events(self):
        pre_stim, post_stim = (self.pre_stim, self.post_stim) * self.sampling_rate
        for i, start in enumerate(self.event_starts):
            spikes = self.unit.find_spikes(start - pre_stim, start + post_stim)
            self._events.append(
                SpikeEvent(self, self.unit, 
                           [((spike - start) / self.sampling_rate) for spike in spikes],
                           [(spike / self.sampling_rate) for spike in spikes], i))

    def get_all_firing_rates(self):
        return [event.get_firing_rates() for event in self.events]
    
    def get_all_spike_counts(self):
        return [event.get_spike_counts() for event in self.events]

    def mean_firing_rate(self):
        return np.mean(self.get_firing_rates())
    
    def mean_spike_counts(self):
        return np.mean(self.get_spike_counts())


class SpikeEvent(Event):
    def __init__(self, period, unit, spikes, spikes_original_times, index):
        super().__init__(period, index)
        self.unit = unit
        self.spikes = spikes
        self.spikes_original_times = spikes_original_times
        self.cls = 'spike'
       
    def get_psth(self):
        rates = self.get_firing_rates() 
        reference_rates = self.reference.get_firing_rates()
        rates -= reference_rates
        rates /= self.unit.get_firing_std_dev(period_types=self.period_type,)  # same as dividing unit psth by std dev 
        return rates

    @cache_method
    def get_firing_rates(self):
        bin_size = self.calc_opts['bin_size']
        spike_range = (-self.pre_stim, self.post_stim)
        rates = calc_rates(self.spikes, self.num_bins_per_event, spike_range, bin_size)
        return self.refer(rates)
    
    def get_spike_counts(self):
        return calc_hist(self.spikes, self.num_bins_per_event, (-self.pre_stim, self.post_stim))

    def get_cross_correlations(self, pair=None):
        other = pair.periods[self.period_type][self.period.identifier].events[self.identifier]
        cross_corr = cross_correlation(self.get_unadjusted_rates(), other.get_unadjusted_rates(), mode='full')
        boundary = round(self.calc_opts['max_lag'] / self.calc_opts['bin_size'])
        midpoint = cross_corr.size // 2
        return cross_corr[midpoint - boundary:midpoint + boundary + 1]

    def get_correlogram(self, pair=None, num_pairs=None):
        max_lag, bin_size = (self.calc_opts[opt] for opt in ['max_lag', 'bin_size'])
        lags = round(max_lag/bin_size)
        return correlogram(lags, bin_size, self.spikes, pair.spikes, num_pairs)

    def get_autocorrelogram(self):
        max_lag, bin_size = (self.calc_opts[opt] for opt in ['max_lag', 'bin_size'])
        lags = round(max_lag / bin_size)
        return correlogram(lags, bin_size, self.spikes, self.spikes, 1)

    

