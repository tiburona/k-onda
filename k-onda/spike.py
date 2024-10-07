
import numpy as np
from collections import defaultdict
from bisect import bisect_left as bs_left, bisect_right as bs_right
from base_data import Data
from period_event import Period, Event
from period_constructor import PeriodConstructorMethods
from spike_methods import SpikeMethods
from math_functions import calc_rates, calc_hist, cross_correlation, correlogram
from bins import BinMethods
from phy_interface import PhyInterface


class SpikeMethods:

    def get_psth(self):
        return self.get_average('get_psth', stop_at=self.calc_opts.get('base', 'event'))
    
    def get_firing_rates(self):
        return self.get_average('get_firing_rates', stop_at=self.calc_opts.get('base', 'event'))
    
class RateMethods:

    # Methods shared by SpikePeriod and SpikeEvent
    
    @property
    def spikes(self):
        return [spike for spike in self.unit.find_spikes(*self.spike_range)]
    
    @property
    def spikes_in_seconds_from_start(self):
        return [(spike - self.start)/self.sampling_rate 
                for spike in self.unit.find_spikes(*self.spike_range)]
    
    @property
    def spike_range(self):
        return (self.start, self.stop)
    
    def get_psth(self):
        stop_at=self.calc_opts.get('base', 'event')
        if self.name == stop_at:
            return self._get_psth()
        else:
            return self.get_average('get_psth', stop_at=stop_at)
    
    def get_firing_rates(self):
        stop_at=self.calc_opts.get('base', 'event')
        if self.name == stop_at:
            return self._get_firing_rates()
        else:
            return self.get_average('get_firing_rates', stop_at=stop_at)
    

    def _get_psth(self):
        rates = self.get_firing_rates() 
        reference_rates = self.reference.get_firing_rates()
        rates -= reference_rates
        rates /= self.unit.get_firing_std_dev(period_types=self.period_type,)  # same as dividing unit psth by std dev 
        self.private_cache = {}
        return rates

    def _get_firing_rates(self):
        bin_size = self.calc_opts.get('bin_size', .01)
        if 'rates' in self.private_cache:
            rates = self.private_cache['rates']
        else:
            rates = calc_rates(self.spikes, self.num_bins_per, self.spike_range, bin_size)
        if self.calc_type == 'psth':
            self.private_cache['rates'] = rates
        return self.refer(rates)


class Unit(Data, PeriodConstructorMethods, SpikeMethods):

    _name = 'unit'
    
    def __init__(self, animal, category, spike_times, cluster_id, waveform=None, experiment=None, 
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
        self.kind_of_data_to_period_type = {
            'spike': SpikePeriod
        }

    @property
    def children(self):
        return self.select_children('spike_periods')
        
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

    def get_pairs(self):
        return [UnitPair(self, other) for other in [unit for unit in self.animal if unit.identifier != self.identifier]]

    def find_spikes(self, start, stop):
        return np.array(self.spike_times[bs_left(self.spike_times, start): bs_right(self.spike_times, stop)])

    def get_spikes_by_events(self):
        return [event.spikes for period in self.children for event in period.children]

    def get_firing_std_dev(self):
        return np.std([self.concatenate(method='get_firing_rates', level=-2)])

    def get_cross_correlations(self, axis=0):
        return np.mean([pair.get_cross_correlations(axis=axis, stop_at=self.calc_opts.get('base', 'period'))
                        for pair in self.unit_pairs], axis=axis)

    def get_correlogram(self, axis=0):
        return np.mean([pair.get_correlogram(axis=axis, stop_at=self.calc_opts.get('base', 'period'))
                        for pair in self.unit_pairs], axis=axis)
    
    def get_waveform(self):
        if self.waveform is not None:
            return self.waveform
        else:
            phy = PhyInterface(self.calc_opts['data_path'], self.parent.identifier)
            electrodes = phy.cluster_dict[self.cluster_id]['electrodes']
            wf = phy.get_mean_waveforms(self.cluster_id, electrodes)
            self.waveform = wf
            return wf


class UnitPair:
    pass



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
        self.start = self.onset
        self.stop = self.onset + self.duration*self.sampling_rate
        
    def get_events(self):
        self._events = [SpikeEvent(self, self.unit, start, i) 
                        for i, start in enumerate(self.event_starts)]


class SpikeEvent(Event, RateMethods, BinMethods):
    def __init__(self, period, unit, start,  index):
        super().__init__(period, index)
        self.unit = unit
        self.private_cache = {}
        self._start = start

    @property
    def start(self):
        self._start - self.pre_stim*self.sampling_rate

    @property
    def stop(self):
        self._start + self.post_stim*self.sampling_rate

    @property
    def spike_range(self):
        return (-self.pre_stim, self.post_stim)
    
    def get_spike_counts(self):
        return calc_hist(self.spikes, self.num_bins_per, self.spike_range)

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
    

class SpikePrepMethods:

    def select_spike_children(self):
        if self.selected_neuron_type:
            return getattr(self, self.selected_neuron_type)
        else: 
            return self.units['good']

    
