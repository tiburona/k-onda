# an experiment can have types of data
# it can have groups
# it can have animals
# maybe an animal can have SpikeData, LFPData, BehaviorData

from data import Data, SpikeMethods
from data_generator import DataGenerator
from plotting_helpers import formatted_now
from collections import defaultdict, namedtuple
from period_constructor import PeriodConstructor
import numpy as np
from bisect import bisect_left as bs_left, bisect_right as bs_right
from math_functions import calc_rates, calc_hist, cross_correlation, correlogram




class Experiment(Data):

    name = 'experiment'

    def __init__(self, info):
        super().__init__()
        self.info = info
        self.identifier = info['identifier'] + formatted_now()
        self.conditions = info['conditions']
        self._sampling_rate = info.get('sampling_rate')
        self._lfp_sampling_rate = info.get('lfp_sampling_rate')
        self.stimulus_duration = info.get('stimulus_duration')
        self.groups = None
        self.all_groups = None
        self.lfp_data = 'foo'

    @property
    def all_units(self):
        return [unit for animal in self.all_animals for unit in animal.units['good']]

    @property
    def all_spike_periods(self):
        return [period for unit in self.all_units for period in unit.all_periods]

    @property
    def all_spike_events(self):
        return [event for period in self.all_spike_periods for event in period.events]

    @property
    def all_unit_pairs(self):
        return [unit_pair for unit in self.all_units for unit_pair in unit.get_pairs()]
    
    @property
    def all_lfp_periods(self):
        return [period for animal in self.all_animals for period in animal.all_lfp_periods]

    def initialize_groups(self, groups):
        self.groups = groups
        self.all_groups = groups
        self.all_animals = [animal for group in self.groups for animal in group.animals]
        self.period_types = set(period_type for animal in self.all_animals 
                                for period_type in animal.period_info)
        self.neuron_types = set([unit.neuron_type for unit in self.all_units])
        for entity in self.all_animals + self.all_groups:
            entity.experiment = self

    def initialize_data(self):
        if self.data_class == 'spike':
            for unit in self.all_units:
                unit.prepare_periods()
                unit.children = [period for pt in unit.periods for period in unit.periods[pt]]
            for animal in self.all_animals:
                animal.children = animal.units['good']
        elif self.data_class == 'lfp':
            for animal in self.all_animals:
                animal.prepare_periods()
                if self.data_type == 'power':
                    animal.children = animal.lfp_periods
                else:
                    animal.children = getattr(animal, f"{self.data_type}_calculators")
        elif self.data_class == 'behavior':
            pass
        else:
            raise ValueError("Unknown data class")


class Group(Data, SpikeMethods):
    name = 'group'

    def __init__(self, name, animals=None, experiment=None):
        super().__init__()
        self.identifier = name
        self.animals = animals if animals else []
        for animal in self.animals:
            animal.parent = self
        self.experiment = experiment
        self.parent = experiment
        self.children = self.animals


class Animal(Data, PeriodConstructor, SpikeMethods):
    name = 'animal'

    def __init__(self, identifier, condition, animal_info, experiment=None, neuron_types=None):
        super().__init__()
        self.identifier = identifier
        self.condition = condition
        self.animal_info = animal_info
        self.experiment = experiment
        self.period_info = animal_info['period_info'] if 'period_info' in animal_info is not None else {}
        if neuron_types is not None:
            for nt in neuron_types:
                setattr(self, nt, [])
        self.units = defaultdict(list)
        self.raw_lfp = None 
        self.group = None
        self.parent = self.group
        self.children = []

  
class Unit(Data, PeriodConstructor, SpikeMethods):

    name = 'unit'
    
    def __init__(self, animal, category, spike_times, cluster_id, waveform, experiment=None, neuron_type=None, 
                 quality=None):
        super().__init__()
        self.animal = animal
        self.category = category
        self.spike_times = np.array(spike_times)
        self.cluster_id = cluster_id
        self.neuron_type = neuron_type
        self.waveform = waveform
        self.experiment = experiment
        self.animal.units[category].append(self)
        self.identifier = str(self.animal.units[category].index(self) + 1)
        self.quality = quality
        self.period_class = SpikePeriod
        self.periods = defaultdict(list)
        self.parent = animal
        self.cls = 'spike'
        
    @property
    def all_periods(self):
        return [period for key in self.periods for period in self.periods[key]]

    @property
    def firing_rate(self):
        return self.animal.sampling_rate * len(self.spike_times) / float(self.spike_times[-1] - self.spike_times[0])

    @property
    def unit_pairs(self):
        all_unit_pairs = self.get_pairs()
        pairs_to_select = self.data_opts.get('unit_pair')
        if pairs_to_select is None:
            return all_unit_pairs
        else:
            return [unit_pair for unit_pair in all_unit_pairs
                    if ','.join([unit_pair.unit.neuron_type, unit_pair.pair.neuron_type]) == pairs_to_select]

    def get_pairs(self):
        return [UnitPair(self, other) for other in [unit for unit in self.animal if unit.identifier != self.identifier]]

    def find_spikes(self, start, stop):
        return np.array(self.spike_times[bs_left(self.spike_times, start): bs_right(self.spike_times, stop)])

    def get_spikes_by_events(self):
        return [event.spikes for period in self.children for event in period.children]

    def get_spontaneous_firing(self):
        spontaneous_period = self.data_opts.get('spontaneous', 120)
        if not isinstance(spontaneous_period, tuple):
            start = self.earliest_period.onset - spontaneous_period * self.sampling_rate - 1
            stop = self.earliest_period.onset - 1
        else:
            start = spontaneous_period[0] * self.sampling_rate
            stop = spontaneous_period[1] * self.sampling_rate
        num_bins = round((stop-start) / (self.sampling_rate * self.data_opts['bin_size']))
        return calc_rates(self.find_spikes(start, stop), num_bins, (start, stop), self.data_opts['bin_size'])

    def get_firing_std_dev(self, period_types=None):
        if period_types is None:  # default: take all period_types
            period_types = [period_type for period_type in self.periods]
        return np.std([rate for period_type, periods in self.periods.items() for period in periods
                       for rate in period.get_all_firing_rates() if period_type in period_types])

    def get_cross_correlations(self, axis=0):
        return np.mean([pair.get_cross_correlations(axis=axis, stop_at=self.data_opts.get('base', 'period'))
                        for pair in self.unit_pairs], axis=axis)

    def get_correlogram(self, axis=0):
        return np.mean([pair.get_correlogram(axis=axis, stop_at=self.data_opts.get('base', 'period'))
                        for pair in self.unit_pairs], axis=axis)


class UnitPair:
    pass


class Period(Data):
    def __init__(self, index, period_type, period_info, onset, target_period=None, 
                 is_relative=False, experiment=None):
        self.identifier = index
        self.period_type = period_type
        self.onset = onset
        self.experiment = experiment
        self.period_info = period_info
        self._events = []
        self.shift = period_info.get('shift')
        self.duration = period_info.get('duration')
        self.reference_period_type = period_info.get('reference_period_type')
        self.target_period = target_period
        self.is_relative = is_relative

        
    @property
    def reference(self):
        if self.is_relative:
            return None
        else:
            return self.parent.periods[self.reference_period_type][self.identifier]
        
    @property
    def reference(self):
        if self.is_relative:
            return None
        if not self.reference_period_type:
            return None
        else:
            return self.periods[self.reference_period_type][self.identifier]
        

class SpikePeriod(Period, SpikeMethods):

    name = 'period'

    def __init__(self, unit, index, period_type, period_info, onset, events=None, 
                 target_period=None, is_relative=False, experiment=None):
        super().__init__(index, period_type, period_info, onset, experiment=experiment, 
                         target_period=target_period, is_relative=is_relative)
        self.unit = unit
        self.event_starts = events if events is not None else []
        self.animal = self.unit.animal
        self.parent = unit
        self.cls = 'spike'
    
    @property
    def children(self):
        return self.events
       
    @property
    def events(self):
        if not self._events:
            self.get_events()
        return self._events


    def get_events(self):
        events_settings = self.data_opts['events'].get(
            self.period_type, {'pre_stim': 0, 'post_stim': 1})
        pre_stim, post_stim = (events_settings[opt] * self.experiment.sampling_rate 
                               for opt in ['pre_stim', 'post_stim'])
        if self.is_relative:
            event_starts = self.target_period.event_starts - self.shift * self.experiment.sampling_rate
        else:
            event_starts = self.event_starts
        for i, start in enumerate(event_starts):
            spikes = self.unit.find_spikes(start - pre_stim, start + post_stim)
            self._events.append(
                SpikeEvent(
                    self, self.unit, 
                    [((spike - start) / self.experiment.sampling_rate) for spike in spikes],
                    [(spike / self.experiment.sampling_rate) for spike in spikes], i))

    def get_all_firing_rates(self):
        return [event.get_firing_rates() for event in self.events]
    
    def get_all_spike_counts(self):
        return [event.get_spike_counts() for event in self.events]

    def mean_firing_rate(self):
        return np.mean(self.get_firing_rates())
    
    def mean_spike_counts(self):
        return np.mean(self.get_spike_counts())


class Event(Data):

    name = 'event'

    def __init__(self, period, index):
        super().__init__()
        self.period = period
        self.period_type = self.period.period_type
        self.identifier = index
        events_settings = self.data_opts['events'].get(self.period_type, 
                                                       {'pre_stim': 0, 'post_stim': 1})
        self.pre_stim, self.post_stim = (events_settings[opt] for opt in ['pre_stim', 'post_stim'])
        self.duration = self.pre_stim + self.post_stim
        self.experiment = self.period.experiment
        self.data_cache = {}
        self.cache_objects.append(self)

    @property
    def reference(self):
        if self.period.is_relative:
            return None
        reference_period_type = self.period.reference_period_type
        if not reference_period_type:
            return None
        else:
            return self.period.parent.periods[reference_period_type][self.period.identifier]
        
    @property
    def num_bins_per_event(self):
        bin_size = self.data_opts.get('bin_size')
        pre_stim, post_stim = (self.data_opts['events'][self.period_type].get(opt) 
                               for opt in ['pre_stim', 'post_stim'])
        return round((pre_stim + post_stim) / bin_size)


class SpikeEvent(Event, SpikeMethods):
    def __init__(self, period, unit, spikes, spikes_original_times, index):
        super().__init__(period, index)
        self.unit = unit
        self.spikes = spikes
        self.spikes_original_times = spikes_original_times
        self.parent = period
        self.cls = 'spike'
        self.cache = {}
       
    def get_psth(self):
        rates = self.get_firing_rates() 
        self.selected_period_type = self.reference.period_type
        reference_rates = self.reference.get_firing_rates()
        self.selected_period_type = self.period_type
        rates -= reference_rates
        if self.data_opts.get('adjustment') == 'normalized':
            rates /= self.unit.get_firing_std_dev(period_types=self.period_type,)  # same as dividing unit psth by std dev 
        a = 'foo'  
        self.cache = {}    
        return rates

    def get_firing_rates(self):
        bin_size = self.data_opts['bin_size']
        spike_range = (-self.pre_stim, self.post_stim)
        if 'rates' in self.cache:
            return self.cache['rates']
        else:
            rates = calc_rates(self.spikes, self.num_bins_per_event, spike_range, bin_size)
            self.cache['rates'] = rates
        return rates
    
    def get_spike_counts(self):
        return calc_hist(self.spikes, self.num_bins_per_event, (-self.pre_stim, self.post_stim))

    def get_cross_correlations(self, pair=None):
        other = pair.periods[self.period_type][self.period.identifier].events[self.identifier]
        cross_corr = cross_correlation(self.get_unadjusted_rates(), other.get_unadjusted_rates(), mode='full')
        boundary = round(self.data_opts['max_lag'] / self.data_opts['bin_size'])
        midpoint = cross_corr.size // 2
        return cross_corr[midpoint - boundary:midpoint + boundary + 1]

    def get_correlogram(self, pair=None, num_pairs=None):
        max_lag, bin_size = (self.data_opts[opt] for opt in ['max_lag', 'bin_size'])
        lags = round(max_lag/bin_size)
        return correlogram(lags, bin_size, self.spikes, pair.spikes, num_pairs)

    def get_autocorrelogram(self):
        max_lag, bin_size = (self.data_opts[opt] for opt in ['max_lag', 'bin_size'])
        lags = round(max_lag / bin_size)
        return correlogram(lags, bin_size, self.spikes, self.spikes, 1)

    

class BehaviorEvent(Event):
    pass


class MRLCalculator:
    pass


class RegionRelationshipCalculator:
    pass

