from base_data import Data
from plotting_helpers import formatted_now
from collections import defaultdict
from period_constructor import PeriodConstructor
import numpy as np


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
    def children(self):
        return self.events
       
    @property
    def events(self):
        if not self._events:
            self.get_events()
        return self._events

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
        

class Event(Data):

    name = 'event'

    def __init__(self, period, index):
        super().__init__()
        self.period = period
        self.parent = period
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


class TimeBin(Data):
    name = 'time_bin'

    def __init__(self, i, val, parent):
        self.parent = parent
        self.identifier = i
        self.val = val
        self.period = None
        if self.period:
            self.period_type = self.period.period_type

    @property
    def data(self):
        return self.val

    @property
    def time(self):
        if self.data_type == 'correlation':
            ts = np.arange(-self.data_opts['lags'], self.data_opts['lags'] + 1) / self.sampling_rate
        else:
            pre_stim, post_stim = [self.data_opts['events'][self.parent.period_type][val] 
                                for val in ['pre_stim', 'post_stim']]
            ts = np.arange(-pre_stim, post_stim, self.data_opts['bin_size'])
        
        # Round the timestamps to the nearest 10th of a microsecond
        ts = np.round(ts, decimals=7)

        return ts[self.identifier]