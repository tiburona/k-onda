from base_data import Data
from plotting_helpers import formatted_now
from spike_methods import SpikeMethods
from lfp_methods import LFPMethods, LFPPrepMethods
from lfp_data_structures import LFPPeriod
from collections import defaultdict
from period_constructor import PeriodConstructor
from bins import BinMethods


class Experiment(Data):

    _name = 'experiment'

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
        self.children = self.groups

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
        return [period for animal in self.all_animals for period in animal.get_all('lfp_periods')]

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
        if self.kind_of_data == 'spike':
            for unit in self.all_units:
                unit.spike_prep()
        elif self.kind_of_data == 'lfp':
            for animal in self.all_animals:
                if not animal.include():
                    continue
                animal.lfp_prep()
        elif self.kind_of_data == 'behavior':
            pass
        else:
            raise ValueError("Unknown data class") 

    def validate_lfp_events(self, calc_opts):
        self.calc_opts = calc_opts
        self.initialize_data()
        for animal in self.all_animals:
            animal.validate_events()
        

class Group(Data, SpikeMethods, LFPMethods, BinMethods):
    _name = 'group'

    def __init__(self, name, animals=None, experiment=None):
        super().__init__()
        self.identifier = name
        self.animals = animals if animals else []
        for animal in self.animals:
            animal.parent = self
        self.experiment = experiment
        self.parent = experiment
        self.children = self.animals


class Animal(Data, PeriodConstructor, SpikeMethods, LFPPrepMethods, LFPMethods, BinMethods):
    _name = 'animal'

    def __init__(self, identifier, condition, animal_info, experiment=None, neuron_types=None):
        super().__init__()
        self.identifier = identifier
        self.condition = condition
        self.animal_info = animal_info
        self.experiment = experiment
        self.group = None
        self.period_info = animal_info['period_info'] if 'period_info' in animal_info is not None else {}
        if neuron_types is not None:
            for nt in neuron_types:
                setattr(self, nt, [])
        self.units = defaultdict(list)
        self.lfp_periods = defaultdict(list)
        self.mrl_calculators = defaultdict(list)
        self.coherence_calculators = defaultdict(list)
        self.correlation_calculators = defaultdict(list)
        self.granger_calculators = defaultdict(list)
        self.phase_relationship_calculators = defaultdict(list)
        self.raw_lfp = None 
        self._processed_lfp = {}
        self.kind_of_data_to_period_type = {
            'lfp': LFPPeriod
        }
        self.lfp_event_validity = defaultdict(dict)

    @property
    def children(self):
        return getattr(self, f"select_{self.kind_of_data}_children")()
