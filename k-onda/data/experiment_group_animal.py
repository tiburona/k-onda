from collections import defaultdict

from data.data import Data
from calc.lfp.lfp import LFPPeriod, LFPMethods, LFPPrepMethods
from calc.spike.spike import SpikePeriod, SpikeMethods, SpikePrepMethods
from calc.mrl.mrl import MRLPrepMethods, MRLMethods
from data.period_constructor import PeriodConstructor
from calc.spike.neuron_classifier import NeuronClassifier
from data.bins import BinMethods
from utils.utils import formatted_now


class Experiment(Data, SpikePrepMethods):

    _name = 'experiment'

    def __init__(self, info):
        super().__init__()
        self.exp_info = info
        self.identifier = info['identifier'] 
        self.now = formatted_now
        self.conditions = info['conditions']
        self._sampling_rate = info.get('sampling_rate')
        self._lfp_sampling_rate = info.get('lfp_sampling_rate')
        self.stimulus_duration = info.get('stimulus_duration')
        self.experiment = self
        self.groups = None
        self.all_groups = None
        self.all_animals = []
        self.neuron_classifier = NeuronClassifier(self)
        self.children = self.groups
        self._ancestors = [self]
        self.kind_of_data_to_period_type = {
            'lfp': LFPPeriod,
            'spike': SpikePeriod
        }
        self.state = {}
        self.initialized = []

    @property
    def ancestors(self):
        return self._ancestors
    
    @property
    def all_units(self):
        return [unit for animal in self.all_animals 
                for unit in animal.all_units if unit.include(check_ancestors=True)]

    @property
    def all_spike_periods(self):
        return [period for unit in self.all_units for period in unit.all_periods 
                if period.include(check_ancestors=True)]

    @property
    def all_spike_events(self):
        return [event for period in self.all_spike_periods for event in period.events 
                if event.include(check_ancestors=True)]

    @property
    def all_unit_pairs(self):
        return [unit_pair for unit in self.all_units for unit_pair in unit.get_pairs() 
                if unit_pair.include(check_ancestors=True)]
    
    @property
    def all_lfp_periods(self):
        return [period for animal in self.all_animals for period in animal.get_all('lfp_periods') 
                if period.include(check_ancestors=True)]
    
    @property
    def all_mrl_calculators(self):
        return [mrl_calc for unit in self.all_units for mrl_calc in unit.get_all('mrl_calculators') 
                if mrl_calc.include(check_ancestors=True)]

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
        getattr(self, f"{self.kind_of_data}_prep")()

    def spike_prep(self):
        self.prep_animals()
        if 'neurons' in self.initialized:
            return
        self.neuron_classifier.classify()
        self.initialized.append('neurons')
        
    def lfp_prep(self):
        self.prep_animals()

    def mrl_prep(self):
        self.calc_opts['kind_of_data'] = 'spike'
        self.spike_prep()
        self.calc_opts['kind_of_data'] = 'lfp'
        self.lfp_prep()
        self.calc_opts['kind_of_data'] = 'mrl'
        self.prep_animals()

    def prep_animals(self):
        for animal in self.all_animals:
            if not animal.include():
                continue
            getattr(animal, f"{self.kind_of_data}_prep")()
          
    def validate_lfp_events(self, calc_opts):
        self.calc_opts = calc_opts
        self.initialize_data()
        for animal in self.all_animals:
            animal.validate_events()
        

class Group(Data, SpikeMethods, LFPMethods, MRLMethods, BinMethods):
    _name = 'group'

    def __init__(self, name, animals=None, experiment=None):
        super().__init__()
        self.identifier = name
        self.animals = animals if animals else []
        self.experiment = experiment
        self.parent = experiment
        for animal in self.animals:
            animal.parent = self
            animal.group = self
        self.children = self.animals


class Animal(Data, PeriodConstructor, SpikeMethods, LFPMethods, MRLPrepMethods, MRLMethods, BinMethods):
    _name = 'animal'

    def __init__(self, identifier, condition, animal_info, experiment=None, neuron_types=None):
        super().__init__()
        PeriodConstructor().__init__()
        self.identifier = identifier
        self.condition = condition
        self.animal_info = animal_info
        self.experiment = experiment
        self.experiment.all_animals.append(self)
        self.group = None
        self.period_info = animal_info['period_info'] if 'period_info' in animal_info is not None else {}
        if neuron_types is not None:
            for nt in neuron_types:
                setattr(self, nt, [])
        self._processed_lfp = {}
        self.units = defaultdict(list)
        self.neurons = defaultdict(list)
        self.lfp_periods = defaultdict(list)
        self.mrl_calculators = defaultdict(lambda: defaultdict(list))
        self.granger_calculators = defaultdict(list)
        self.coherence_calculators = defaultdict(list)
        self.correlation_calculators = defaultdict(list)
        self.phase_relationship_calculators = defaultdict(list)
        self.lfp_event_validity = defaultdict(dict)
        self.initialized = []

    @property
    def children(self):
        return getattr(self, f"select_{self.kind_of_data}_children")()
    
    @property
    def all_units(self):
        return [unit for _, units in self.units.items() for unit in units]
