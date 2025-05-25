from collections import defaultdict
import functools

from ..data import Data
from ..bins import BinMethods
from ..period_constructor import PeriodConstructor
from k_onda.calc import (LFPPeriod, LFPMethods, SpikePeriod, SpikeMethods, SpikePrepMethods, 
                         NeuronClassifier, MRLPrepMethods, MRLMethods)
from k_onda.utils import formatted_now


# can't import this from utils for some weird reason
def sorted_prop(key):
    """Decorator to automatically fetch the sort key and apply sorting."""
    def decorator(func):
        @property
        @functools.wraps(func)
        def wrapper(self):
            items = func(self)
            sort = self.calc_opts.get('sort', {}).get(key)
            return self.sort(sort, items)
        return wrapper
    return decorator

class Experiment(Data, SpikePrepMethods, SpikeMethods, LFPMethods):

    _name = 'experiment'

    def __init__(self, info, **kwargs):
        super().__init__(**kwargs)
        self.exp_info = info
        self.identifier = info['identifier'] 
        self.now = formatted_now
        self.group_names = info.get('group_names', [])
        self._sampling_rate = info.get('sampling_rate')
        self._lfp_sampling_rate = info.get('lfp_sampling_rate')
        self.stimulus_duration = info.get('stimulus_duration')
        self.experiment = self
        self.groups = None
        self.all_groups = None
        self.all_animals = []
        self.neuron_classifier = NeuronClassifier(self)
        self._ancestors = [self]
        self.kind_of_data_to_period_type = {
            'lfp': LFPPeriod,
            'spike': SpikePeriod
        }
        self.initialized = []

    @property
    def ancestors(self):
        return self._ancestors
    
    @sorted_prop('group')
    def children(self):
        return self.groups if self.groups else [an for an in self.all_animals if an.include()]
    
    @sorted_prop('unit')
    def all_units(self):
        return [unit for animal in self.all_animals 
                for unit in animal.all_units if unit.include(check_ancestors=True)]

    @sorted_prop('period')
    def all_spike_periods(self):
        return [period for unit in self.all_units for period in unit.all_periods 
                if period.include(check_ancestors=True)]

    @sorted_prop('event')
    def all_spike_events(self):
        return [event for period in self.all_spike_periods for event in period.events 
                if event.include(check_ancestors=True)]

    @sorted_prop('unit_pair')
    def all_unit_pairs(self):
        return [unit_pair for unit in self.all_units for unit_pair in unit.get_pairs() 
                if unit_pair.include(check_ancestors=True)]
    
    @sorted_prop('period')
    def all_lfp_periods(self):
        return [period for animal in self.all_animals for period in animal.get_all('lfp_periods') 
                if period.include(check_ancestors=True)]
    
    @sorted_prop('event')
    def all_lfp_events(self):
        return [event for period in self.all_lfp_periods for event in period.events 
                if event.include(check_ancestors=True)]
    
    @sorted_prop('coherence_calculator')
    def all_coherence_calculators(self):
        return self.get_data_calculated_by_period('coherence_calculators')
    
    @sorted_prop('correlation_calculator')
    def all_correlation_calculators(self):
        return self.get_data_calculated_by_period('correlation_calculators')
    
    @sorted_prop('granger_calculator')
    def all_granger_calculators(self):
        return self.get_data_calculated_by_period('granger_calculators')
    
    @sorted_prop('phase_relationship_calculator')
    def all_phase_relationship_calculators(self):
        return self.get_data_calculated_by_period('phase_relationship_calculators')
    
    @sorted_prop('mrl_calculator')
    def all_mrl_calculators(self):
        return [mrl_calc for unit in self.all_units for mrl_calc in unit.get_all('mrl_calculators') 
                if mrl_calc.include(check_ancestors=True)]
    
    def initialize_data_sources(self, animals, groups=None):
        self.all_animals = animals
        self.groups = groups or []
        self.all_groups = groups or []
        if not self.groups:
            for animal in self.all_animals:
                animal.parent = self
        self.initialize_period_and_neuron_types

    def initialize_period_and_neuron_types(self):
        self.period_types = set(period_type for animal in self.all_animals 
                                for period_type in animal.period_info)
        self.neuron_types = set([unit.neuron_type for unit in self.all_units])

    def initialize_data(self):
        if 'kind_of_data' in self.calc_opts:
            getattr(self, f"{self.kind_of_data}_prep")()

    def delete_lfp_data(self, regions):
        for region in regions:
            for animal in self.all_animals:
                animal.delete_lfp_data(region)

    def spike_prep(self):
        self.prep_animals()
        if all(['neurons' in animal.initialized for animal in self.all_animals if animal.include()]):
            return
        if self.neuron_classifier.config:
            self.neuron_classifier.classify()
            for animal in [animal for animal in self.all_animals if animal.include()]:
                if not animal.neurons:
                   raise ValueError("This animal has not had units classified.")
        
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
          
    def validate_lfp_events(self, opts):
        self.calc_opts = opts['calc_opts']
        self.initialize_data()
        for animal in self.all_animals:
            animal.validate_events()
        

class Group(Data, SpikeMethods, LFPMethods, MRLMethods, BinMethods):
    _name = 'group'

    def __init__(self, name, animals=None, **kwargs):
        super().__init__(**kwargs)
        self.identifier = name
        self.animals = animals if animals else []
        self.parent = self.experiment
        for animal in self.animals:
            animal.parent = self
            animal.group = self

    @property
    def children(self):
        return [an for an in self.animals if an.include()]


class Animal(Data, PeriodConstructor, SpikeMethods, LFPMethods, MRLPrepMethods, MRLMethods, BinMethods):
    _name = 'animal'

    def __init__(self, identifier, animal_info, neuron_types=None, **kwargs):
        super().__init__(**kwargs)
        PeriodConstructor().__init__(**kwargs)
        self.identifier = identifier
        self.animal_info = animal_info
        self.group_name = animal_info.get('group_name')
        self.experiment.all_animals.append(self)
        self.conditions = animal_info.get('conditions')
        self.group = None
        self.period_info = animal_info['period_info'] if 'period_info' in animal_info is not None else {}
        if neuron_types is not None:
            for nt in neuron_types:
                setattr(self, nt, [])
        self._processed_lfp = {}
        self._processed_behavior = {}
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
        children = getattr(self, f"select_{self.kind_of_data}_children")()
        sort = self.calc_opts.get('sort', {})
        if sort:
            return self.sort(children)
        else:
            return children
    
    @property
    def all_units(self):
        return [unit for _, units in self.units.items() for unit in units]
