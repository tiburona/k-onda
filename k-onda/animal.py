from collections import defaultdict

from base_data import Data
from period_constructor import PeriodConstructor
from spike_methods import SpikeMethods
from lfp_methods import LFPPrepMethods
from lfp_methods import LFPMethods


class Animal(Data, PeriodConstructor, SpikeMethods, LFPPrepMethods, LFPMethods):
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
        self.lfp_periods = defaultdict(list)
        self.mrl_calculators = defaultdict(list)
        self.coherence_calculators = defaultdict(list)
        self.correlation_calculators = defaultdict(list)
        self.granger_calculators = defaultdict(list)
        self.phase_relationship_calculators = defaultdict(list)
        self.raw_lfp = None 
        self.group = None
        self._processed_lfp = {}