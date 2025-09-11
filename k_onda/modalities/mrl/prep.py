from k_onda.modalities.spike import SpikePrepMethods
from k_onda.modalities.lfp import LFPPrepMethods
from .data_structures import MRLCalculator

class MRLPrepMethods(SpikePrepMethods, LFPPrepMethods):
    
    def mrl_prep(self):
        self.prepare_mrl_calculators()

    def prepare_mrl_calculators(self):
        for unit in self.units['good']:
            unit.mrl_calculators = {
                period_type: [MRLCalculator(unit, period=period) for period in periods] 
                for period_type, periods in self.lfp_periods.items()}

    def select_mrl_children(self):
        if self.selected_neuron_type:
            units = self.neurons[self.selected_neuron_type]
        else:
            units = [unit for units in self.units.values() for unit in units]

        return units