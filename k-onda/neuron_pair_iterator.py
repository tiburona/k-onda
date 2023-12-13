import numpy as np


class NeuronPairIterator:

    def iterate_through_neuron_pairs(self, filling_func, events_property):
        array_to_fill = []
        if len(events_property) == 0:
            return np.array([np.nan])
        neuron_type_pair = self.data_opts['neuron_type_pair']
        if self.unit.neuron_type == neuron_type_pair[0]:
            if all([getattr(self.animal, neuron_type) for neuron_type in neuron_type_pair]):
                pairs = [self.find_equivalent(unit=unit) for unit in getattr(self.unit.animal, neuron_type_pair[1])]
                for pair in pairs:
                    array_to_fill.append(getattr(self, filling_func)(pair, len(pairs)))
        return np.nanmean(np.array(array_to_fill), axis=0)
