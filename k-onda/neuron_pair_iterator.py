import numpy as np


class NeuronPairIterator:

    def iterate_through_neuron_pairs(self, filling_func, events_property):
        neuron_type_pair = self.data_opts['neuron_type_pair']
        if not all([getattr(self.animal, neuron_type) for neuron_type in neuron_type_pair]) or not len(events_property) \
                or not self.unit.neuron_type == neuron_type_pair[0]:
            return np.full(int(self.data_opts['max_lag']/self.data_opts['bin_size']), np.nan)
        array_to_fill = []
        pairs = [self.find_equivalent(unit=unit) for unit in getattr(self.unit.animal, neuron_type_pair[1])]
        for pair in pairs:
            array_to_fill.append(getattr(self, filling_func)(pair, len(pairs)))
        return np.nanmean(np.array(array_to_fill), axis=0)

