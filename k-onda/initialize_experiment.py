import scipy.io as sio
import numpy as np
from spike_data import Experiment, Group, Animal, Unit
from contexts import NeuronTypeContext, DataTypeContext


def init_animal(entry):
    name = entry[1][0]
    condition = entry[2][0]
    tone_period_onsets = entry[4][0]
    tone_onsets_expanded = entry[6][0]
    animal = Animal(name, condition, tone_period_onsets=tone_period_onsets, tone_onsets_expanded=tone_onsets_expanded)
    categories = entry[3][0][0]
    cat_names = [k for k in categories.dtype.fields.keys()]
    cat_units = dict(zip(cat_names, [category[0] for category in categories]))
    units = {cat: [{'spikes': [spike_time[0].astype(np.int64) for spike_time in unit[0]]} for unit in cat_units[cat]]
             for cat in cat_names}
    [Unit(animal, cat, unit['spikes']) for cat in units for unit in units[cat]]
    for i, unit in enumerate(animal.units['good']):
        unit.neuron_type = 'IN' if cat_units['good'][i][8][0][0] < 2 else 'PN'
    return animal


mat_contents = sio.loadmat('/Users/katie/likhtik/data/single_cell_data.mat')['single_cell_data']
animals = [init_animal(entry) for entry in mat_contents[0]]
for animal in animals:
    for neuron_type in ['PN', 'IN']:
        for unit in animal.units['good']:
            if unit.neuron_type == neuron_type:
                getattr(animal, neuron_type).append(unit)

experiment = Experiment({name: Group(name=name, animals=[animal for animal in animals if animal.condition == name])
                         for name in ['control', 'stressed']})
experiment.all_units = [unit for group in experiment.groups for animal in group.animals for unit in animal.units['good']]

neuron_type_context = NeuronTypeContext()
data_type_context = DataTypeContext()

for unit in experiment.all_units:
    unit.subscribe(data_type_context)

for animal in animals:
    animal.subscribe(data_type_context)
    animal.subscribe(neuron_type_context)

for group in experiment.groups:
    group.subscribe(data_type_context)
    group.subscribe(neuron_type_context)






