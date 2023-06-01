import scipy.io as sio
from spike_data import Experiment, Group, Animal, Unit


def init_animal(entry, Animal, Unit):
    name = entry[1][0]
    condition = entry[2][0]
    tone_period_onsets = entry[4][0]
    tone_onsets_expanded = entry[6][0]
    animal = Animal(name, condition, tone_period_onsets=tone_period_onsets, tone_onsets_expanded=tone_onsets_expanded)
    categories = entry[3][0][0]
    cat_names = [k for k in categories.dtype.fields.keys()]
    cat_units = dict(zip(cat_names, [category[0] for category in categories]))
    units = {cat: [{'spikes': [spike_time[0] for spike_time in unit[0]]} for unit in cat_units[cat]] 
             for cat in cat_names}
    {cat: [Unit(animal, cat, unit['spikes']) for unit in units[cat]] for cat in units}
    for i, unit in enumerate(animal.units['good']):
        unit.neuron_type = 'PN' if cat_units['good'][i][8][0][0] < 2 else 'IN'
    return animal


mat_contents = sio.loadmat('/Users/katie/likhtik/data/single_cell_data.mat')['single_cell_data']
animals = [init_animal(entry, Animal, Unit) for entry in mat_contents[0]]
experiment = Experiment({name: Group(name=name, animals=[animal for animal in animals if animal.condition == name])
                         for name in ['control', 'stressed']})