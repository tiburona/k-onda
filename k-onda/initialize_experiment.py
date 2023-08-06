import os
import scipy.io as sio
import numpy as np
from spike_data import Experiment, Group, Animal, Unit, Level
from context import Context
from neo.rawio import BlackrockRawIO


def init_units(entry, animal):
    categories = entry[3][0][0]
    category_names = [k for k in categories.dtype.fields.keys()]
    categorized_unit_data = dict(zip(category_names, [category[0] for category in categories]))

    units_w_spikes = {
        category: [
            {'spikes': [spike_time[0].astype(np.int64) for spike_time in unit[0]]}
            for unit in categorized_unit_data[category]
        ] for category in category_names
    }

    initialized_units = [
        Unit(animal, category, unit['spikes'])
        for category in units_w_spikes for unit in units_w_spikes[category]
    ]

    for i, unit in enumerate(animal.units['good']):
        unit.neuron_type = 'IN' if categorized_unit_data['good'][i][8][0][0] > 1 else 'PN'
        unit.fwhm_microseconds = categorized_unit_data['good'][i][6][0][0]
        getattr(animal, unit.neuron_type).append(unit)

    return initialized_units


def init_animal(entry):
    name = entry[1][0]
    condition = entry[2][0]
    tone_period_onsets = entry[4][0]
    tone_onsets_expanded = entry[6][0]
    return Animal(name, condition, tone_period_onsets=tone_period_onsets, tone_onsets_expanded=tone_onsets_expanded)


data_path = '/Users/katie/likhtik/data/'

mat_contents = sio.loadmat(os.path.join(data_path, 'single_cell_data.mat'))['single_cell_data']
entries = mat_contents[0]
animals = [init_animal(entry) for entry in entries]
[init_units(entry, animal) for entry, animal in zip(entries, animals)]

experiment = Experiment({name: Group(name=name, animals=[animal for animal in animals if animal.condition == name])
                         for name in ['control', 'stressed']})
experiment.all_units = [unit for group in experiment.groups for animal in group.animals
                        for unit in animal.units['good']]


data_type_context = Context('data_type_context')
neuron_type_context = Context('neuron_type_context')
Level.subscribe(data_type_context)
Level.subscribe(neuron_type_context)

# experiment.categorize_neurons()


def initialize_lfp_data():
    for animal in animals:
        file_path = os.path.join(data_path, 'single_cell_data', animal.identifier, 'Safety')
        reader = BlackrockRawIO(filename=file_path, nsx_to_load=3)
        reader.parse_header()
        animal.raw_lfp = {'hpc': reader.nsx_datas[3][0][:, 0], 'bla': reader.nsx_datas[3][0][:, 2],
                          'pl': np.mean([reader.nsx_datas[3][0][:, 1], reader.nsx_datas[3][0][:, 3]], axis=0)}
        a = 'foo'

initialize_lfp_data()



