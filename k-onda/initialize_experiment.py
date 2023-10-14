import os
import scipy.io as sio
import numpy as np
import csv
from spike import Experiment, Group, Animal, Unit
from lfp import LFPExperiment
from behavior import Behavior
from context import data_type_context, neuron_type_context, period_type_context


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


def init_hpc_animal(entry, ):
    conditions = {'IG155': 'stressed', 'IG162': 'control', 'IG171': 'control', 'IG173': 'control',
                  'IG174': 'stressed', 'IG175': 'stressed'}

    name = entry[1][0]
    condition = conditions[name]
    tone_period_onsets = entry[2][0]
    tone_onsets_expanded = entry[3][0]
    return Animal(name, condition, tone_period_onsets=tone_period_onsets, tone_onsets_expanded=tone_onsets_expanded)


def read_itamar_spreadsheet():
    data_dict = {}

    with open(os.path.join(data_path, 'percent_freezing.csv'), 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            data_dict[row['ID']] = {
                'group': row['Group'],
                'pretone': [float(row['Pretone 1']), float(row['Pretone 2']), float(row['Pretone 3']),
                            float(row['Pretone 4']), float(row['Pretone 5'])],
                'tone': [float(row['Tone 1']), float(row['Tone 2']), float(row['Tone 3']), float(row['Tone 4']),
                         float(row['Tone 5'])]
            }
    return data_dict


data_path = '/Users/katie/likhtik/data/'

mat_contents = sio.loadmat(os.path.join(data_path, 'single_cell_data.mat'))['single_cell_data']
entries = mat_contents[0]
animals = [init_animal(entry) for entry in entries]
[init_units(entry, animal) for entry, animal in zip(entries, animals)]

# hpc_mat_contents = sio.loadmat(os.path.join(data_path, 'hpc_power_test_data.mat'))['data']
# entries = hpc_mat_contents[0]
# hpc_power_animals = [init_hpc_animal(entry) for entry in entries]
# animals += hpc_power_animals

experiment = Experiment({name: Group(name=name, animals=[animal for animal in animals if animal.condition == name])
                         for name in ['control', 'stressed']})

# experiment.categorize_neurons()

experiment.subscribe(data_type_context)
experiment.subscribe(neuron_type_context)

lfp_experiment = LFPExperiment(experiment)
lfp_experiment.subscribe(data_type_context)
lfp_experiment.subscribe(period_type_context)

behavior_experiment = Behavior(experiment, read_itamar_spreadsheet())






