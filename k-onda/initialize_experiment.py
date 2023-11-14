import os
import scipy.io as sio
import numpy as np
import csv
from neo.rawio import BlackrockRawIO
from scipy.signal import resample

from spike import Experiment, Group, Animal, Unit
from lfp import LFPExperiment
from behavior import Behavior
from context import data_type_context, neuron_type_context, period_type_context

CAROLINA_DATA_PATH = '/Users/katie/likhtik/CH_forKatie/'

PL_ELECTRODES = {
    'IG154': (4, 6), 'IG155': (12, 14), 'IG156': (12, 14), 'IG158': (7, 14), 'IG160': (1, 8), 'IG161': (9, 11),
    'IG162': (13, 3), 'IG163': (14, 8), 'IG175': (15, 4), 'IG176': (11, 12), 'IG177': (15, 4), 'IG178': (6, 14),
    'IG179': (13, 15), 'IG180': (15, 4)
}


carolina_neuron_classification_rule = {
    'PV_IN': lambda i, categorized_unit_data: categorized_unit_data['good'][i][8][0][0] > 2,
    'ACH': lambda i, categorized_unit_data: categorized_unit_data['good'][i][8][0][0] <= 2
}

itamar_neuron_classification_rule = {
    'IN': lambda i, categorized_unit_data: categorized_unit_data['good'][i][8][0][0] > 1,
    'PN': lambda i, categorized_unit_data: categorized_unit_data['good'][i][8][0][0] <= 1
}


def init_units(entry, animal, neuron_classification_rule):
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
        for category in ['good', 'MUA'] for unit in units_w_spikes[category]
    ]

    for i, unit in enumerate(animal.units['good']):
        for key in neuron_classification_rule:
            if neuron_classification_rule[key](i, categorized_unit_data):
                unit.neuron_type = key
        unit.fwhm_microseconds = categorized_unit_data['good'][i][6][0][0]
        getattr(animal, unit.neuron_type).append(unit)

    return initialized_units


def init_animal(entry, neuron_types=('IN', 'PN')):
    name = entry[1][0]
    condition = entry[2][0]
    tone_period_onsets = entry[4][0]
    tone_onsets_expanded = entry[6][0]
    return Animal(name, condition, tone_period_onsets=tone_period_onsets, tone_onsets_expanded=tone_onsets_expanded,
                  neuron_types=neuron_types)


def init_hpc_animal(entry, ):
    conditions = {'IG155': 'stressed', 'IG162': 'control', 'IG171': 'control', 'IG173': 'control',
                  'IG174': 'stressed', 'IG175': 'stressed'}

    name = entry[1][0]
    condition = conditions[name]
    tone_period_onsets = entry[2][0]
    tone_onsets_expanded = entry[3][0]
    return Animal(name, condition, tone_period_onsets=tone_period_onsets, tone_onsets_expanded=tone_onsets_expanded)


def init_carolina_animal(entry):
    neuron_types = ('PV_IN', 'ACH')
    conditions = {'CH272': 'control', 'CH274': 'arch', 'CH275': 'arch'}

    name = entry[1][0]
    condition = conditions[name]
    tone_period_onsets = entry[4][0]
    tone_onsets_expanded = entry[6][0]
    return Animal(name, condition, tone_period_onsets=tone_period_onsets, tone_onsets_expanded=tone_onsets_expanded,
                  neuron_types=neuron_types)


def read_itamar_spreadsheet():
    data_dict = {}

    with open(os.path.join('/Users/katie/likhtik/data', 'percent_freezing.csv'), 'r', encoding='utf-8-sig') as csvfile:
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


def get_itamar_raw_lfp(animal):
    file_path = os.path.join(animal.data_path, animal.identifier, 'Safety')
    reader = BlackrockRawIO(filename=file_path, nsx_to_load=3)
    reader.parse_header()
    if animal.identifier in PL_ELECTRODES:
        pl1, pl2 = PL_ELECTRODES[animal.identifier]
    else:
        print(f"no selected electrodes found for PL for animal {animal.identifier}, using 1 and 3")
        pl1, pl2 = (1, 3)
    return {
        'hpc': reader.nsx_datas[3][0][:, 0],
        'bla': reader.nsx_datas[3][0][:, 2],
        'pl': np.mean([reader.nsx_datas[3][0][:, pl1], reader.nsx_datas[3][0][:, pl2]], axis=0)
    }


def get_carolina_raw_lfp(animal):
    file_path = os.path.join(CAROLINA_DATA_PATH, animal.identifier, 'EXTREC')
    reader = BlackrockRawIO(filename=file_path, nsx_to_load=3)
    reader.parse_header()
    data_to_return = {
        'bla': reader.nsx_datas[3][0][:, 0],
        'il': reader.nsx_datas[3][0][:, 2]
    }
    reader = BlackrockRawIO(filename=file_path, nsx_to_load=5)
    reader.parse_header()
    bf_data = np.mean([reader.nsx_datas[5][0][:, 0], reader.nsx_datas[5][0][:, 1]], axis=0)
    original_rate = 30000
    new_rate = 2000
    num_samples = len(bf_data)

    # Calculate the number of samples in the downsampled data
    new_num_samples = int(num_samples * new_rate / original_rate)

    # Resample
    downsampled_bf = resample(bf_data, new_num_samples)
    data_to_return['bf'] = downsampled_bf
    return data_to_return


mat_contents = sio.loadmat(os.path.join(CAROLINA_DATA_PATH, 'single_cell_data.mat'))['single_cell_data']
entries = mat_contents[0]
animals = [init_carolina_animal(entry) for entry in entries]
[init_units(entry, animal, carolina_neuron_classification_rule) for entry, animal in zip(entries, animals)]

# hpc_mat_contents = sio.loadmat(os.path.join(data_path, 'hpc_power_test_data.mat'))['data']
# entries = hpc_mat_contents[0]
# hpc_power_animals = [init_hpc_animal(entry) for entry in entries]
# animals += hpc_power_animals

# experiment = Experiment({name: Group(name=name, animals=[animal for animal in animals if animal.condition == name])
#                          for name in ['control', 'stressed']})

experiment = Experiment({name: Group(name=name, animals=[animal for animal in animals if animal.condition == name])
                         for name in ['control', 'arch']})

# experiment.categorize_neurons()

experiment.subscribe(data_type_context)
experiment.subscribe(neuron_type_context)
experiment.subscribe(period_type_context)

lfp_experiment = LFPExperiment(experiment, get_carolina_raw_lfp)
lfp_experiment.subscribe(data_type_context)
lfp_experiment.subscribe(period_type_context)
#
# behavior_experiment = Behavior(experiment, read_itamar_spreadsheet())


behavior_experiment = 'bar'





