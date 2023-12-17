import os
import json
import numpy as np
from neo.rawio import BlackrockRawIO
from scipy.signal import resample

from spike import Experiment, Group, Animal, Unit
from lfp import LFPExperiment


class Initializer:

    def __init__(self, config):
        if type(config) == dict:
            self.exp_info = config
        elif type(config) == str:
            with open(config, 'r',  encoding='utf-8') as file:
                data = file.read()
                self.exp_info = json.loads(data)
        else:
            raise ValueError('Unknown input type')
        self.conditions = self.exp_info['conditions']
        self.animals_info = self.exp_info['animals']
        self.neuron_types = self.exp_info['neuron_types']
        self.neuron_classification_rule = self.exp_info['neuron_classification_rule']
        self.sampling_rate = self.exp_info['sampling_rate']
        self.experiment = None
        self.groups = None
        self.animals = None
        self.raw_lfp = None
        self.lfp_experiment = None

    def init_experiment(self):
        self.animals = [self.init_animal(animal_info) for animal_info in self.animals_info]
        for animal, animal_info in zip(self.animals, self.animals_info):
            self.init_units(animal_info['units'], animal)
        self.groups = [
            Group(name=condition, animals=[animal for animal in self.animals if animal.condition == condition])
            for condition in self.conditions]
        self.experiment = Experiment(self.exp_info, self.groups)
        return self.experiment

    def init_animal(self, animal_info):
        animal = Animal(animal_info['identifier'], animal_info['condition'],
                        block_info=animal_info['block_info'],
                        neuron_types=self.neuron_types)
        return animal

    def init_units(self, units_info, animal):
        for category in ['good', 'MUA']:
            for unit_info in units_info[category]:
                unit = Unit(animal, category, unit_info['spike_times'])
                animal.units[category].append(unit)
                if category == 'good':
                    classification_val = unit_info[self.neuron_classification_rule['column_name']]
                    classifications = self.neuron_classification_rule['classifications']
                    for neuron_type, values in classifications.items():
                        if classification_val in values:
                            unit.neuron_type = neuron_type
                    unit.fwhm_microseconds = unit_info['FWHM_microseconds']
                    getattr(animal, unit.neuron_type).append(unit)

    def init_lfp_experiment(self):
        self.raw_lfp = {}
        for animal in self.animals:
            self.raw_lfp[animal.identifier] = self.get_raw_lfp(animal)
        self.lfp_experiment = LFPExperiment(self.experiment, self.exp_info, self.raw_lfp)
        return self.lfp_experiment

    def get_raw_lfp(self, animal):
        file_path = os.path.join(self.exp_info['lfp_root'], animal.identifier, *self.exp_info['lfp_path_constructor'])
        reader = BlackrockRawIO(filename=file_path, nsx_to_load=3)
        reader.parse_header()
        data_to_return = {region: reader.nsx_datas[3][0][:, val] for region, val in self.exp_info['lfp_electrodes'].items()}
        if self.exp_info.get('lfp_from_stereotrodes') is not None:
            data_to_return = self.get_lfp_from_stereotrodes(animal, data_to_return, file_path)
        return data_to_return

    def get_lfp_from_stereotrodes(self, animal, data_to_return, file_path):
        lfp_from_stereotrodes_info = self.exp_info['lfp_from_stereotrodes']
        nsx_num = lfp_from_stereotrodes_info['nsx_num']
        reader = BlackrockRawIO(filename=file_path, nsx_to_load=nsx_num)
        reader.parse_header()
        for region, region_data in lfp_from_stereotrodes_info['electrodes'].items():
            electrodes = region_data if isinstance(region_data, list) else region_data[animal.identifier]
            data = np.mean([reader.nsx_datas[nsx_num][0][:, electrode] for electrode in electrodes], axis=0)
            num_samples = len(data)
            new_num_samples = int(num_samples * self.exp_info['lfp_sampling_rate'] / self.exp_info['sampling_rate'])
            downsampled_data = resample(data, new_num_samples)
            data_to_return[region] = downsampled_data
        return data_to_return


def init_experiment(json_path):
    exp_data = json.loads(json_path)
    conditions = exp_data['conditions']
    animals_info = exp_data['animals']
    neuron_types = exp_data['neuron_types']
    neuron_classification_rule = exp_data['neuron_classification_rule']
    animals = []
    for animal in animals:
        animals.append(init_animal(animal, neuron_types, neuron_classification_rule))
    groups = [Group(name=condition, animals=[animal for animal in animals_info if animal.condition == condition])
              for condition in conditions]
    experiment = Experiment(exp_data, groups)
    return experiment


def init_animal(animal_data, neuron_types, neuron_classification_rule):
    """
    Note:

    Every animal needs a `block_info` dict.  This should be a dictionary with `block_type` keys.
    Values in this dict should also be dicts.  They should contain the following keys if the block
    is not a reference block:

    `onsets`: a list of onset times, one for each block of that block_type
    `events`: a list of lists of event times within the block.

    If it is a reference block, i.e., a block that provides a baseline value, it should contain

    `target`: a block_type that provides the point in time from which the reference is shifted
    `shift`: a value, in seconds for how far back in time to locate the reference block
    `duration`: the duration of the reference block. Optional, if not included duration will be that of
    the target block.
    `is_reference`: True
    """

    name = animal_data['name']
    block_info = animal_data['block_info']
    condition = animal_data['condition']
    units = animal_data['units']
    animal = Animal(name, condition, block_info=block_info, neuron_types=neuron_types)
    init_units(units, animal, neuron_classification_rule)
    return animal


def init_units(units, animal, neuron_classification_rule):
    """

    Notes: the format of `neuron_classification_rule` is as follows:
    {column_name: <column_name>,
    classifications:
        {<neuron_type1>: [<value1>, <value2>, ...], <neuron_type2>: [<value3>, ...], ...}
    }
    """

    for category in ['good', 'MUA']:
        for unit_data in units[category]:
            unit = Unit(animal, category, unit_data['spike_times'])
            classification_val = unit_data[neuron_classification_rule['column_name']]
            classifications = neuron_classification_rule['classifications']
            for neuron_type, values in classifications.items():
                if classification_val in values:
                    unit.neuron_type = neuron_type
            unit.fwhm_microseconds = unit_data['fwhm_microseconds']
            animal.units[category].append(unit)
            getattr(animal, unit.neuron_type).append(unit)



def get_lfp_from_stereotrodes(animal, data_to_return, file_path, lfp_from_stereotrodes, exp_info):
    nsx_num = lfp_from_stereotrodes['nsx_num']
    reader = BlackrockRawIO(filename=file_path, nsx_to_load=nsx_num)
    reader.parse_header()
    for region, region_data in lfp_from_stereotrodes.items():
        animal_specific = region_data.get('animal_specific')
        if animal_specific is not None:
            electrodes = animal_specific[animal.identifier]
        else:
            electrodes = region_data['electrodes']
        data = np.mean([reader.nsx_datas[nsx_num][0][:, electrode] for electrode in electrodes], axis=0)
        original_rate = exp_info['sampling_rate']
        new_rate = exp_info['lfp_sampling_rate']
        num_samples = len(data)
        new_num_samples = int(num_samples * new_rate / original_rate)
        downsampled_data = resample(data, new_num_samples)
        data_to_return[region] = downsampled_data
    return data_to_return


def get_raw_lfp(animal, json_path):
    exp_info = json.loads(json_path)
    lfp_root = exp_info['lfp_root']
    sub_dirs = exp_info['lfp_path_constructor']
    lfp_electrodes = exp_info['lfp_electrodes']
    file_path = os.path.join(lfp_root, animal.identifier, *sub_dirs)
    reader = BlackrockRawIO(filename=file_path, nsx_to_load=3)
    reader.parse_header()
    data_to_return = {region: reader.nsx_datas[3][0][:, val] for region, val in lfp_electrodes}
    lfp_from_stereotrodes = exp_info.get('lfp_from_stereotrodes')
    if lfp_from_stereotrodes is not None:
        data_to_return = get_lfp_from_stereotrodes(animal, data_to_return, file_path, lfp_from_stereotrodes,
                                                   exp_info)
    return data_to_return





