import os
import json
import numpy as np
from neo.rawio import BlackrockRawIO
from scipy.signal import resample
from copy import deepcopy

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
        self.neuron_classification_rule = self.exp_info.get('neuron_classification_rule')
        self.sampling_rate = self.exp_info['sampling_rate']
        self.experiment = None
        self.groups = None
        self.animals = None
        self.raw_lfp = None
        self.lfp_experiment = None

    def init_experiment(self):
        self.animals = [self.init_animal(animal_info) for animal_info in self.animals_info]
        for animal, animal_info in zip(self.animals, self.animals_info):
            if 'units' in animal_info:
                self.init_units(animal_info['units'], animal)
        self.groups = [
            Group(name=condition, animals=[animal for animal in self.animals if animal.condition == condition])
            for condition in self.conditions]
        self.experiment = Experiment(self.exp_info, self.groups)
        return self.experiment

    def init_animal(self, animal_info):  # TODO make sure animal info gets all animal info so it can pass it to lfp
        animal = Animal(animal_info['identifier'], animal_info['condition'], animal_info=animal_info,
                        neuron_types=self.neuron_types)
        return animal

    def init_units(self, units_info, animal):
        for category in [cat for cat in ['good', 'MUA'] if cat in units_info]:
            for unit_info in units_info[category]:
                unit = Unit(animal, category, unit_info['spike_times'])
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
        path_constructor = deepcopy(self.exp_info['lfp_path_constructor'])
        if path_constructor[-1] == 'identifier':
            path_constructor[-1] = animal.identifier
        file_path = os.path.join(self.exp_info['lfp_root'], animal.identifier, *path_constructor)
        reader = BlackrockRawIO(filename=file_path, nsx_to_load=3)
        reader.parse_header()
        data_to_return = {region: reader.nsx_datas[3][0][:, val]
                          for region, val in animal.animal_info['lfp_electrodes'].items()}
        if animal.animal_info.get('lfp_from_stereotrodes') is not None:
            data_to_return = self.get_lfp_from_stereotrodes(animal, data_to_return, file_path)
        return data_to_return

    def get_lfp_from_stereotrodes(self, animal, data_to_return, file_path):
        lfp_from_stereotrodes_info = animal.animal_info['lfp_from_stereotrodes']
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

