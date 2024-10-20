import os
import json
import numpy as np
from neo.rawio import BlackrockRawIO
from copy import deepcopy
import csv
from scipy.signal import firwin, lfilter


from experiment_group_animal import Experiment, Group, Animal
from spike import Unit
from behavior import Behavior


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
        self.neuron_types = self.exp_info.get('neuron_types')
        self.neuron_classification_rule = self.exp_info.get('neuron_classification_rule')
        self.sampling_rate = self.exp_info['sampling_rate']
        self.experiment = None
        self.groups = None
        self.animals = None
        self.raw_lfp = None
        self.lfp_experiment = None
        self.behavior_experiment = None
        self.behavior_data_source = None

    def init_experiment(self):
        self.experiment = Experiment(self.exp_info)
        self.animals = [self.init_animal(animal_info) for animal_info in self.animals_info]
        for animal, animal_info in zip(self.animals, self.animals_info):
            if 'units' in animal_info:
                self.init_units(animal_info['units'], animal)
        self.groups = [
            Group(name=condition, 
                  animals=[animal for animal in self.animals if animal.condition == condition],
                  experiment=self.experiment)
            for condition in self.conditions]
        self.experiment.initialize_groups(self.groups)
        return self.experiment

    def init_animal(self, animal_info):  
        animal = Animal(animal_info['identifier'], animal_info['condition'], animal_info=animal_info,
                        neuron_types=self.neuron_types, experiment=self.experiment)
        return animal

    def init_units(self, units_info, animal):
        for category in [cat for cat in ['good', 'MUA'] if cat in units_info]:
            for unit_info in units_info[category]:
                unit = Unit(animal, category, unit_info['spike_times'], unit_info['cluster'], 
                            waveform=unit_info.get('mean_waveform'), experiment=self.experiment)
                if category == 'good':
                    unit.neuron_type = unit_info.get('neuron_type')
                    unit.quality = unit_info.get('quality')
                    if unit.waveform:
                        unit.fwhm_microseconds = unit_info.get('fwhm') * 1000000
                    else:
                        unit.fwhm_microseconds = None
                    getattr(animal, unit.neuron_type).append(unit)

    def init_behavior_experiment(self):
        data_source = self.exp_info['behavior_data']
        behavior_id_column = self.exp_info['behavior_animal_id_column']
        behavior_data = {}
        if isinstance(data_source, str):
            with open(data_source, mode='r') as f:
                csv_reader = csv.DictReader(f)
                self.behavior_data_source = {row[behavior_id_column]: row for row in csv_reader}
        for animal in self.experiment.all_animals:
            if animal.identifier in self.behavior_data_source:
                animal_data = self.process_spreadsheet_row(animal)
                behavior_data[animal.identifier] = animal_data
        self.behavior_experiment = Behavior(self.experiment, self.exp_info, behavior_data)
        return self.behavior_experiment

    def process_spreadsheet_row(self, animal):
        row = self.behavior_data_source[animal.identifier]
        animal_data = {key: [] for key in animal.period_info.keys()}
        for period_type in animal_data:
            animal_data[period_type] = [
                float(row[key]) for key in row if self.process_column_name(key, period_type)]
        return animal_data

    @staticmethod
    def process_column_name(column_name, period_type):
        tokens = column_name.split(' ')
        if period_type.lower() != tokens[0].lower():
            return False
        try:
            int(tokens[1])
        except (ValueError, IndexError) as e:
            print(f"Skipping column {column_name} due to error {e}.  This is likely not a problem.")
            return False
        return True


def downsample(data, orig_freq, dest_freq):
    # Design a low-pass FIR filter
    nyquist_rate = dest_freq/ 2
    cutoff_frequency = nyquist_rate - 100  # For example, 900 Hz to have some margin
    numtaps = 101  # Number of taps in the FIR filter, adjust based on your needs
    fir_coeff = firwin(numtaps, cutoff_frequency, nyq=nyquist_rate)

    # Apply the filter
    filtered_data = lfilter(fir_coeff, 1.0, data)

    ratio = round(orig_freq/dest_freq)

    return filtered_data[::ratio]



