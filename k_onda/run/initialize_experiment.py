import json
from copy import deepcopy
from collections import defaultdict
from pathlib import PosixPath

from scipy.signal import firwin, lfilter
import h5py

from k_onda.data.experiment_group_animal import Experiment, Group, Animal
from k_onda.utils import group_to_dict, PrepMethods


class Initializer(PrepMethods):

    def __init__(self, config):
        if type(config) == dict:
            self.exp_info = config
        elif type(config) in [str, PosixPath]:
            with open(config, 'r',  encoding='utf-8') as file:
                data = file.read()
                self.exp_info = json.loads(data)
        else:
            raise ValueError('Unknown input type')
        self.group_names = self.exp_info.get('group_names', [])
        self.animals_info = self.exp_info['animals']
        self.neuron_types = self.exp_info.get('neuron_types')
        self.neuron_classification_rule = self.exp_info.get('neuron_classification_rule')
        self.sampling_rate = self.exp_info.get('sampling_rate')
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
        self.groups = [
            Group(name=group, 
                  animals=[animal for animal in self.animals if animal.group_name == group])
            for group in self.group_names]
        self.experiment.initialize_data_sources(self.animals, groups=self.groups)
        return self.experiment

    def init_animal(self, animal_info):  
        animal = Animal(animal_info['identifier'], animal_info=animal_info, 
                        neuron_types=self.neuron_types)
        self.get_periods_from_nev(animal)
        return animal
           
    def get_periods_from_nev(self, animal):
        codes_and_onsets = None
        for period_info in animal.period_info.values():
            if 'nev' not in period_info:
                continue
            if codes_and_onsets is None:
                codes_and_onsets = self.get_onsets_from_nev(animal)
            nev = period_info['nev']
            indices = nev.get('indices')
            if not indices:
                indices = range(len(codes_and_onsets[nev['code']]))
            period_info['onsets'] = [
                onset 
                for i, onset in enumerate(codes_and_onsets[nev['code']]) 
                if i in indices
                ]
     
    def get_onsets_from_nev(self, animal):
        file_path = animal.animal_info.get('nev_file_path')
        if not file_path:
            file_path = animal.construct_path('nev')

        with h5py.File(file_path, 'r') as mat_file:
            data = group_to_dict(mat_file['NEV'])
            onsets_and_codes = defaultdict(list)

            for i, code in enumerate(data['Data']['SerialDigitalIO']['UnparsedData'][0]):
                onset = int(data['Data']['SerialDigitalIO']['TimeStamp'][i][0])
                onsets_and_codes[code].append(onset)

            return onsets_and_codes

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



