

import numpy as np
import pint 
import pint_xarray
from collections import defaultdict

from .time import Session
from .sources import (
    Collection,
    LFPChannel,
    LFPRecording,
    PhyOutput,
    initialize_neurons_from_phy,
)
from .central import LFP_SAMPLING_RATE
from .signals import TimeFrequencySignal
from .transformers import FrequencyBand
from .transformers import Spectrogram
from .central import ureg


lfp_data_loader_config = {
    "file_path": "/Users/katie/likhtik/IG_INED_SAFETY_RECALL/INED18/INED18.ns3",
    "file_ext": "ns3"
}

spike_data_loader_config = {
    'file_path': "/Users/katie/likhtik/IG_INED_SAFETY_RECALL/IG180/"
}

session_config = {
    'nev': {
        'path': "/Users/katie/likhtik/IG_INED_SAFETY_RECALL/INED18/INED18.mat",
        'epochs': {
            'tone': {
                'code': 65502,
                'duration': 30
                }
    }
}}

filter_config = {"method": "iir_notch", "f_lo": 59, "f_hi": 61
    }

freqs = np.arange(1, 21, 1)

power_config = {
    "freqs": freqs, 
    "decim": 20, 
    "n_cycles": freqs * 0.5,
    "time_bandwidth": 2, 
    "output": "power"}


class Experiment:
    
    def __init__(self, experiment_id, subjects=None):
        self.experiment_id = experiment_id
        if subjects is None:
            self.subjects = []
        else:
            self.subjects = subjects

    @property
    def all_neurons(self):
        return Collection([n for s in self.subjects for n in s.neurons])


class Subject:
    
    def __init__(self, subject_id):
        self.subject_id = subject_id
        self.sessions = []
        self.data_identities = defaultdict(list)


    def add_to_experiment(self, experiment):
        experiment.subjects.append(self)

    @property
    def experiments(self):
        return list({s.experiment for s in self.sessions})
    
    @property
    def neurons(self):
        return self.data_identities['neuron']
    

experiment = Experiment("IG_INED_SAFETY_RECALL")
animal = Subject("INED18")
animal.add_to_experiment(experiment)

session = Session(experiment, animal, session_config, ureg)

recording = LFPRecording(session, lfp_data_loader_config, sampling_rate=LFP_SAMPLING_RATE)

phy_output = PhyOutput(session, spike_data_loader_config)

neurons = initialize_neurons_from_phy(phy_output)

#experiment.all_neurons.signals().median_filter(key='waveform')
# recording an idea I had: you should be able to define mini pipelines, assign them to identities or components with names
# and then do extract features with the named pipeline

# # Vectorized
# experiment.all_neurons
#     .stack_signals(stack_dim='spikes').
#     .reduce(key='waveforms', dim='electrodes', method='mean')
#     .median_filter(key='waveforms', kernel_sizes={'samples': 5})
#     .split_signals()
#     .group_by('identity')
#     .extract_features('firing_rate', 'fwhm')
#     .kmeans()
#     .apply_labels(to='neuron')


# # Iterating
# experiment.all_neurons
#     .reduce(key='waveforms', dim='electrode s', method='mean')
#     .median_filter(key='waveforms', kernel_sizes={'samples': 5})
#     .group_by('identity')





lfp_channel_1 = LFPChannel(recording, channel_idx=1)
lfp_channel_2 = LFPChannel(recording, channel_idx=2)


epoch_0 = session.epochs['tone'][0]
epoch_1 = session.epochs['tone'][1]

preprocessed_signal_1 = (
    lfp_channel_1
    .scale(.25)
    .filter(filter_config))



preprocessed_signal_1 = (
    lfp_channel_1
    .scale(.25)
    .filter(filter_config)
    .normalize("rms")
    )

epoch_0_power = preprocessed_signal_1.spectrogram(power_config).window(epoch_0)
epoch_0_power.data


power_calculator = Spectrogram(power_config)

epoch_0_power_sig_1 = power_calculator(preprocessed_signal_1).window(epoch_0).data
epoch_1_power_sig_1 = power_calculator(preprocessed_signal_1).window(epoch_1).data

assert not epoch_0_power_sig_1.equals(epoch_1_power_sig_1)

preprocessed_signal_2 = (
    lfp_channel_2
    .scale(.25)
    .filter(filter_config)
    .normalize("rms")
    )

epoch_0_power_sig_2 = power_calculator(preprocessed_signal_2).window(epoch_0).data

assert not epoch_0_power_sig_2.equals(epoch_0_power_sig_1)


fb = FrequencyBand(4, 8)

band_power = (
    epoch_0_power
    .band(fb)) 

band_power = (
    epoch_0_power.select(frequency=(4, 8)).data
)

threshold_1 = epoch_0_power_sig_1.threshold('lt', 100)
threshold_2 = epoch_0_power_sig_2.threshold('gt', 20)

intersection = threshold_1 & threshold_2



masked_epoch_0_power_sig_2 = epoch_0_power_sig_2 & epoch_0_power_sig_2.threshold('gt', 20)

masked_epoch_0_power_sig_1 = epoch_0_power_sig_2 


masked_epoch_0_power_sig_2.data

spikes_and_filtered_waveforms = (experiment
 .all_neurons
 .stack_signals(dim='spikes')
 .reduce(key='waveforms', dim='electrodes', method='mean')
 .median_filter(key='waveforms', kernel_sizes={'samples': 5}).data)


spikes_and_filtered_waveforms = (experiment
 .all_neurons
 .stack_signals(dim='spikes')
 .reduce(key='waveforms', dim='electrodes', method='mean')
 .median_filter(key='waveforms', kernel_sizes={'samples': 5})
 .unstack_signals(dim='spikes')
 .group_by('neuron')
 )






