

import numpy as np
import pint 
import pint_xarray

from .time import Session
from .sources import LFPChannel, LFPRecording
from .central import LFP_SAMPLING_RATE
from .signal import TimeFrequencySignal
from .select_mixin import FrequencyBand
from .time_frequency_calculators import Spectrogram


data_loader_config = {
    "file_path": "/Users/katie/likhtik/IG_INED_SAFETY_RECALL/INED18/INED18.ns3",
    "file_ext": "ns3"
}

session_config = {
    'onsets_from_nev': {
        'nev_path': "/Users/katie/likhtik/IG_INED_SAFETY_RECALL/INED18/INED18.mat",
        'epochs': {
        'tone': {
            'nev_code': 65502,
            'duration': 30
        }
    }
    }
}

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


class Subject:
    
    def __init__(self, subject_id):
        self.subject_id = subject_id
        self.sessions = []

    def add_to_experiment(self, experiment):
        experiment.subjects.append(self)

    @property
    def experiments(self):
        return list({s.experiment for s in self.sessions})
    

experiment = Experiment("IG_INED_SAFETY_RECALL")
animal = Subject("INED18")
animal.add_to_experiment(experiment)

session = Session(experiment, animal, session_config)

recording = LFPRecording(session, data_loader_config, sampling_rate=LFP_SAMPLING_RATE)

lfp_channel_1 = LFPChannel(recording, channel_idx=1)
lfp_channel_2 = LFPChannel(recording, channel_idx=2)


epoch_0 = session.epochs['tone'][0]
epoch_1 = session.epochs['tone'][1]

preprocessed_signal_1 = (
    lfp_channel_1
    .scale(.25)
    .filter(filter_config)
    .normalize("rms")
    )

epoch_0_power = preprocessed_signal_1.spectrogram(power_config).window(epoch_0)

power_calculator = Spectrogram(power_config)

epoch_0_power_sig_1 = power_calculator(preprocessed_signal_1).window(epoch_0)
epoch_1_power_sig_1 = power_calculator(preprocessed_signal_1).window(epoch_1)

assert epoch_0_power_sig_1 != epoch_1_power_sig_1

preprocessed_signal_2 = (
    lfp_channel_2
    .scale(.25)
    .filter(filter_config)
    .normalize("rms")
    )

epoch_0_power_sig_2 = power_calculator(preprocessed_signal_2).window(epoch_0)

assert epoch_0_power_sig_2 != epoch_0_power_sig_1

preprocessed_signal_1.window(epoch_0).filter(filter_config)

signal_from_data = TimeFrequencySignal.from_data(epoch_0_power.data)

fb = FrequencyBand(4, 8)

band_power = (
    epoch_0_power
    .band(fb)) 

band_power = (
    epoch_0_power.select(frequency=(4, 8))
)

threshold = epoch_0_power_sig_2.threshold('lt', 20)


masked_epoch_0_power_sig_2 = epoch_0_power_sig_2 & epoch_0_power_sig_2.threshold('lt', 20)


masked_epoch_0_power_sig_2.data


  

