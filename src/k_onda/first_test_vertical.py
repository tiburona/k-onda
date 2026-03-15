

import numpy as np
import pint 
import pint_xarray


from .model import Session, Experiment, Subject
from .sources import (
    LFPChannel,
    LFPRecording,
    PhyOutput,
    initialize_neurons_from_phy,
)
from .central import LFP_SAMPLING_RATE

from .transformers import FrequencyBand
from .transformers import Spectrogram
from .central import ureg


lfp_data_loader_config = {
    "file_path": "/Users/katie/likhtik/IG_INED_SAFETY_RECALL/INED18/INED18.ns3",
    "file_ext": "ns3",
    "row_to_brain_region": {0: 'bla'}
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
#     .unstack_signals()
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


label_spec = """
- type: classifier
  feature: fwhm
  order: ascending
  labels:
    - IN
    - PN
"""

categorized_neurons = (
  experiment
 .all_neurons
 .stack_signals(dim='spikes')
 .reduce(key='waveforms', dim='electrodes', method='mean')
 .median_filter(key='waveforms', kernel_sizes={'samples': 5})
 .unstack_signals()
 .extract_features('fwhm', 'firing_rate', group_by='neuron')
 .normalize(method='zscore', dim='index')
 .kmeans(n_clusters=2, random_state=0)
 .classify('neuron_type', label_spec=label_spec)
 )


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
 .unstack_signals()
)


selected_data =(spikes_and_filtered_waveforms
 .select(epoch_0)
 .signals[0].data
)








# spikes_and_filtered_waveforms = (experiment
#  .all_neurons
#  .stack_signals(dim='spikes')
#  .reduce(key='waveforms', dim='electrodes', method='mean')
#  .median_filter(key='waveforms', kernel_sizes={'samples': 5})
#  .unstack_signals(dim='spikes')
#  .group_by('neuron')
#  .extract_features('fwhm', 'firing_rate')
#  )





preprocessed_signal_1 = (
    lfp_channel_1
    .signal
    .scale(.25)
    .filter(filter_config))



preprocessed_signal_1 = (
    lfp_channel_1
    .signal
    .scale(.25)
    .filter(filter_config)
    .normalize("rms")
    )

epoch_0_power = preprocessed_signal_1.spectrogram(power_config).span(epoch_0)
epoch_0_power.data


power_calculator = Spectrogram(power_config)

epoch_0_power_sig_1 = power_calculator(preprocessed_signal_1).span(epoch_0)
epoch_1_power_sig_1 = power_calculator(preprocessed_signal_1).span(epoch_1)

assert not epoch_0_power_sig_1.data.equals(epoch_1_power_sig_1.data)

preprocessed_signal_2 = (
    lfp_channel_2
    .signal
    .scale(.25)
    .filter(filter_config)
    .normalize("rms")
    )

epoch_0_power_sig_2 = power_calculator(preprocessed_signal_2).span(epoch_0)

assert not epoch_0_power_sig_2.data.equals(epoch_0_power_sig_1.data)


fb = FrequencyBand(4, 8)

band_power = (
    epoch_0_power
    .band(fb)) 

band_power = (
    epoch_0_power.select(frequency=(4, 8)).data
)



threshold_1 = epoch_0_power.threshold('lt', 100)
threshold_2 = epoch_0_power_sig_2.threshold('gt', 20)

intersection = threshold_1 & threshold_2



masked_epoch_0_power_sig_2 = epoch_0_power_sig_2 & epoch_0_power_sig_2.threshold('gt', 20)

masked_epoch_0_power_sig_1 = epoch_0_power_sig_2 


masked_epoch_0_power_sig_2.data








