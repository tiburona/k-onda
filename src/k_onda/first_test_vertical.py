

import numpy as np
import pint 
import pint_xarray


from .model import Session, Experiment, Subject
from .sources import (
    LFPChannel,
    LFPRecording,
    PhyOutput, 
    SpikeCluster,
    Neuron
)
from .central import LFP_SAMPLING_RATE

from .transformers import Spectrogram
from .central import ureg


lfp_data_loader_config = {
    "path": "/Users/katie/likhtik/IG_INED_SAFETY_RECALL/INED18/INED18.ns3",
    "file_ext": "ns3",
    "row_to_brain_region": {0: 'bla'}
}

spike_data_loader_config = {
    'path': "/Users/katie/likhtik/IG_INED_SAFETY_RECALL/IG180/"
}


session_config = {
    'nev': {
        'path': "/Users/katie/likhtik/IG_INED_SAFETY_RECALL/INED18/INED18.mat",
    },
    'epochs': {
        'tone': {
            'inherits': 'base',
            'from_nev': True,
            'code': 65502,
            'duration': 30,
            'conditions': {'stimulus': 'tone'},
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


# Experiment.from_config(
#     'Safety_Recall',
#     global_config='/Users/katie/likhtik/analysis/k-onda-analysis/IG_INED_Safety/config/k_onda/ig_safety_recall.yaml'
#     )



    

experiment = Experiment("IG_INED_SAFETY_RECALL")
experiment.configure(top_level_config={'units_to_set': 
                      {'raw_sample': (1/30000, 's', 'rs'),
                       'lfp_sample': (1/2000, 's', 'ls')}})

experiment.initialize()

animal = experiment.create_subject("INED18")

session = Session(experiment, animal, session_config, ureg)

recording = LFPRecording(session, lfp_data_loader_config, sampling_rate=LFP_SAMPLING_RATE)

phy_output = PhyOutput(session, spike_data_loader_config)


def initialize_neurons_from_phy(phy_output):
    neurons = []
    for cluster_id, group in phy_output.cluster_groups.items():
        if group == "good":
            spike_cluster = SpikeCluster(phy_output, cluster_id)
            neuron = Neuron(data_components=[spike_cluster], config={'source': 'phy', 'match_by': None})
            neurons.append(neuron)
    return neurons

neurons = initialize_neurons_from_phy(phy_output)

# tone_vals = experiment.all_neurons.histogram(some_args).select_grid(tone_epochs)
# pretone_vals = experiment.all_neurons.histogram(some_args).select_grid(pretone_epochs).reduce('time_bin')

# evoked_vals = tone_vals - pretone_vals

# let's say experiment already knows about its epochs
# evoked_vals = experiment.all_neurons.histogram(some_args).select_grid(experiment.epochs['tone'], experiment.epochs['pretone']).pipe(a, b.reduce(some args)).pipe(a - b) 

# evoked_vals.aggregate('event').aggregate('epoch').aggregate('neuron', group_by='neuron_type').aggregate('subject', group_by='condition')
# evoked_vals.view('event', config).aggregate( 'subject', 'neuron', 'epoch', group_by={'neuron': 'neuron_type', 'subject': 'condition'}, method='mean'))

# yaml

# - view: 
#    - event
#    - config
# - aggregate
#   - subject

# There are two ways to generate markers -- from onsets
# experiment.all_neurons



lfp_channel_1 = LFPChannel(recording, channel_idx=1)
lfp_channel_2 = LFPChannel(recording, channel_idx=2)


epoch_0 = session.epochs.where(stimulus='tone')[0]
epoch_1 = session.epochs.where(stimulus='tone')[1]


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


initialized_experiment = experiment.configure('some_config').initialize()
classified_neurons = initialized_experiment.classify_neurons('some_config')
# .configure(some_config)
# .intialize()


# or Experiment.from_config(some_config).initialize(). etc

# experiment
# .configure(some_config)
# .intialize()
# .all_neurons
# .classify_neurons(some_config)   <- classify_neurons comes from a recipe for the already built pipeline
# .select('epochs', stimulus='tone', new_dim='trial', mode='pushdown')
# .select('events', window=(-0.05, 0.3), new_dim='pip')  <- this is gonna have to know to iterate over the epochs
# .count(some_config)
# .mean([{'across':'pip'}, {'across': 'trial'}, 
# {'across': 'neuron', 'group_by': 'neuron_type'}, {'across': 'animal', group_by: 'treatment_group'}]) <- mean is gonna have to route to Aggregator or ReduceDim as apporpriate
# 

# or: .mean(hierarchical=True, 'group_by': ['stimulus', 'neuron_type', 'treatment_group'])


top_level_config = {
    'root': '/Users/katie/likhtik/IG_INED_SAFETY_RECALL',
    'units_to_set': {
        'raw_sample': (1/30000, 's', 'rs'),
        'lfp_sample': (1/2000, 's', 'ls')
    }
}

subjects_config = {
   
    '_base': {
        'sessions': ['learning_day_1', 'learning_day_2', 'recall']
    },
    'IG144': {
        'conditions': {'treatment': 'control'},
        },
    'IG145': {
        'conditions': {'treatment': 'defeat'},
        'sessions': ['learning_day_1', 'learning_day_2', 'recall145']
        }
    }




pretone_vals = (
    classified_neurons
    .select(experiment.epochs, dim='trial', condition='pretone')
    .select('events', window=(-0.05, 0.3), new_dim='pip')
    .count('some_config')
    .mean({'across': 'pip'})
)

tone_vals = (
    classified_neurons
    .select(experiment.epochs, dim='trial', condition='pretone')
    .select('events', window=(-0.05, 0.3), dim='pip')  # for this to work an EpochSet needs to have an 'events' property and one of the things 'select' needs to be able to take as an argument is a string attr
    .count('some_config')
)

tone_vals_std = tone_vals.std()

normalized_tone_vals = (tone_vals - pretone_vals)/tone_vals_std

final_vals = (normalized_tone_vals
              .mean([
                  {'across': 'trial', 'group_by': 'stimulus_type'},
                  {'across': 'neuron', 'group_by': 'neuron_type'},
                  {'across': 'subject', 'group_by': 'treatment_group'}
                  ]))





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








