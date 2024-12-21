from scipy.io import loadmat
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
import json

root = '/Users/katie/likhtik/MS_26'

data_dir = '/Users/katie/likhtik/MS_26/MS_26_Optrode_Extinction_Learning'

def group_to_dict(group):
    result = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = group_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            result[key] = item[()]
        else:
            result[key] = item
    return result


with h5py.File(os.path.join(data_dir, 'MS26_extinction_lightON.mat'), 'r') as mat_file:
    data = group_to_dict(mat_file['NEV'])
    light_on_timestamps = []
    tone_timestamps = []
    for i, code in enumerate(data['Data']['SerialDigitalIO']['UnparsedData'][0]):
        if code == 65534:
            light_on_timestamps.append(int(data['Data']['SerialDigitalIO']['TimeStamp'][i][0]))
        if code == 65502:
            tone_timestamps.append(int(data['Data']['SerialDigitalIO']['TimeStamp'][i][0]))



spike_times = np.load(os.path.join(data_dir, 'spike_times.npy'), allow_pickle=True).astype(int)
spike_clusters = np.load(os.path.join(data_dir, 'spike_clusters.npy'), allow_pickle=True).astype(int)


units = {49: {'spike_times': [], 'neuron_type': 'PN'}, 
         20: {'spike_times': [], 'neuron_type': 'PN'}, 
         4: {'spike_times': [], 'neuron_type': 'PN'}}

period_info = {
    'prelight': {'relative': True,  'shift': -5, 'duration': 5, 'target': 'light'},
    'light': {'relative': False, 'onsets': light_on_timestamps, 'reference_period_type': 'prelight',
              'duration': 5},
    'tone': {'relative': False, 'onsets': tone_timestamps, 'reference_period_type': 'prelight', 
             'duration': 30}
}

for i, cluster in enumerate(spike_clusters):
    if cluster in units:
        spike_time = int(spike_times[i][0])
        units[cluster]['spike_times'].append(spike_time)

units_info = {'good': [val | {'cluster': clust} for clust, val in units.items()]}



ms_26_info = {'identifier': 'MS26', 'period_info': period_info, 'units': units_info, 'condition': 'foo'}
exp_info = {}



exp_info['animals'] = [ms_26_info]
exp_info['neuron_types'] = ['PN']
exp_info['conditions'] = ['foo']
exp_info['sampling_rate'] = 30000
exp_info['identifier'] = 'MS26'


with open(os.path.join(root, 'init_config.json'), 'w', encoding='utf-8') as file:
    json.dump(exp_info, file, ensure_ascii=False, indent=4)


