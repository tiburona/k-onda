import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import medfilt
import os
import h5py
from copy import deepcopy
from pathlib import Path
import phylib
from phylib.io.model import load_model, read_python, _make_abs_path, TemplateModel


import inspect
print(inspect.getsource(load_model))
print(inspect.getsource(read_python))
#print(inspect.getsource(TemplateModel))



params_path = '/Users/katie/likhtik/MS_26/MS_26_Optrode_Extinction_Learning/params.py'

# Check if the file is being read correctly
with open(params_path, 'r') as file:
    print(file.read())

def get_template_params(params_path):
    print("Monkey-patched get_template_params called")
    
    # Return the dictionary you know works
    return {
        'dat_path': 'data_binary.bin',
        'n_channels_dat': 28,
        'dtype': 'int16',
        'offset': 0,
        'sample_rate': 30000.0,
        'hp_filtered': False,
        'dir_path': '/Users/katie/likhtik/MS_26/MS_26_Optrode_Extinction_Learning'
    }

phylib.io.model.get_template_params = get_template_params
ROOT_DIR = '/Users/katie/likhtik/MS_26/MS_26_Optrode_Extinction_Learning'

def read_python(path):
    path = Path(path)
    if not path.exists():
        raise IOError(f"Path {path} does not exist.")
    contents = path.read_text()
    metadata = {}
    exec(contents, {}, metadata)  # Execute the contents of the Python file
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata



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

def get_mean_waveforms(model, cluster_id, electrodes):
    channels_used = model.get_cluster_channels(cluster_id)
    indices = np.where(np.isin(channels_used, electrodes))[0]
    waveforms = model.get_cluster_spike_waveforms(cluster_id)
    filtered_waveforms = medfilt(waveforms, kernel_size=[1, 5, 1])
    averaged_waveforms = np.mean(filtered_waveforms[:, :, indices], axis=(0, 2))
    return averaged_waveforms

with h5py.File(os.path.join(ROOT_DIR, 'MS26_extinction_lightON.mat'), 'r') as mat_file:
    data = group_to_dict(mat_file['NEV'])
    light_on_timestamps = []
    tone_timestamps = []
    for i, code in enumerate(data['Data']['SerialDigitalIO']['UnparsedData'][0]):
        if code == 65534:
            light_on_timestamps.append(data['Data']['SerialDigitalIO']['TimeStamp'][i][0])
        if code == 65502:
            tone_timestamps.append(data['Data']['SerialDigitalIO']['TimeStamp'][i][0])

spike_times = np.load(os.path.join(ROOT_DIR, 'spike_times.npy'))
spike_clusters = np.load(os.path.join(ROOT_DIR, 'spike_clusters.npy'))

params_path = Path(ROOT_DIR) / 'params.py'

params = read_python(params_path)
print(params)  # This will show the variables in your params.py file
print(params['dtype'])  # Add this line to check dtype

model = load_model(params_path)

ds = {'prelight': {'spikes':  [[] for _ in range(10)]},
      'light': {'spikes': [[] for _ in range(10)]},
      'tone': {'spikes': [[] for _ in range(10)]}}

units = {49: deepcopy(ds) | {'electrode': 26},
         20: deepcopy(ds) | {'electrode': 19},
         4: deepcopy(ds) | {'electrode': 7}}

for i, clust in enumerate(spike_clusters):
    if clust in units:
        spike_time = spike_times[i][0]
        for j, time_stamp in enumerate(light_on_timestamps):
            tone_timestamp = tone_timestamps[j]
            distance_from_light_time_stamp = int(time_stamp) - int(spike_time)
            distance_from_tone_time_stamp = int(tone_timestamp) - int(spike_time)
            if 0 < distance_from_light_time_stamp <= 5*30000:
                units[clust]['prelight']['spikes'][j].append(int(spike_time))
            elif -5*30000 < distance_from_light_time_stamp <= 0:
                units[clust]['light']['spikes'][j].append(int(spike_time))
            elif -30 * 30000 < distance_from_tone_time_stamp <= 0:
                units[clust]['tone']['spikes'][j].append(int(spike_time))
            else:
                continue

for unit in units:
    units[unit]['waveform'] = get_mean_waveforms(model, unit, units[unit]['electrode'])
    for period_type in ['light', 'prelight']:
        units[unit][period_type]['rates'] = [len(period)/5 for period in units[unit][period_type]['spikes']]
    units[unit]['tone']['rates'] = [len(period)/30 for period in units[unit]['tone']['spikes']]

# Create the plot with the real data
fig, axs = plt.subplots(4, 4, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1, 1, 1]})

# Set background colors for each period
period_colors = {'prelight': 'white', 'light': 'lightgreen', 'tone': 'lightgreen'}
periods = ['prelight', 'light', 'tone']
units_list = list(units.keys())

# Plot waveforms above each unit
for i, unit in enumerate(units_list):
    axs[0, i].plot(units[unit]['waveform'], color='black')
    axs[0, i].set_title(f'Waveform: Unit {unit}')
    axs[0, i].axis('off')

# Plot scatter points and mean for each unit, with background colors
for i, unit in enumerate(units_list):
    for j, period in enumerate(periods):
        ax = axs[j+1, i]  # Access subplots corresponding to each unit/period
        rates = units[unit][period]['rates']
        ax.scatter(np.arange(len(rates)), rates, color='red', zorder=3)
        ax.axhline(np.mean(rates), color='blue', linestyle='--', lw=1.5)

        # Set the background color according to the period
        ax.set_facecolor(period_colors[period])
        ax.set_title(f'{period.capitalize()} Period')

        # Bell icon for tone period
        if period == 'tone':
            for k, y in enumerate(rates):
                ax.text(k, max(rates) + 0.5, '\uf0f3', fontsize=12, color='black', fontname='FontAwesome')

# Plot the summary plot on the right
all_rates = []
for period in periods:
    period_rates = [np.mean(units[unit][period]['rates']) for unit in units_list]
    all_rates.append(period_rates)

all_means = [np.mean(period_rates) for period_rates in all_rates]
for j, period in enumerate(periods):
    axs[j+1, 3].scatter(np.arange(len(units_list)), all_rates[j], color='red')
    axs[j+1, 3].axhline(all_means[j], color='blue', linestyle='--', lw=1.5)
    axs[j+1, 3].set_facecolor(period_colors[period])

# Layout adjustments
plt.tight_layout()
plt.show()
