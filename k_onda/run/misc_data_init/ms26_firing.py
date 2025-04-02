import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import os
import h5py
from copy import deepcopy
from phylib.io.model import load_model
import pandas as pd

# Updated ROOT_DIR
ROOT_DIR = r"C:\Users\Katie\data\MS_26_Optrode_Extinction_Learning"

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

# Load MAT file and extract timestamps
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
model = load_model(os.path.join(ROOT_DIR, 'params.py'))

# Set up units with initial structure and electrode info
ds = {'prelight': {'spikes': [[] for _ in range(10)]},
      'light': {'spikes': [[] for _ in range(10)]},
      'tone': {'spikes': [[] for _ in range(10)]}}

units = {49: deepcopy(ds) | {'electrode': 26},
         20: deepcopy(ds) | {'electrode': 19},
         4:  deepcopy(ds) | {'electrode': 7}}

# Assign spikes to periods based on timestamp differences
for i, clust in enumerate(spike_clusters):
    if clust in units:
        spike_time = spike_times[i][0]
        for j, time_stamp in enumerate(light_on_timestamps):
            tone_timestamp = tone_timestamps[j]
            distance_from_light_time_stamp = int(time_stamp) - int(spike_time)
            distance_from_tone_time_stamp = int(tone_timestamp) - int(spike_time)
            if 0 < distance_from_light_time_stamp <= 5 * 30000:
                units[clust]['prelight']['spikes'][j].append(int(spike_time))
            elif -5 * 30000 < distance_from_light_time_stamp <= 0:
                units[clust]['light']['spikes'][j].append(int(spike_time))
            elif -30 * 30000 < distance_from_tone_time_stamp <= 0:
                units[clust]['tone']['spikes'][j].append(int(spike_time))
            else:
                continue

# Compute waveform and firing rates per period for each unit
for unit in units:
    units[unit]['waveform'] = get_mean_waveforms(model, unit, units[unit]['electrode'])
    for period in ['prelight', 'light']:
        # Rate = number of spikes / 5 seconds for these periods
        units[unit][period]['rates'] = [len(trial) / 5 for trial in units[unit][period]['spikes']]
    # Tone period: number of spikes / 30 seconds
    units[unit]['tone']['rates'] = [len(trial) / 30 for trial in units[unit]['tone']['spikes']]

# -----------------------------------------------------------------------------
# Plotting Section (Waveforms, Scatter, and Summary Plots)
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rcParams

rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 9

fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(2, 4, height_ratios=[0.5, 2], width_ratios=[1, 1, 1, 1.5], wspace=0.6)

periods = ['prelight', 'light', 'tone']
x_positions = [0, 0.3, 0.6]
period_colors = ['white', 'lightgreen', 'lightgreen']
delta = 0.15
mean_line_width = delta * 0.6

# Plot waveforms (top row)
for i, unit in enumerate(units.keys()):
    ax_waveform = fig.add_subplot(gs[0, i])
    ax_waveform.plot(units[unit]['waveform'], color='black')
    ax_waveform.axis('off')
ax_empty = fig.add_subplot(gs[0, 3])
ax_empty.axis('off')

# Scatter plots for each unit (raw firing rates)
for i, unit in enumerate(units.keys()):
    ax = fig.add_subplot(gs[1, i])
    for j, period in enumerate(periods):
        rates = units[unit][period]['rates']
        ax.scatter(np.full_like(rates, x_positions[j]), rates, color='black', zorder=3)
        ax.hlines(np.mean(rates),
                  x_positions[j] - mean_line_width / 2,
                  x_positions[j] + mean_line_width / 2,
                  colors='blue', linestyles='--', lw=1.5)
    for j, color in enumerate(period_colors):
        ax.axvspan(x_positions[j] - delta, x_positions[j] + delta, color=color, zorder=0)
    for x in [x_positions[j] + delta for j in range(len(x_positions) - 1)]:
        ax.axvline(x, color='black', lw=1.5)
    ax.set_xlim(x_positions[0] - delta, x_positions[-1] + delta)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Prelight', 'Light', 'Tone'], fontsize=8)
    ax.set_ylabel('Spikes per second', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    for label in ax.get_yticklabels():
        label.set_fontsize(8)

# Summary plot (percent changes across periods)
summary_ax = fig.add_subplot(gs[1, 3])
for j, period in enumerate(periods):
    period_percent_changes = []
    for unit in units.keys():
        period_percent_changes.extend(
            [((rate - np.mean(units[unit]['prelight']['rates'])) / np.mean(units[unit]['prelight']['rates']) * 100)
             for rate in units[unit][period]['rates']]
        )
    summary_ax.scatter([x_positions[j]] * len(period_percent_changes), period_percent_changes, color='black', zorder=3)
    summary_ax.hlines(np.mean(period_percent_changes),
                      x_positions[j] - mean_line_width / 2,
                      x_positions[j] + mean_line_width / 2,
                      colors='blue', linestyles='--', lw=1.5)
for j, color in enumerate(period_colors):
    summary_ax.axvspan(x_positions[j] - delta, x_positions[j] + delta, color=color, zorder=0)
for x in [x_positions[j] + delta for j in range(len(x_positions) - 1)]:
    summary_ax.axvline(x, color='black', lw=1.5)
summary_ax.spines['top'].set_visible(False)
summary_ax.spines['right'].set_visible(False)
summary_ax.set_xticks(x_positions)
summary_ax.set_xticklabels(['Prelight', 'Light', 'Tone'], fontsize=8)
summary_ax.set_xlim(x_positions[0] - delta, x_positions[-1] + delta)
summary_ax.set_ylabel('Percent change in firing rate', fontsize=9)
summary_ax.tick_params(axis='both', which='major', labelsize=8)
summary_ax.set_title('Average Cell Firing', fontsize=11)
for label in summary_ax.get_yticklabels():
    label.set_fontsize(8)

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# New Statistical Analysis: One-Way RM ANOVA on Pooled (Trial-Level) Data
# -----------------------------------------------------------------------------
# Here we pool every trial value across units (ignoring unit differences)
# and assume that each unit's trial numbers are comparable.
# We create a subject identifier by combining unit and trial number.
data_list = []
for unit in units:
    n_trials = len(units[unit]['prelight']['rates'])  # Assuming same number of trials per period
    for trial in range(n_trials):
        # For each period, record the trial's firing rate.
        for period in ['prelight', 'light', 'tone']:
            data_list.append({
                'subject': f"{unit}_{trial+1}",
                'period': period,
                'rate': units[unit][period]['rates'][trial]
            })

data_rm = pd.DataFrame(data_list)
data_rm['subject'] = data_rm['subject'].astype('category')
data_rm['period'] = data_rm['period'].astype('category')
data_rm['period'] = data_rm['period'].cat.reorder_categories(['prelight', 'light', 'tone'], ordered=True)

# Run a repeated measures ANOVA using statsmodels AnovaRM.
from statsmodels.stats.anova import AnovaRM

aovrm = AnovaRM(data_rm, depvar='rate', subject='subject', within=['period'])
rm_results = aovrm.fit()
print("Repeated Measures ANOVA Results:")
print(rm_results)

# Perform a post-hoc Tukey HSD test on the pooled data.
# Note: Tukey HSD in statsmodels treats groups as independent, ignoring the repeated measures structure.
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(endog=data_rm['rate'], groups=data_rm['period'], alpha=0.05)
print("\nTukey HSD Post-hoc Test Results:")
print(tukey)

import pingouin as pg

# Assume 'data_rm' is the same DataFrame you used in AnovaRM,
# with columns: ['subject', 'period', 'rate'].

# Run the repeated-measures ANOVA (already shown in your code):
# aovrm = AnovaRM(data_rm, depvar='rate', subject='subject', within=['period'])
# rm_results = aovrm.fit()
# print(rm_results)

# Now do the pairwise repeated-measures tests using Pingouin
posthoc = pg.pairwise_ttests(
    dv='rate',
    within='period',
    subject='subject',
    data=data_rm,
    parametric=True,
    padjust='bonf'  # or 'holm', 'fdr_bh', etc.
)

print("\nPairwise repeated-measures comparisons (Pingouin):")
print(posthoc)

