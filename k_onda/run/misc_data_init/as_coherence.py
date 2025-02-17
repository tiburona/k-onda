#!/usr/bin/env python3
import os
import numpy as np
import subprocess
import tempfile
import scipy.signal as signal
from scipy.signal import coherence
import scipy.io
import matplotlib.pyplot as plt

# =============================================================================
# Helper functions
# =============================================================================

def divide_by_rms(arr):
    """Divide the array by its RMS value."""
    rms = np.sqrt(np.mean(arr ** 2))
    return arr / rms

def filter_60_hz(signal_with_noise, fs):
    """Apply a 60-Hz notch filter to the signal."""
    f0 = 60  # Frequency to remove
    Q = 30   # Quality factor controls notch width
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, signal_with_noise)

def get_electrode_mapping(animal):
    """
    Returns a dictionary with the electrode mapping (0-indexed).
    
    Mapping definitions:
      - General (for animals other than As110 and As113):
            {'bla': 2, 'vhip': 1, 'pl': 3}  in MATLAB 
         becomes {'bla': 1, 'vhip': 0, 'pl': 2} in Python.
      - As110:
            {'vhip': 1, 'pl': 4, 'bla': 2} in MATLAB 
         becomes {'vhip': 0, 'pl': 3, 'bla': 1} in Python.
      - As113:
            {'bla': 3, 'vhip': 1, 'pl': 4} in MATLAB 
         becomes {'bla': 2, 'vhip': 0, 'pl': 3} in Python.
    """
    if animal == 'As110':
        mapping = {'vhip': 0, 'pl': 3, 'bla': 1}
    elif animal == 'As113':
        mapping = {'bla': 2, 'vhip': 0, 'pl': 3}
    else:
        mapping = {'bla': 1, 'vhip': 0, 'pl': 2}
    return mapping

def run_matlab_load(ns3_file, nev_file, temp_mat_file, matlab_path):
    """
    Writes a temporary MATLAB script that:
      - Adds '/Users/katie/likhtik/software' (and its subfolders) to the MATLAB path.
      - Loads NS3 and NEV using openNSx and openNEV.
      - Saves NS3 and NEV to a MAT-file.
    """
    matlab_script = f"""
addpath(genpath('/Users/katie/likhtik/software'));
try
    NS3 = openNSx('{ns3_file}');
    NEV = openNEV('{nev_file}');
    save('{temp_mat_file}', 'NS3', 'NEV');
catch ME
    disp(getReport(ME));
end
exit;
"""
    script_filename = "temp_load_script.m"
    with open(script_filename, "w") as f:
        f.write(matlab_script)
    
    cmd = [matlab_path, "-nodisplay", "-nosplash", "-r", f"run('{os.path.abspath(script_filename)}')"]
    subprocess.run(cmd, check=True)
    os.remove(script_filename)

def calc_coherence(data_1, data_2, sampling_rate, low=4, high=8):
    """
    Calculate the coherence between data_1 and data_2, then restrict and return 
    the coherence values within the [low, high] frequency band.
    """
    nperseg = 2000  
    noverlap = round(nperseg/2)
    window = 'hann'
    f, Cxy = coherence(data_1, data_2, fs=sampling_rate, window=window, nperseg=nperseg, 
                       noverlap=noverlap)
    mask = (f >= low) & (f <= high)
    Cxy_band = Cxy[mask]
    return Cxy_band

# =============================================================================
# Plotting function for evoked coherence
# =============================================================================

def plot_evoked_coherence(group_means, region_set):
    """
    Plots evoked coherence as line plots with SEM error bars.
    
    Six subplots (arranged as 3 rows for learning styles x 2 columns for sex).
    For each group (e.g., "male_discriminator"), two lines are plotted:
      - tone_plus (CS+)
      - tone_minus (CS–)
    
    X-axis: period number (only labeled on the bottom row)
    Y-axis: evoked coherence
    
    Parameters:
      group_means: dictionary with keys like "male_discriminator", "female_generalizer", etc.
                   Each group should contain a dict for "tone_plus" and "tone_minus" with keys:
                     - 'periods': numpy array of event indices
                     - 'mean_coh': numpy array of mean evoked coherence values
                     - 'sem_coh': numpy array of SEM values
      region_set: tuple of two brain regions (e.g., ('vhip','pl')) used in the analysis.
    """
    learning_styles = ['discriminator', 'generalizer', 'bad_learner']
    sexes = ['male', 'female']
    
    fig, axs = plt.subplots(nrows=len(learning_styles), ncols=len(sexes), figsize=(14, 10),
                            sharex=True, sharey=True)
    
    for i, learning in enumerate(learning_styles):
        for j, sex in enumerate(sexes):
            ax = axs[i, j]
            group_key = f"{sex}_{learning}"
            if group_key not in group_means:
                ax.set_title(f"{group_key.capitalize()} (n/a)")
                continue
            for cond, label, color in zip(['tone_plus', 'tone_minus'],
                                          ['CS+', 'CS-'],
                                          ['red', 'blue']):
                if cond not in group_means[group_key]:
                    continue
                data = group_means[group_key][cond]
                periods = data['periods']
                mean_coh = data['mean_coh']
                sem_coh = data['sem_coh']
                ax.errorbar(periods, mean_coh, yerr=sem_coh, marker='o', color=color, label=label)
            ax.set_title(f"{group_key.capitalize()}")
            if i == len(learning_styles) - 1:
                ax.set_xlabel("Period Number")
            if j == 0:
                ax.set_ylabel("Evoked Coherence")
            ax.legend()
    
    fig.suptitle(f"Evoked Coherence (Regions: {region_set[0]} vs. {region_set[1]})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =============================================================================
# Main processing function: compute coherence for events and plot evoked coherence
# =============================================================================

def main():
    # Parameters and directories.
    main_dir = '/Users/katie/likhtik/AS'
    valid_animals = ['As105','As106','As107','As108','As110','As111','As112','As113']
    matlab_path = '/Applications/MATLAB_R2022a.app/bin/matlab'
    region_set = ('vhip', 'bla')
    
    # Learning categorization.
    learning_dict = {
        'As105': 'discriminator',
        'As106': 'bad_learner',
        'As107': 'generalizer',
        'As108': 'generalizer',
        'As110': 'bad_learner',
        'As111': 'generalizer',
        'As112': 'generalizer',
        'As113': 'generalizer'
    }
    # Sex/stress groups.
    male_stressed     = ['As107', 'As108']
    male_non_stressed = ['As105', 'As106']
    female_stressed   = ['As112', 'As113']
    female_non_stressed = ['As110', 'As111']

    animal_info = {
        'As105': {'sex': 'male', 'learning': 'discriminator'},
        'As106': {'sex': 'male', 'learning': 'bad_learner'},
        'As107': {'sex': 'male', 'learning': 'generalizer'},
        'As108': {'sex': 'male', 'learning': 'generalizer'},
        'As110': {'sex': 'female', 'learning': 'bad_learner'},
        'As111': {'sex': 'female', 'learning': 'generalizer'},
        'As112': {'sex': 'female', 'learning': 'generalizer'},
        'As113': {'sex': 'female', 'learning': 'generalizer'},
    }
    
    tone_on_code = 65503
    expectedToneEvents = 12  # per animal
    fs = 2000  # sampling rate (Hz)
    # Coherence parameters (theta band by default, can be changed)
    coh_low = 4
    coh_high = 8
    cs_minus_ids = [0, 2, 4, 5, 8, 11]
    
    # Data structure to store event-level coherence.
    animals_data = {}
    
    for animal in valid_animals:
        data_by_condition = {'tone_plus': [], 'tone_minus': [], 'pretone': []}
        
        animal_path = os.path.join(main_dir, animal)
        if not os.path.isdir(animal_path):
            continue
        print(f"Processing animal: {animal}")
        
        ns3_file = os.path.join(animal_path, "Testing.ns3")
        nev_file = os.path.join(animal_path, "Testing.nev")
        temp_mat_file = os.path.abspath(f"tempdata_{animal}.mat")
        
        try:
            run_matlab_load(ns3_file, nev_file, temp_mat_file, matlab_path)
        except Exception as e:
            print(f"Error loading MATLAB files for {animal}: {e}")
            continue
        
        try:
            mat_data = scipy.io.loadmat(temp_mat_file, struct_as_record=False, squeeze_me=True)
            os.remove(temp_mat_file)
        except Exception as e:
            print(f"Error loading MAT file for {animal}: {e}")
            continue
        
        NS3 = mat_data['NS3']
        NEV = mat_data['NEV']
        NS3_data = NS3.Data  # assume shape: (electrodes x samples)
        
        digital_io = NEV.Data.SerialDigitalIO
        tone_timestamps = np.atleast_1d(digital_io.TimeStampSec)
        codes = np.atleast_1d(digital_io.UnparsedData)
        
        tone_indices = np.where(codes == tone_on_code)[0]
        if len(tone_indices) != expectedToneEvents:
            print(f"Warning: {animal} has {len(tone_indices)} tone events (expected {expectedToneEvents}).")
        tone_period_times = tone_timestamps[tone_indices]
        
        mapping = get_electrode_mapping(animal)
        ch1 = mapping.get(region_set[0])
        ch2 = mapping.get(region_set[1])
        if ch1 is None or ch2 is None:
            print(f"Error: Could not find mapping for regions in {animal}.")
            continue
        
        pretone_coh_values = []
        tone_events = []
        
        for period_idx, tone_on in enumerate(tone_period_times):
            tone_seg_start = tone_on - 1
            tone_seg_end   = tone_on + 31
            pretone_seg_start = tone_on - 32
            pretone_seg_end   = tone_on
            
            tone_idx_start = int(round(tone_seg_start * fs))
            tone_idx_end   = int(round(tone_seg_end * fs))
            pretone_idx_start = int(round(pretone_seg_start * fs))
            pretone_idx_end   = int(round(pretone_seg_end * fs))
            
            try:
                tone_data1 = NS3_data[ch1, tone_idx_start:tone_idx_end]
                tone_data2 = NS3_data[ch2, tone_idx_start:tone_idx_end]
                pretone_data1 = NS3_data[ch1, pretone_idx_start:pretone_idx_end]
                pretone_data2 = NS3_data[ch2, pretone_idx_start:pretone_idx_end]
            except Exception as e:
                print(f"Error extracting data for {animal} event {period_idx}: {e}")
                continue
            
            tone_data1 = divide_by_rms(filter_60_hz(tone_data1, fs))
            tone_data2 = divide_by_rms(filter_60_hz(tone_data2, fs))
            pretone_data1 = divide_by_rms(filter_60_hz(pretone_data1, fs))
            pretone_data2 = divide_by_rms(filter_60_hz(pretone_data2, fs))
            
            tone_coh_band = calc_coherence(tone_data1, tone_data2, fs, low=coh_low, high=coh_high)
            pretone_coh_band = calc_coherence(pretone_data1, pretone_data2, fs, low=coh_low, high=coh_high)
            
            tone_coh = np.mean(tone_coh_band)
            pretone_coh = np.mean(pretone_coh_band)
            
            pretone_coh_values.append(pretone_coh)
            
            if period_idx in cs_minus_ids:
                tone_condition = 'tone_minus'
            else:
                tone_condition = 'tone_plus'
            
            tone_event = {
                'period_index': period_idx,
                'tone_coh': tone_coh,
                'animal': animal,
                'learning': learning_dict[animal],
                'sex': ('male' if animal in male_stressed or animal in male_non_stressed 
                        else 'female')
            }
            tone_events.append((tone_condition, tone_event))
            
            pretone_event = {
                'period_index': period_idx,
                'pretone_coh': pretone_coh,
                'animal': animal
            }
            data_by_condition['pretone'].append(pretone_event)
        
        if len(pretone_coh_values) == 0:
            print(f"No pretone data for {animal}. Skipping animal.")
            continue
        
        animal_pretone_mean = np.mean(pretone_coh_values)
        
        for tone_condition, event in tone_events:
            event['evoked_coh'] = event['tone_coh'] - animal_pretone_mean
            data_by_condition[tone_condition].append(event)
        
        animals_data[animal] = data_by_condition
    
    # =============================================================================
    # Compute group-level statistics.
    # For each group and for each tone condition, compute the mean and SEM of evoked coherence.
    # =============================================================================
    group_animals = {}
    for animal, events in animals_data.items():
        info = animal_info[animal]
        group_key = f"{info['sex']}_{info['learning']}"
        if group_key not in group_animals:
            group_animals[group_key] = []
        group_animals[group_key].append(events)
    
    group_means = {}
    for group_key, animals_events in group_animals.items():
        group_means[group_key] = {}
        for cond in ['tone_plus', 'tone_minus']:
            period_dict = {}
            for animal_ev in animals_events:
                for event in animal_ev.get(cond, []):
                    idx = event['period_index']
                    period_dict.setdefault(idx, []).append(event['evoked_coh'])
            if not period_dict:
                continue
            sorted_periods = sorted(period_dict.keys())
            means = []
            sems = []
            for p in sorted_periods:
                values = np.array(period_dict[p])
                means.append(np.mean(values))
                sems.append(np.std(values) / np.sqrt(len(values)))
            group_means[group_key][cond] = {
                'periods': np.array(sorted_periods),
                'mean_coh': np.array(means),
                'sem_coh': np.array(sems)
            }
    
    plot_evoked_coherence(group_means, region_set)

if __name__ == "__main__":
    main()