#!/usr/bin/env python3
import os
import numpy as np
import subprocess
import scipy.signal as signal
from scipy.signal import coherence
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import scipy.stats as stats  # For the Wilcoxon signed-rank test
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


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
            {'bla': 2, 'vhip': 1, 'pl': 3} in MATLAB 
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
# New: Wilcoxon Test by Learning Category
# =============================================================================

def perform_wilcoxon_test_by_learning(animals_data, animal_info):
    """
    Performs a paired Wilcoxon signed-rank test comparing the evoked coherence 
    of tone_plus (CS+) and tone_minus (CS–) events within each learning category.
    
    For each learning category, tone_plus and tone_minus events are paired (by order)
    for each animal, and then combined across animals in that category.
    
    Returns a dictionary keyed by learning category with (stat, p) as values.
    """
    results = {}
    # Get unique learning categories
    learning_categories = set(info['learning'] for info in animal_info.values())
    for learning in learning_categories:
        paired_plus = []
        paired_minus = []
        # Iterate over animals in this learning category
        for animal, events in animals_data.items():
            if animal_info[animal]['learning'] == learning:
                plus_events = events.get('tone_plus', [])
                minus_events = events.get('tone_minus', [])
                n = min(len(plus_events), len(minus_events))
                for i in range(n):
                    paired_plus.append(plus_events[i]['evoked_coh'])
                    paired_minus.append(minus_events[i]['evoked_coh'])
        if paired_plus:
            paired_plus = np.array(paired_plus)
            paired_minus = np.array(paired_minus)
            stat, p = stats.wilcoxon(paired_plus, paired_minus)
            results[learning] = (stat, p)
        else:
            results[learning] = (None, None)
    return results

# =============================================================================
# Plotting function for evoked coherence (bar graph)
# =============================================================================

def plot_group_bar_graph(animals_data, animal_info, region_set, num_periods):
    """
    Plots a bar graph comparing evoked CS+ (tone_plus) and CS– (tone_minus)
    coherence across learning groups (collapsing over sexes), arranged in 
    three side-by-side subplots, similar to the power plot style.

    The figure title includes the region pair and the number of periods used.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 1) Aggregate data by group
    group_data = {}
    for animal, events in animals_data.items():
        learning = animal_info[animal]['learning']
        group = "misplaced fear" if learning == "bad_learner" else learning
        for cond in ['tone_plus', 'tone_minus']:
            if cond in events:
                for ev in events[cond]:
                    group_data.setdefault(group, {}).setdefault(cond, []).append(ev['evoked_coh'])
    
    # 2) Define learning styles and colors
    group_order = ['discriminator', 'generalizer', 'misplaced fear']
    colors = {
        'discriminator': '#0434ff',
        'generalizer': '#797977',
        'misplaced fear': '#fffc00'
    }
    
    # 3) Compute means and SEMs for each group
    means_plus, sems_plus = {}, {}
    means_minus, sems_minus = {}, {}
    
    for grp in group_order:
        plus_data = np.array(group_data.get(grp, {}).get('tone_plus', []))
        minus_data = np.array(group_data.get(grp, {}).get('tone_minus', []))
        
        if plus_data.size > 0:
            means_plus[grp] = np.mean(plus_data)
            sems_plus[grp] = np.std(plus_data) / np.sqrt(len(plus_data))
        else:
            means_plus[grp] = 0
            sems_plus[grp] = 0
        
        if minus_data.size > 0:
            means_minus[grp] = np.mean(minus_data)
            sems_minus[grp] = np.std(minus_data) / np.sqrt(len(minus_data))
        else:
            means_minus[grp] = 0
            sems_minus[grp] = 0

    # 4) Create a figure with 1 row and 3 columns, share the y-axis
    fig, axs = plt.subplots(1, 3, figsize=(8, 4), sharey=True)
    
    # We'll track global min/max to unify y-limits across subplots
    global_min = np.inf
    global_max = -np.inf
    
    # 5) For each group (left to right), plot 2 bars: CS+ and CS–
    for i, grp in enumerate(group_order):
        ax = axs[i]
        
        # If the group doesn't exist, just label it "n/a"
        if grp not in group_data:
            ax.set_title(f"{grp}\n(n/a)")
            ax.axis('off')
            continue
        
        # Extract stats
        plus_mean = means_plus[grp]
        plus_sem  = sems_plus[grp]
        minus_mean = means_minus[grp]
        minus_sem  = sems_minus[grp]
        
        # Update global min/max
        local_vals = [
            plus_mean - plus_sem,
            plus_mean + plus_sem,
            minus_mean - minus_sem,
            minus_mean + minus_sem
        ]
        local_min, local_max = min(local_vals), max(local_vals)
        if local_min < global_min:
            global_min = local_min
        if local_max > global_max:
            global_max = local_max
        
        # We'll place two bars side by side at x=[0.25, 0.75], 
        # with a narrower axis range [0,1]
        x_positions = [0.25, 0.75]
        bar_width = 0.4
        
        # Plot bars
        bars = ax.bar(
            x_positions,
            [plus_mean, minus_mean],
            yerr=[plus_sem, minus_sem],
            capsize=5,
            width=bar_width,
            color=colors[grp]
        )
        
        # Apply a hatch for the CS+ bar (the first bar)
        bars[0].set_hatch('//')
        
        # X-axis formatting
        ax.set_xticks(x_positions)
        ax.set_xticklabels(['CS+', 'CS–'])
        ax.set_xlim(0, 1)
        
        # Title for each subplot is the group name
        ax.set_title(grp)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Only label the y-axis on the first subplot
        if i == 0:
            ax.set_ylabel("Evoked Coherence")
    
    # 6) Unify the y-limits across all subplots
    if global_max > -np.inf and global_min < np.inf:
        margin = (global_max - global_min) * 0.1 if global_max != global_min else 1
        y_lower = global_min - margin
        y_upper = global_max + margin
        for ax in axs:
            ax.set_ylim(y_lower, y_upper)
    
    # 7) Set a suptitle for the figure
    fig.suptitle(
        f"Evoked Coherence by Learning (Combined Sex)\n(Regions: {region_set[0]} vs. {region_set[1]}, {num_periods} periods)",
        fontsize=14,
        x=0.5
    )
    
    # 8) Adjust layout so the suptitle doesn't span the entire figure width
    plt.tight_layout(rect=[0.25, 0.05, 0.75, 0.90])
    plt.show()

# =============================================================================
# Data Gathering Function
# =============================================================================

def gather_data(region_set, num_periods):
    """
    Gathers and processes data for a given region set, using only the first 
    'num_periods' events per condition (CS+ and CS–) for each animal.
    
    The region_set parameter is a tuple (e.g., ('pl','bla')).
    """
    main_dir = '/Users/katie/likhtik/AS'
    valid_animals = ['As105','As106','As107','As108','As110','As111','As112','As113']
    matlab_path = '/Applications/MATLAB_R2022a.app/bin/matlab'
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
    coh_low = 4
    coh_high = 8
    cs_minus_ids = [0, 2, 4, 5, 8, 11]

    animals_data = {}
    
    for animal in valid_animals:
        animal_path = os.path.join(main_dir, animal)
        if not os.path.isdir(animal_path):
            continue
        print(f"Processing animal: {animal} for region set {region_set}")
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
        NS3_data = NS3.Data
        
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
        data_by_condition = {'tone_plus': [], 'tone_minus': [], 'pretone': []}
        
        # Counters to limit events per condition
        count_plus = 0
        count_minus = 0
        
        for period_idx, tone_on in enumerate(tone_period_times):
            # Determine tone condition from fixed IDs
            if period_idx in cs_minus_ids:
                tone_condition = 'tone_minus'
            else:
                tone_condition = 'tone_plus'
            
            # Skip if we've already reached the limit for this condition.
            if tone_condition == 'tone_plus' and count_plus >= num_periods:
                continue
            if tone_condition == 'tone_minus' and count_minus >= num_periods:
                continue
            
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
            
            tone_event = {
                'period_index': period_idx,
                'tone_coh': tone_coh,
                'animal': animal,
                'learning': learning_dict[animal],
                'sex': ('male' if animal in male_stressed or animal in male_non_stressed else 'female')
            }
            tone_events.append((tone_condition, tone_event))
            pretone_event = {
                'period_index': period_idx,
                'pretone_coh': pretone_coh,
                'animal': animal
            }
            data_by_condition['pretone'].append(pretone_event)
            
            if tone_condition == 'tone_plus':
                count_plus += 1
            else:
                count_minus += 1
            
            if count_plus >= num_periods and count_minus >= num_periods:
                break
        
        if len(pretone_coh_values) == 0:
            print(f"No pretone data for {animal}. Skipping animal.")
            continue
        
        animal_pretone_mean = np.mean(pretone_coh_values)
        
        for tone_condition, event in tone_events:
            event['evoked_coh'] = event['tone_coh'] - animal_pretone_mean
            data_by_condition[tone_condition].append(event)
        
        animals_data[animal] = data_by_condition
    
    return animals_data, animal_info, learning_dict, fs, cs_minus_ids, coh_low, coh_high


def anova_tone_learning(animals_data, animal_info):
    """
    Performs a two-way ANOVA on evoked coherence values with factors:
      - tone_type: tone_plus vs. tone_minus
      - learning: the learning style as given in animal_info

    Uses individual period observations (averaged over the frequency band) as the dependent variable,
    ignoring the clustering within animals.

    Returns:
        anova_table (DataFrame): Full ANOVA table.
        df (DataFrame): The DataFrame of observations used for the analysis.
    """
    rows = []
    for animal, data in animals_data.items():
        learning = animal_info[animal]['learning']
        for tone in ['tone_plus', 'tone_minus']:
            for event in data.get(tone, []):
                rows.append({
                    'evoked_coh': event['evoked_coh'],
                    'tone_type': tone,
                    'learning': learning
                })
    df = pd.DataFrame(rows)
    # Fit the two-way ANOVA model with interaction using statsmodels
    model = smf.ols('evoked_coh ~ C(tone_type) * C(learning)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return model, anova_table, df

import numpy as np
import scipy.stats as stats

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def wilcoxon_followup_test(animals_data, animal_info, by_tone_type=False):
    """
    Performs follow-up tests on evoked coherence values with two components:
      1. A non-parametric Mann–Whitney U test comparing:
           - Group 1: Pooled CS+ and CS– events from Discriminators and Generalizers (combined)
           - Group 2: Pooled CS+ and CS– events from Bad Learners.
      2. A Tukey HSD test comparing all three learning groups (discriminator, generalizer, bad_learner)
         pairwise.
    
    Parameters:
        animals_data (dict): Data for each animal with keys 'tone_plus' and 'tone_minus',
                             where each event has an 'evoked_coh' value.
        animal_info (dict): Dictionary with animal learning style information.
        by_tone_type (bool): 
            If False (default), pools CS+ and CS– values together.
            If True, runs the tests separately for CS+ and CS– events.
    
    Returns:
        If by_tone_type is False, returns a dictionary with key 'pooled' containing:
            - 'mann_whitney': Dictionary with Mann–Whitney U test results (U statistic, p-value, sample sizes).
            - 'data': The pooled data used for the Mann–Whitney test (lists for group1 and group2).
            - 'tukey': A dictionary with:
                - 'summary': The Tukey HSD test summary as a string.
                - 'df': A DataFrame of observations used for the Tukey test (columns: evoked_coh, learning).
                
        If by_tone_type is True, returns a dictionary with keys 'tone_plus' and 'tone_minus', where
        each value is a dictionary structured as above.
    """
    if not by_tone_type:
        # For Mann–Whitney, combine discriminators and generalizers vs. bad learners
        group1 = []  # discriminators and generalizers
        group2 = []  # bad learners
        # For Tukey, we want to keep all three groups separate.
        tukey_rows = []
        
        for animal, data in animals_data.items():
            learning = animal_info[animal]['learning']
            for tone in ['tone_plus', 'tone_minus']:
                if tone in data:
                    for event in data[tone]:
                        val = event['evoked_coh']
                        tukey_rows.append({'evoked_coh': val, 'learning': learning})
                        if learning in ['discriminator', 'generalizer']:
                            group1.append(val)
                        elif learning == 'bad_learner':
                            group2.append(val)
        
        if len(group1) == 0 or len(group2) == 0:
            raise ValueError("One of the groups has no data for Mann–Whitney testing.")
        
        stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        mann_whitney_result = {
            'U_statistic': stat,
            'p_value': p_value,
            'n_group1': len(group1),
            'n_group2': len(group2)
        }
        
        df_tukey = pd.DataFrame(tukey_rows)
        tukey_result = pairwise_tukeyhsd(endog=df_tukey['evoked_coh'], groups=df_tukey['learning'], alpha=0.05)
        tukey_summary_str = tukey_result.summary().as_text()
        
        result = {
            'pooled': {
                'mann_whitney': mann_whitney_result,
                'data': {'group1': group1, 'group2': group2},
                'tukey': {'summary': tukey_summary_str, 'df': df_tukey}
            }
        }
        return result
    
    else:
        # Separate the analysis for each tone type.
        results_by_tone = {}
        for tone in ['tone_plus', 'tone_minus']:
            group1 = []  # discriminators and generalizers
            group2 = []  # bad learners
            tukey_rows = []
            
            for animal, data in animals_data.items():
                learning = animal_info[animal]['learning']
                if tone in data:
                    for event in data[tone]:
                        val = event['evoked_coh']
                        tukey_rows.append({'evoked_coh': val, 'learning': learning})
                        if learning in ['discriminator', 'generalizer']:
                            group1.append(val)
                        elif learning == 'bad_learner':
                            group2.append(val)
            
            if len(group1) == 0 or len(group2) == 0:
                raise ValueError(f"One of the groups has no data for Mann–Whitney testing for tone {tone}.")
            
            stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            mann_whitney_result = {
                'U_statistic': stat,
                'p_value': p_value,
                'n_group1': len(group1),
                'n_group2': len(group2)
            }
            
            df_tukey = pd.DataFrame(tukey_rows)
            tukey_result = pairwise_tukeyhsd(endog=df_tukey['evoked_coh'], groups=df_tukey['learning'], alpha=0.05)
            tukey_summary_str = tukey_result.summary().as_text()
            
            results_by_tone[tone] = {
                'mann_whitney': mann_whitney_result,
                'data': {'group1': group1, 'group2': group2},
                'tukey': {'summary': tukey_summary_str, 'df': df_tukey}
            }
        return results_by_tone
    
# =============================================================================
# Main Function: Call gather_data for different region sets and period limits
# =============================================================================

def main():
    # Define the three region set combinations.
    region_sets = [
        ('pl', 'bla'),
        ('pl', 'vhip'),
        ('bla', 'vhip')
    ]
    # Specify how many periods (per condition) to use.
    num_periods = 6
    
    for region_set in region_sets:
        print(f"\n=== Processing region set: {region_set} with {num_periods} periods per condition ===")
        animals_data, animal_info, learning_dict, fs, cs_minus_ids, coh_low, coh_high = gather_data(region_set, num_periods)
        
        # Perform paired Wilcoxon signed-rank test within each learning category.
        wilcoxon_results = perform_wilcoxon_test_by_learning(animals_data, animal_info)
        print(f"Paired Wilcoxon signed-rank tests for region set {region_set}:")
        for learning, (stat, p) in wilcoxon_results.items():
            if stat is not None:
                print(f"Learning '{learning}': Wilcoxon statistic: {stat:.0f}, p-value: {p:.4f}")
            else:
                print(f"Learning '{learning}': Insufficient data for Wilcoxon test.")
        
        # Compute group-level statistics.
        group_animals = {}
        for animal, events in animals_data.items():
            info = animal_info[animal]
            group_key = f"{info['sex']}_{info['learning']}"
            group_animals.setdefault(group_key, []).append(events)
        
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
                for p_idx in sorted_periods:
                    values = np.array(period_dict[p_idx])
                    means.append(np.mean(values))
                    sems.append(np.std(values) / np.sqrt(len(values)))
                group_means[group_key][cond] = {
                    'periods': np.array(sorted_periods),
                    'mean_coh': np.array(means),
                    'sem_coh': np.array(sems)
                }
        
        model, anova_table, df = anova_tone_learning(animals_data, animal_info)
        print(region_set)
        print(anova_table)
        print(df.head)
        print("")

        tukey_result = pairwise_tukeyhsd(endog=df['evoked_coh'], groups=df['learning'], alpha=0.05)
        print(tukey_result)

        print(wilcoxon_followup_test(animals_data, animal_info))
        print(wilcoxon_followup_test(animals_data, animal_info, by_tone_type=True))
        
        # Plot the results.
        plot_group_bar_graph(animals_data, animal_info, region_set, num_periods)
        


if __name__ == "__main__":
    main()