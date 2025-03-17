#!/usr/bin/env python3
import os
import numpy as np
import subprocess
import tempfile
import scipy.signal as signal
import scipy.io
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon  # for Wilcoxon tests

# =============================================================================
# Helper functions for filtering, MATLAB interfacing, etc.
# =============================================================================

CURRENT_ANIMAL = ''
CURRENT_IDX = ''

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
            MATLAB: {'bla': 2, 'vhip': 1, 'pl': 3} 
         becomes Python: {'bla': 1, 'vhip': 0, 'pl': 2}.
      - As110:
            MATLAB: {'vhip': 1, 'pl': 4, 'bla': 2} 
         becomes Python: {'vhip': 0, 'pl': 3, 'bla': 1}.
      - As113:
            MATLAB: {'bla': 3, 'vhip': 1, 'pl': 4} 
         becomes Python: {'bla': 2, 'vhip': 0, 'pl': 3}.
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
    with open(script_filename, "w") as g:
        g.write(matlab_script)
    
    cmd = [matlab_path, "-nodisplay", "-nosplash", "-r", f"run('{os.path.abspath(script_filename)}')"]
    subprocess.run(cmd, check=True)
    os.remove(script_filename)

def run_matlab_mtcsg(data, matlab_path, brain_region,
                     animal, period_number, period_type, filter_first, rms, params=('2048', '2000', '1000', '980', '2'),
                     output_dir='/Users/katie/likhtik/data/temp'):
    """
    Calls MATLAB's mtcsg on the provided data.
    
    Instead of writing temporary output files, this function writes S, f, and t to permanent files
    whose names encode the animal name, period number, period type, and the parameters with which
    mtcsg is called. If these files already exist, they are loaded directly rather than re-running
    mtcsg.
    
    Parameters:
      data: 1D numpy array.
      matlab_path: path to the MATLAB executable.
      animal: string identifier for the animal.
      period_number: period number (can be numeric or string).
      period_type: a string describing the period type.
      params: tuple of parameters for mtcsg (defaults match the original call).
      output_dir: directory where S, f, and t are saved.
    
    Returns:
      S, f, t as numpy arrays.
    """
    print(f"mtcsg params {params}")

    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = f"{brain_region}_{animal}_period{period_number}_{period_type}_mtcsg_filt_1st_{filter_first}_rms_{rms}_{'_'.join(params)}"
    S_file_name = os.path.join(output_dir, base_filename + '_S.txt')
    f_file_name = os.path.join(output_dir, base_filename + '_f.txt')
    t_file_name = os.path.join(output_dir, base_filename + '_t.txt')
    
    if os.path.exists(S_file_name) and os.path.exists(f_file_name) and os.path.exists(t_file_name):
        S = np.loadtxt(S_file_name)
        f = np.loadtxt(f_file_name)
        t = np.loadtxt(t_file_name)
        return S, f, t
    
    seg_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    seg_file_name = seg_file.name
    np.savetxt(seg_file_name, data, fmt='%.6f')
    seg_file.close()
    
    matlab_script = f"""
addpath(genpath('/Users/katie/likhtik/software'));
data = load('{seg_file_name}');
[S, f, t] = mtcsg(data, {', '.join(params)});
save('-ascii','{S_file_name}','S');
save('-ascii','{f_file_name}','f');
save('-ascii','{t_file_name}','t');
exit;
"""
    script_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.m')
    script_file_name = script_file.name
    script_file.write(matlab_script)
    script_file.close()
    
    cmd = [matlab_path, "-nodisplay", "-nosplash", "-r", f"run('{os.path.abspath(script_file_name)}')"]
    subprocess.run(cmd, check=True)
    
    S = np.loadtxt(S_file_name)
    f = np.loadtxt(f_file_name)
    t = np.loadtxt(t_file_name)
    
    # Optionally, remove temporary files.
    # os.remove(seg_file_name)
    # os.remove(script_file_name)
    
    return S, f, t

# =============================================================================
# Process one event's mtcsg output into an averaged pip spectrogram
# =============================================================================

def process_event_segment(S, f, t, freq_band, freq_range, event_boundary):
    """
    Processes one event's spectrogram as gollows:
      1. For each pip (with onsets defined as 0.75, 1.75, ..., 29.75 seconds),
         extract the segment of the spectrogram corresponding to the time window 
         [pip_onset + event_boundary[0], pip_onset + event_boundary[1]].  
         (For example, if event_boundary is (-0.1, 0.3), the window for the first pip 
         is [0.65, 1.05] seconds.)
      2. Average the extracted pip segments to yield an event spectrogram.
      3. Restrict the frequency axis to the specified freq_range.
      4. Compute a bar value by averaging the event spectrogram over the restricted frequency range.
    
    Parameters:
      S: 2D array (frequency x time) from mtcsg.
      f: 1D frequency vector.
      t: 1D time vector.
      freq_range: tuple (f_low, f_high) defining the frequency range to retain.
      event_boundary: tuple (t_offset_start, t_offset_end) relative to each pip onset.
                     (If a single number is passed, it is interpreted as (0, value).)
    
    Returns a tuple: (event_spectrogram, rel_time, f_restricted, bar_value)
      - event_spectrogram: 2D array (n_freq x n_time) averaged over all pips.
      - rel_time: 1D array for the relative time axis (spanning event_boundary).
      - f_restricted: 1D array of frequencies (within freq_range).
      - bar_value: scalar average power over the restricted frequency range.
    """
    # Ensure event_boundary is a tuple
    if not isinstance(event_boundary, (tuple, list)):
        event_boundary = (0, event_boundary)
    
    # Define pip onsets (in seconds).
    if freq_band in ['theta', 'low frequencies']:
        pip_onsets = np.arange(0.75, 30.75, 1)  # 0.75, 1.75, ..., 29.75 s
    else:
        pip_onsets = np.arange(1.0, 31.0, 1)
    
    pip_slices = []
    rel_time_common = None  # store the relative time axis from the first valid pip
    
    for onset in pip_onsets:
        # Extract S for times between (onset + event_boundary[0]) and (onset + event_boundary[1]).
        win_inds = np.where((t >= onset + event_boundary[0]) & (t <= onset + event_boundary[1]))[0]
        if win_inds.size == 0:
            continue  # skip if no data in this window
        S_pip = S[:, win_inds]
        if CURRENT_ANIMAL == 'As107' and CURRENT_IDX == 1 and onset == 0.75:
            with open('/Users/katie/likhtik/data/temp/test_as_log.txt', 'a') as g:
                g.write(f"first pip vals: {S_pip[0, 0:10]}, win_inds: {win_inds}\n")
        t_pip = t[win_inds]
        # Define relative time axis (relative to current pip onset)
        rel_time = t_pip - onset
        if rel_time_common is None:
            rel_time_common = rel_time  # assume all pips have the same grid
        pip_slices.append(S_pip)
    
    if len(pip_slices) == 0:
        raise ValueError("No pip slices extracted for event.")
    
    # Average across pips.
    pip_stack = np.stack(pip_slices, axis=0)  # shape: (n_pips, n_freq, n_time)
    event_S = np.mean(pip_stack, axis=0)         # shape: (n_freq, n_time)
    if CURRENT_ANIMAL == 'As107' and CURRENT_IDX == 1:
        with open('/Users/katie/likhtik/data/temp/test_as_log.txt', 'a') as g:
            g.write(f"equivalent of get_power for the period: {event_S[0][0:10]}\n")
    # Restrict the frequency axis.
    freq_inds = np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]
    event_S_restricted = event_S[freq_inds, :]
    f_restricted = f[freq_inds]
    
    # Compute bar value as the average power over the restricted frequency range.
    bar_value = np.mean(event_S_restricted)
    
    return event_S_restricted, rel_time_common, f_restricted, bar_value

def freq_band_to_mtcsg_args(freq_band):
    if freq_band in ['theta', 'low frequencies']:
        return ('2048', '2000', '1000', '980', '2')
    else:
        return ('2048', '2000', '500', '480', '2')

# =============================================================================
# Plotting functions (original, splitting by sex and learning)
# =============================================================================


def plot_evoked_heatmaps(group_means, brain_region, freq_band, title_suffix=''):
    """
    Plots evoked heat maps for 12 conditions arranged as gollows:
      - Rows correspond to learning styles:
           Row 1: Discriminators
           Row 2: Generalizers
           Row 3: Bad Learners
      - Within each row, the four columns are:
           Column 1: Male CS+ (tone_plus)
           Column 2: Male CS– (tone_minus)
           Column 3: Female CS+ (tone_plus)
           Column 4: Female CS– (tone_minus)
    
    [See original docstring for more details…]
    """
    learning_styles = ['discriminator', 'generalizer', 'misplaced fear']
    sexes = ['male', 'female']
    stimulus_types = ['tone_plus', 'tone_minus']
    
    nrows = len(learning_styles)
    ncols = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 12))
    
    pair_scales = {}
    for learning in learning_styles:
        for sex in sexes:
            group_key = f"{sex}_{learning}"
            pair_vmin = np.inf
            pair_vmax = -np.inf
            for stim in stimulus_types:
                if group_key in group_means and stim in group_means[group_key]:
                    spec = group_means[group_key][stim].get('evoked_spec', None)
                    if spec is not None:
                        pair_vmin = min(pair_vmin, np.min(spec))
                        pair_vmax = max(pair_vmax, np.max(spec))
            if pair_vmin == np.inf or pair_vmax == -np.inf:
                pair_scales[(learning, sex)] = (None, None)
            else:
                pair_scales[(learning, sex)] = (pair_vmin, pair_vmax)
    
    for row, learning in enumerate(learning_styles):
        for col in range(ncols):
            ax = axs[row, col]
            sex = 'male' if col < 2 else 'female'
            stim = 'tone_plus' if (col % 2 == 0) else 'tone_minus'
            group_key = f"{sex}_{learning}"
            title_str = f"{group_key.title()} - {'CS+' if stim=='tone_plus' else 'CS-'}"
            vmin, vmax = pair_scales.get((learning, sex), (None, None))
            
            if group_key not in group_means or stim not in group_means[group_key]:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14)
                ax.set_title(title_str + "\n(n/a)")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 20)
            else:
                data = group_means[group_key][stim]
                spec = data.get('evoked_spec', None)
                if spec is None:
                    ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                            ha="center", va="center", fontsize=14)
                    ax.set_title(title_str + "\n(n/a)")
                else:
                    rel_time = data['rel_time']
                    f_vec = data['f']
                    im = ax.imshow(spec, aspect='auto', origin='lower',
                                   extent=[rel_time[0], rel_time[-1], f_vec[0], f_vec[-1]],
                                   cmap='jet', vmin=vmin, vmax=vmax)
                    ax.set_title(title_str)
                    ax.axvspan(0, 0.05, color='gray', alpha=0.5)
                    fig.colorbar(im, ax=ax)
                    
                    # Set custom x-ticks at 5 evenly spaced points
                    x_ticks = np.linspace(rel_time[0], rel_time[-1], 5)
                    ax.set_xticks(x_ticks)

            if row == nrows - 1:
                ax.set_xlabel("Time (s)")
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel("Frequency (Hz)")
            else:
                ax.set_yticklabels([])

    fig.suptitle(f"{brain_region.upper()} {freq_band.title()} Evoked Heat Maps by Sex, Learning, and CS Condition" + title_suffix, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_evoked_bar_graph(group_means, brain_region, freq_band, title_suffix=''):
    """
    Plots evoked bar values for each group defined by sex and learning style,
    separately for CS+ (tone_plus; with hatch) and CS– (tone_minus).
    """
    sexes = ['male', 'female']
    learning_styles = ['misplaced fear', 'discriminator', 'generalizer']
    colors = {'misplaced fear': 'r', 'discriminator': 'g', 'generalizer': 'pink'}
    
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
    
    global_min = np.inf
    global_max = -np.inf
    
    for i, sex in enumerate(sexes):
        for j, learning in enumerate(learning_styles):
            group_key = f"{sex}_{learning}"
            ax = axs[i, j]
            if group_key not in group_means:
                ax.set_title(f"{group_key}\n(n/a)")
                ax.axis('off')
                continue
            
            tone_plus = group_means[group_key].get('tone_plus')
            tone_minus = group_means[group_key].get('tone_minus')
            if tone_plus is None or tone_minus is None:
                ax.set_title(f"{group_key}\n(n/a)")
                ax.axis('off')
                continue
            
            cs_plus_mean  = tone_plus['evoked_bar']
            cs_plus_err   = tone_plus['std_evoked_bar']
            cs_minus_mean = tone_minus['evoked_bar']
            cs_minus_err  = tone_minus['std_evoked_bar']
            
            global_min = min(global_min, cs_plus_mean - cs_plus_err, cs_minus_mean - cs_minus_err)
            global_max = max(global_max, cs_plus_mean + cs_plus_err, cs_minus_mean + cs_minus_err)
            
            x = np.array([0, 1])
            bars = ax.bar(x, [cs_plus_mean, cs_minus_mean],
                          yerr=[cs_plus_err, cs_minus_err],
                          capsize=5,
                          color=colors[learning])
            bars[0].set_hatch('//')
            
            ax.set_xticks(x)
            ax.set_xticklabels(['CS+', 'CS-'])
            ax.set_title(group_key)
            if freq_band == 'theta':
                freq_range_str = '(4-8 Hz)'
            elif freq_band == 'gamma':
                freq_range_str = '(20-50 Hz)'
            elif freq_band == 'high gamma':
                freq_range_str = '(70-120 Hz)'
            else:
                freq_range_str = '(?)'
    
            if j == 0:
                ax.set_ylabel(f'Evoked {freq_band.title()} {freq_range_str} Power')
    
    if global_max > -np.inf and global_min < np.inf:
        margin = (global_max - global_min) * 0.1 if global_max != global_min else 1
        y_lower = global_min - margin
        y_upper = global_max + margin
        for ax in axs.flatten():
            if ax.has_data():
                ax.set_ylim(y_lower, y_upper)
    
    fig.suptitle(f'{brain_region.upper()} Group Evoked {freq_band.title()} Power by Sex, Learning, and CS Condition' + title_suffix, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_group_spectrum_lines(all_animal_means, brain_region, spec_label='Power', 
                              conditions=('tone_plus', 'tone_minus', 'pretone')):
    """
    Plots group spectra (raw, non-evoked) as line plots arranged in a 2x3 grid:
      - Rows correspond to sex (male and female)
      - Columns correspond to learning style (misplaced fear, discriminator, generalizer)
    """
    # Hard-coded grouping using animal_info.
    learning_styles = ['misplaced fear', 'discriminator', 'generalizer']
    sexes = ['male', 'female']
    animal_info = {
        'As107': {'sex': 'male', 'learning': 'generalizer'},
        'As105': {'sex': 'male', 'learning': 'discriminator'},
        'As106': {'sex': 'male', 'learning': 'misplaced fear'},
        'As108': {'sex': 'male', 'learning': 'generalizer'},
        'As110': {'sex': 'female', 'learning': 'misplaced fear'},
        'As111': {'sex': 'female', 'learning': 'generalizer'},
        'As112': {'sex': 'female', 'learning': 'generalizer'},
        'As113': {'sex': 'female', 'learning': 'generalizer'},
    }

    # Group animals by sex and learning style.
    group_animals = {}
    for animal, am in all_animal_means.items():
        if animal not in animal_info:
            continue
        info = animal_info[animal]
        group_key = f"{info['sex']}_{info['learning']}"
        group_animals.setdefault(group_key, []).append(am)

    # Define colors and labels.
    colors = {'tone_plus': 'r', 'tone_minus': 'b', 'pretone': 'gray'}
    cond_labels = {'tone_plus': 'CS+', 'tone_minus': 'CS-', 'pretone': 'Pretone'}

    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

    for i, sex in enumerate(sexes):
        for j, learning in enumerate(learning_styles):
            ax = axs[i, j]
            group_key = f"{sex}_{learning}"
            
            if group_key not in group_animals:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14)
                ax.set_title(group_key)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # For each condition, collect each animal's average spectrum.
            group_spec_mean = {}
            group_spec_err = {}
            f_vec = None
            for cond in conditions:
                animal_lines = []
                for am in group_animals[group_key]:
                    if cond not in am:
                        continue
                    # For each event in this animal, get the event_spectrogram.
                    # Average over events (axis=0) then over time (axis=1) to get a (n_freq,) vector.
                    specs = [d['event_spectrogram'] for d in am[cond]]
                    animal_avg_spec = np.mean(np.array(specs), axis=0)
                    animal_line = np.mean(animal_avg_spec, axis=1)
                    animal_lines.append(animal_line)
                    if f_vec is None and 'f' in am[cond][0]:
                        f_vec = am[cond][0]['f']
                if animal_lines:
                    animal_lines = np.array(animal_lines)  # shape: (n_animals, n_freq)
                    mean_line = np.mean(animal_lines, axis=0)
                    # Compute standard error.
                    se_line = np.std(animal_lines, axis=0, ddof=1) / np.sqrt(animal_lines.shape[0])
                    group_spec_mean[cond] = mean_line
                    group_spec_err[cond] = se_line

            # Plot each condition's mean line and error band.
            for cond, color in colors.items():
                if cond in group_spec_mean and f_vec is not None:
                    ax.plot(f_vec, group_spec_mean[cond], color=color, label=cond_labels[cond])
                    ax.fill_between(f_vec, group_spec_mean[cond] - group_spec_err[cond],
                                    group_spec_mean[cond] + group_spec_err[cond],
                                    color=color, alpha=0.3)
            
            ax.set_title(group_key.capitalize())
            
            # Set x-axis ticks: 5 evenly spaced, rounded to whole numbers.
            if f_vec is not None:
                xticks = np.linspace(f_vec[0], f_vec[-1], 5)
                xticks = np.round(xticks).astype(int)
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticks)
                ax.set_xlabel('Frequency (Hz)')
            
            # Compute y-axis ticks.
            ymin, ymax = ax.get_ylim()
            candidate_steps = [5, 10, 20, 50, 100]
            for step in candidate_steps:
                start_tick = np.floor(ymin / step) * step
                end_tick = np.ceil(ymax / step) * step
                num_ticks = int((end_tick - start_tick) / step + 1)
                if num_ticks <= 6:
                    yticks = np.arange(start_tick, end_tick + step/2, step)
                    break
            else:
                step = candidate_steps[-1]
                start_tick = np.floor(ymin / step) * step
                end_tick = np.ceil(ymax / step) * step
                yticks = np.arange(start_tick, end_tick + step/2, step)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks)
            
            if j == 0:
                ax.set_ylabel(spec_label)
            ax.legend(fontsize=10)
    
    fig.suptitle(f"{brain_region.upper()} Group Spectrum (Raw, Non-Evoked)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =============================================================================
# New helper functions for combined-sex analysis and Wilcoxon tests.
# =============================================================================

def get_animal_info():
    """Return a dictionary of animal info for grouping."""
    return {
        'As105': {'sex': 'male', 'learning': 'discriminator'},
        'As106': {'sex': 'male', 'learning': 'misplaced fear'},
        'As107': {'sex': 'male', 'learning': 'generalizer'},
        'As108': {'sex': 'male', 'learning': 'generalizer'},
        'As110': {'sex': 'female', 'learning': 'misplaced fear'},
        'As111': {'sex': 'female', 'learning': 'generalizer'},
        'As112': {'sex': 'female', 'learning': 'generalizer'},
        'As113': {'sex': 'female', 'learning': 'generalizer'},
    }

def compute_individual_statistics(animals_data, conditions):
    """
    Computes per-animal measures (mean bar and spectrogram) for each condition.
    Returns a dictionary keyed by animal.
    """
    animal_info = get_animal_info()
    all_animal_means = {}
    for animal, data_by_within_subject_condition in animals_data.items():
        animal_means = {}
        for cond in conditions:
            events = data_by_within_subject_condition[cond]
            if len(events) == 0:
                continue
            bar_vals = np.array([ev['bar_value'] for ev in events])
            mean_bar = np.mean(bar_vals)
            spec_stack = np.stack([ev['event_spectrogram'] for ev in events], axis=0)
            mean_spec = np.mean(spec_stack, axis=0)
            animal_means[cond] = {
                'mean_bar': mean_bar,
                'mean_spec': mean_spec,
                'rel_time': events[0]['rel_time'],
                'f': events[0]['f']
            }
        if 'tone_plus' in animal_means and 'pretone' in animal_means:
            animal_means['tone_plus']['evoked_bar'] = animal_means['tone_plus']['mean_bar'] - animal_means['pretone']['mean_bar']
            animal_means['tone_plus']['evoked_spec'] = animal_means['tone_plus']['mean_spec'] - animal_means['pretone']['mean_spec']
        if 'tone_minus' in animal_means and 'pretone' in animal_means:
            animal_means['tone_minus']['evoked_bar'] = animal_means['tone_minus']['mean_bar'] - animal_means['pretone']['mean_bar']
            animal_means['tone_minus']['evoked_spec'] = animal_means['tone_minus']['mean_spec'] - animal_means['pretone']['mean_spec']
        all_animal_means[animal] = animal_means
    return all_animal_means

def group_animal_means(all_animal_means, group_by='sex_learning'):
    """
    Aggregates per-animal measures into group-level means.
    group_by can be 'sex_learning' (default) or 'learning' (combined sexes).
    Returns a dictionary keyed by the grouping category.
    """
    animal_info = get_animal_info()
    grouped = {}
    for animal, measures in all_animal_means.items():
        if animal not in animal_info:
            continue
        info = animal_info[animal]
        if group_by == 'sex_learning':
            group_key = f"{info['sex']}_{info['learning']}"
        elif group_by == 'learning':
            group_key = info['learning']
        else:
            group_key = animal
        grouped.setdefault(group_key, []).append(measures)
    
    group_means = {}
    for group_key, animal_means_list in grouped.items():
        group_means[group_key] = {}
        for cond in ['tone_plus', 'tone_minus', 'pretone']:
            bar_vals = []
            spec_vals = []
            evoked_bar_vals = []
            evoked_spec_vals = []
            rel_time = None
            f_vec = None
            for am in animal_means_list:
                if cond not in am:
                    continue
                bar_vals.append(am[cond]['mean_bar'])
                spec_vals.append(am[cond]['mean_spec'])
                if cond in ['tone_plus', 'tone_minus']:
                    evoked_bar_vals.append(am[cond]['evoked_bar'])
                    evoked_spec_vals.append(am[cond]['evoked_spec'])
                if rel_time is None:
                    rel_time = am[cond]['rel_time']
                if f_vec is None:
                    f_vec = am[cond]['f']
            if len(bar_vals) == 0:
                continue
            group_means[group_key][cond] = {
                'mean_bar': np.mean(bar_vals),
                'evoked_bar': np.mean(evoked_bar_vals) if evoked_bar_vals else None,
                'std_bar': np.std(bar_vals),
                'std_evoked_bar': np.std(evoked_bar_vals) if evoked_bar_vals else None,
                'sem_evoked_bar': np.std(evoked_bar_vals) / np.sqrt(len(evoked_bar_vals)) if evoked_bar_vals else None,
                'mean_spec': np.mean(spec_vals, axis=0),
                'evoked_spec': np.mean(evoked_spec_vals, axis=0) if evoked_spec_vals else None,
                'std_spec': np.std(spec_vals, axis=0),
                'std_evoked_spec': np.std(evoked_spec_vals, axis=0) if evoked_spec_vals else None,
                'sem_evoked_spec': np.std(evoked_spec_vals, axis=0) / np.sqrt(len(evoked_spec_vals)) if evoked_spec_vals else None,
                'rel_time': rel_time,
                'f': f_vec
            }
    return group_means

def perform_paired_wilcoxon_tests(animals_data, freq_band, brain_region, group_by='sex_learning'):
    """
    Pools all event-level evoked power observations (tone minus pretone)
    across animals according to the specified grouping and performs a paired
    Wilcoxon signed-rank test on the two dependent populations.

    For each animal, the evoked power is computed as:
      CS+ evoked = tone_plus_event['bar_value'] - pretone_plus_event['bar_value']
      CS– evoked = tone_minus_event['bar_value'] - pretone_minus_event['bar_value']

    Parameters:
      animals_data: dictionary of animal data from collect_animals_data().
      freq_band: frequency band label (for printing).
      brain_region: brain region label (for printing).
      group_by: either 'sex_learning' (grouping by sex & learning) or 'learning' (ignoring sex).

    The function pools paired observations across animals in each group and then
    performs the paired Wilcoxon test, printing out the statistic and p-value.
    """
    from scipy.stats import wilcoxon
    animal_info = get_animal_info()
    groups = {}  # Dictionary to collect data for each group.
    
    for animal, data in animals_data.items():
        if animal not in animal_info:
            continue
        # Determine group key based on the group_by parameter.
        if group_by == 'sex_learning':
            group_key = f"{animal_info[animal]['sex']}_{animal_info[animal]['learning']}"
        elif group_by == 'learning':
            group_key = animal_info[animal]['learning']
        else:
            print(f"Invalid group_by argument: {group_by}.")
            return
        
        # Verify that the necessary event data exists.
        required_keys = ['tone_plus', 'pretone_plus', 'tone_minus', 'pretone_minus']
        if not all(key in data for key in required_keys):
            continue
        if (len(data['tone_plus']) != len(data['pretone_plus']) or 
            len(data['tone_minus']) != len(data['pretone_minus'])):
            print(f"Animal {animal} has mismatched event counts; skipping.")
            continue
        
        # Compute paired evoked power values for this animal.
        evoked_plus = [tp['bar_value'] - pp['bar_value'] 
                       for tp, pp in zip(data['tone_plus'], data['pretone_plus'])]
        evoked_minus = [tm['bar_value'] - pm['bar_value'] 
                        for tm, pm in zip(data['tone_minus'], data['pretone_minus'])]
        
        # Pool the data into the appropriate group.
        if group_key not in groups:
            groups[group_key] = {'evoked_plus': [], 'evoked_minus': []}
        groups[group_key]['evoked_plus'].extend(evoked_plus)
        groups[group_key]['evoked_minus'].extend(evoked_minus)
    
    # Perform the paired Wilcoxon signed-rank test for each group.
    print(f"\nPaired Wilcoxon tests ({group_by}) for brain region {brain_region}, frequency band {freq_band}:")
    for group, vals in groups.items():
        evoked_plus_all = vals['evoked_plus']
        evoked_minus_all = vals['evoked_minus']
        if len(evoked_plus_all) != len(evoked_minus_all) or len(evoked_plus_all) == 0:
            print(f"Group {group}: Unequal or insufficient paired events; test not performed.")
            continue
        stat, p = wilcoxon(evoked_plus_all, evoked_minus_all)
        print(f"Group {group}: Wilcoxon statistic = {stat:.3f}, p-value = {p:.3f}")

# =============================================================================
# New plotting functions for combined-sex (learning-only) graphs.
# =============================================================================

def plot_evoked_bar_graph_combined(group_means, brain_region, freq_band, title_suffix=''):
    """
    Plots evoked bar graphs by learning style only (combining sexes).
    """
    learning_styles = ['discriminator', 'generalizer', 'misplaced fear']
    colors = {'discriminator': '#0434ff', 'generalizer': '#797977', 'misplaced fear': '#fffc00'}
    
    # Use a narrower overall figure.
    fig, axs = plt.subplots(1, 3, figsize=(4, 4), sharey=True)
    global_min = np.inf
    global_max = -np.inf
    
    for j, learning in enumerate(learning_styles):
        ax = axs[j]
        if learning not in group_means:
            ax.set_title(f"{learning}\n(n/a)")
            ax.axis('off')
            continue
        tone_plus = group_means[learning].get('tone_plus')
        tone_minus = group_means[learning].get('tone_minus')
        if tone_plus is None or tone_minus is None:
            ax.set_title(f"{learning}\n(n/a)")
            ax.axis('off')
            continue
        
        cs_plus_mean  = tone_plus['evoked_bar']
        cs_plus_err   = tone_plus['sem_evoked_bar']
        cs_minus_mean = tone_minus['evoked_bar']
        cs_minus_err  = tone_minus['sem_evoked_bar']
        
        global_min = min(global_min, cs_plus_mean - cs_plus_err, cs_minus_mean - cs_minus_err)
        global_max = max(global_max, cs_plus_mean + cs_plus_err, cs_minus_mean + cs_minus_err)
        
        # Place two half-as-wide bars (width=0.2) within a narrow x-axis (0 to 0.5).
        x = [0.15, 0.35]
        bars = ax.bar(
            x, [cs_plus_mean, cs_minus_mean],
            yerr=[cs_plus_err, cs_minus_err],
            capsize=5,
            width=0.2,
            color=colors[learning]
        )
        bars[0].set_hatch('//')
        ax.set_xticks(x)
        ax.set_xticklabels(['CS+', 'CS-'])
        ax.set_xlim(0, 0.5)
        ax.set_title(learning)
        
        # Remove top and right spines.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if j == 0:
            ax.set_ylabel(f'Evoked {freq_band.title()} Power')
    
    if global_max > -np.inf and global_min < np.inf:
        margin = (global_max - global_min) * 0.1 if global_max != global_min else 1
        y_lower = global_min - margin
        y_upper = global_max + margin
        for ax in axs:
            if ax.has_data():
                ax.set_ylim(y_lower, y_upper)
    
    # Adjust suptitle to span only the middle part of the figure.
    fig.suptitle(f'{brain_region.upper()} Group Evoked {freq_band.title()} Power by Learning (Combined Sex)' + title_suffix,
                 fontsize=12, x=0.5)
    plt.tight_layout(rect=[0.25, 0.03, 0.75, 0.90])
    plt.show()

def plot_evoked_heatmaps_combined(group_means, brain_region, freq_band, title_suffix='', cbar_vmin=None, cbar_vmax=None):
    """
    Plots evoked heat maps by learning style only (combining sexes).
    Each row corresponds to a learning style; columns are CS+ and CS–.
    
    Optional Parameters:
    cbar_vmin, cbar_vmax : float or None
        If provided, these values set the colorbar range for the heat maps.
        If None, the range is computed from the data for each learning style.
    """
    learning_styles = ['discriminator', 'generalizer', 'misplaced fear']
    stimulus_types = ['tone_plus', 'tone_minus']
    
    nrows = len(learning_styles)
    ncols = 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 12))
    
    for i, learning in enumerate(learning_styles):
        # Calculate the colorbar range if not provided
        if cbar_vmin is None or cbar_vmax is None:
            pair_vmin = np.inf
            pair_vmax = -np.inf
            for stim in stimulus_types:
                if learning in group_means and stim in group_means[learning]:
                    spec = group_means[learning][stim].get('evoked_spec', None)
                    if spec is not None:
                        pair_vmin = min(pair_vmin, np.min(spec))
                        pair_vmax = max(pair_vmax, np.max(spec))
            if pair_vmin == np.inf or pair_vmax == -np.inf:
                pair_vmin, pair_vmax = None, None
        else:
            pair_vmin, pair_vmax = cbar_vmin, cbar_vmax
        
        for j, stim in enumerate(stimulus_types):
            ax = axs[i, j]
            title_str = f"{learning.title()} - {'CS+' if stim=='tone_plus' else 'CS-'}"
            if learning not in group_means or stim not in group_means[learning]:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14)
                ax.set_title(title_str + "\n(n/a)")
                continue
            data = group_means[learning][stim]
            spec = data.get('evoked_spec', None)
            if spec is None:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14)
                ax.set_title(title_str + "\n(n/a)")
            else:
                rel_time = data['rel_time']
                f_vec = data['f']
                im = ax.imshow(spec, aspect='auto', origin='lower',
                               extent=[rel_time[0], rel_time[-1], f_vec[0], f_vec[-1]],
                               cmap='jet', vmin=pair_vmin, vmax=pair_vmax)
                ax.set_title(title_str)
                ax.axvspan(0, 0.05, color='gray', alpha=0.5)
                fig.colorbar(im, ax=ax)
            if i == nrows - 1:
                ax.set_xlabel("Time (s)")
            else:
                ax.set_xticklabels([])
            #if j == 0:
            ax.set_ylabel("Frequency (Hz)")
            #else:
            #ax.set_yticklabels([])
    
    fig.suptitle(f"{brain_region.upper()} {freq_band.title()} Evoked Heat Maps by Learning (Combined Sex)" + title_suffix, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_group_spectrum_lines_combined(group_means, brain_region, spec_label='Power', 
                                       conditions=('tone_plus', 'tone_minus', 'pretone')):
    """
    Plots group spectra as line plots arranged in a 1x3 grid for each learning style (combined sexes).
    For each learning style, plots lines for CS+, CS–, and Pretone.
    """
    learning_styles = ['misplaced fear', 'discriminator', 'generalizer']
    colors = {'tone_plus': 'r', 'tone_minus': 'b', 'pretone': 'gray'}
    cond_labels = {'tone_plus': 'CS+', 'tone_minus': 'CS-', 'pretone': 'Pretone'}
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    
    for idx, learning in enumerate(learning_styles):
        ax = axs[idx]
        if learning not in group_means:
            ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            ax.set_title(learning)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        data_group = group_means[learning]
        f_vec = None
        for cond in conditions:
            if cond in data_group:
                if f_vec is None:
                    f_vec = data_group[cond]['f']
                ax.plot(f_vec, data_group[cond]['mean_spec'].mean(axis=1), 
                        color=colors.get(cond, 'k'), label=cond_labels.get(cond, cond))
        ax.set_title(learning.title())
        if f_vec is not None:
            xticks = np.linspace(f_vec[0], f_vec[-1], 5)
            xticks = np.round(xticks).astype(int)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks)
            ax.set_xlabel('Frequency (Hz)')
        ax.legend(fontsize=10)
        if idx == 0:
            ax.set_ylabel(spec_label)
    
    fig.suptitle(f"{brain_region.upper()} Group Spectrum (Raw, Non-Evoked) by Learning (Combined Sex)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_evoked_by_period_single_animal(animal_data, animal_name):
    """
    Plots CS+ and CS– evoked values period by period for a single animal.
    
    The evoked value for each event is computed as:
       evoked_value = tone_event['bar_value'] - pretone_event['bar_value']
    
    The function extracts events stored under 'tone_plus'/'pretone_plus' (CS+)
    and 'tone_minus'/'pretone_minus' (CS–), then plots them versus the event index.
    
    Parameters:
        animal_data: dict, event data for a single animal (as produced by collect_animals_data)
        animal_name: string, identifier of the animal.
    """
    # Extract CS+ events
    tone_plus_events = animal_data.get('tone_plus', [])
    pretone_plus_events = animal_data.get('pretone_plus', [])
    cs_plus_periods = []
    cs_plus_evoked = []
    for t_event, p_event in zip(tone_plus_events, pretone_plus_events):
        cs_plus_periods.append(t_event['event_index'])
        cs_plus_evoked.append(t_event['bar_value'] - p_event['bar_value'])
    
    # Extract CS– events
    tone_minus_events = animal_data.get('tone_minus', [])
    pretone_minus_events = animal_data.get('pretone_minus', [])
    cs_minus_periods = []
    cs_minus_evoked = []
    for t_event, p_event in zip(tone_minus_events, pretone_minus_events):
        cs_minus_periods.append(t_event['event_index'])
        cs_minus_evoked.append(t_event['bar_value'] - p_event['bar_value'])
    
    # Sort the data by period index (if not already sorted)
    cs_plus_periods, cs_plus_evoked = zip(*sorted(zip(cs_plus_periods, cs_plus_evoked)))
    cs_minus_periods, cs_minus_evoked = zip(*sorted(zip(cs_minus_periods, cs_minus_evoked)))
    
    plt.figure(figsize=(10, 5))
    plt.plot(cs_plus_periods, cs_plus_evoked, marker='o', linestyle='-', color='red', label='CS+')
    plt.plot(cs_minus_periods, cs_minus_evoked, marker='o', linestyle='-', color='blue', label='CS–')
    plt.xlabel('Period')
    plt.ylabel('Evoked Value (tone - pretone)')
    plt.title(f'CS+ and CS– Evoked Values by Period for {animal_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================================================================
# Data collection and statistics
# =============================================================================

def collect_animals_data(conditions, brain_region, freq_band, event_boundary, num_periods=6, filter_first=True, rms=True):
    main_dir = '/Users/katie/likhtik/AS'
    valid_animals = ['As107', 'As112', 'As106', 'As105','As108','As110', 'As111', 'As113']
    matlab_path = '/Applications/MATLAB_R2022a.app/bin/matlab'  # Use MATLAB executable path
    fs = 2000
    tone_on_code = 65503
    expected_tone_periods = 12
    
    if freq_band == 'theta':
        freq_range = (3.8, 8.2) 
    elif freq_band == 'low frequencies':
        freq_range = (0, 20)
    elif freq_band == 'gamma':
        freq_range = (20, 50)
    elif freq_band == 'high gamma':
        freq_range = (70, 120)
    elif freq_band == 'delta to gamma':
        freq_range = (0, 120)
    else:
        raise ValueError('unknown frequency')
    
    mtcsg_args = freq_band_to_mtcsg_args(freq_band)
    
    animals_data = {}
    
    for animal in valid_animals:
        global CURRENT_ANIMAL
        CURRENT_ANIMAL = animal
        data_by_within_subject_condition = {cond: [] for cond in conditions}
    
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
        if len(tone_indices) != expected_tone_periods:
            print(f"Warning: {animal} has {len(tone_indices)} tone events (expected {expected_tone_periods}).")
        tone_period_times = tone_timestamps[tone_indices]
        
        mapping = get_electrode_mapping(animal)
        ch = mapping[brain_region]  
        
        for period_idx, tone_on in enumerate(tone_period_times):
            global CURRENT_IDX
            CURRENT_IDX = period_idx
            # Define extraction windows for tone and pretone segments.
            if freq_band in ['theta', 'low frequencies']:
                tone_seg_start = tone_on - 1
                tone_seg_end   = tone_on + 31
                pretone_seg_start = tone_on - 31
                pretone_seg_end   = tone_on + 1
            else:
                tone_seg_start = tone_on - 1.125
                tone_seg_end   = tone_on + 31.125
                pretone_seg_start = tone_on - 31.25
                pretone_seg_end   = tone_on + 1.125
    
            tone_idx_start = round(tone_seg_start * fs)
            tone_idx_end   = round(tone_seg_end * fs)
            pretone_idx_start = round(pretone_seg_start * fs)
            pretone_idx_end   = round(pretone_seg_end * fs)
            
            if filter_first:
                filtered_data = filter_60_hz(NS3_data[ch, :], fs)
                if rms:
                    filtered_data = divide_by_rms(filtered_data)

                try:
                    tone_data = filtered_data[tone_idx_start:tone_idx_end]
                    pretone_data = filtered_data[pretone_idx_start:pretone_idx_end]
                except Exception as e:
                    print(f"Error extracting data for {animal} event {period_idx}: {e}")
                    continue


            else:
                try:
                    tone_data = NS3_data[ch, tone_idx_start:tone_idx_end]
                    tone_data = filter_60_hz(tone_data, fs)
                    if rms:
                        tone_data = divide_by_rms(tone_data)
                    pretone_data = NS3_data[ch, pretone_idx_start:pretone_idx_end]
                    pretone_data = filter_60_hz(pretone_data, fs)
                    if rms:
                        pretone_data = divide_by_rms(pretone_data)
                except Exception as e:
                    print(f"Error extracting data for {animal} event {period_idx}: {e}")
                    continue

            try:
                S_tone, f_tone, t_tone = run_matlab_mtcsg(tone_data, matlab_path, brain_region, animal, period_idx, 'tone', str(filter_first), str(rms), mtcsg_args)
                S_pretone, f_pretone, t_pretone = run_matlab_mtcsg(pretone_data, matlab_path, brain_region, animal, period_idx, 'pretone', str(filter_first), str(rms), mtcsg_args)
        
            except Exception as e:
                print(f"Error computing mtcsg for {animal} event {period_idx}: {e}")
                continue
            
            
            try:
                event_tone_S, rel_time, f_restricted, tone_bar = process_event_segment(
                    S_tone, f_tone, t_tone, freq_band, freq_range, event_boundary)
                event_pretone_S, rel_time, f_restricted, pretone_bar = process_event_segment(
                    S_pretone, f_pretone, t_pretone, freq_band, freq_range, event_boundary)
            except Exception as e:
                print(f"Error processing event segmentation for {animal} event {period_idx}: {e}")
                continue

            cs_minus_ids = [0, 2, 4, 5, 8, 11]
            if period_idx in cs_minus_ids:
                tone_condition = 'tone_minus'
                pretone_condition = 'pretone_minus'
            else:
                tone_condition = 'tone_plus'
                pretone_condition = 'pretone_plus'
            
            tone_event = {
                'event_spectrogram': event_tone_S,
                'rel_time': rel_time,
                'f': f_restricted,
                'bar_value': tone_bar,
                'animal': animal,
                'event_index': period_idx,
                'learning': {'As105': 'discriminator','As106': 'misplaced fear','As107': 'generalizer','As108': 'generalizer',
                             'As110': 'misplaced fear','As111': 'generalizer','As112': 'generalizer','As113': 'generalizer'}[animal],
                'sex': ('male' if animal in (['As107', 'As108'] + ['As105', 'As106'])
                        else 'female')
            }
            pretone_event = {
                'event_spectrogram': event_pretone_S,
                'rel_time': rel_time,
                'f': f_restricted,
                'bar_value': pretone_bar,
                'animal': animal,
                'event_index': period_idx,
                'learning': {'As105': 'discriminator','As106': 'misplaced fear','As107': 'generalizer','As108': 'generalizer',
                             'As110': 'misplaced fear','As111': 'generalizer','As112': 'generalizer','As113': 'generalizer'}[animal],
                'sex': ('male' if animal in (['As107', 'As108'] + ['As105', 'As106'])
                        else 'female')
            }
            
            data_by_within_subject_condition[tone_condition].append(tone_event)
            data_by_within_subject_condition[pretone_condition].append(pretone_event)
          
        animals_data[animal] = data_by_within_subject_condition
       
        for condition in animals_data[animal]:
            animals_data[animal][condition] = animals_data[animal][condition][0:num_periods]
        
        animals_data[animal]['pretone'] = (
        animals_data[animal].get('pretone_plus', []) +
        animals_data[animal].get('pretone_minus', [])
        )

    
    return animals_data

def compute_group_statistics(animals_data, conditions):
    """
    (Original grouping by sex and learning)
    Computes group-level statistics by averaging per-animal measures.
    """
    all_animal_means = compute_individual_statistics(animals_data, conditions)
    return group_animal_means(all_animal_means, group_by='sex_learning')

# =============================================================================
# Main processing function: iterate over animals and events, build data structures,
# perform Wilcoxon tests, and generate both original and combined plots.
# =============================================================================

def main():
    conditions = ['tone_plus', 'tone_minus', 'pretone_plus', 'pretone_minus', 'pretone']
    discriminator_animal = 'As105'

    for brain_region in ['pl']:
        filter_first_animals_data = collect_animals_data(conditions, brain_region, 'theta', event_boundary=(0, 0.3), num_periods=3, filter_first=True, rms=False)
        plot_evoked_by_period_single_animal(filter_first_animals_data[discriminator_animal], discriminator_animal)

        filter_first_all_animal_means = compute_individual_statistics(filter_first_animals_data, conditions)
        perform_paired_wilcoxon_tests(filter_first_animals_data, freq_band='theta', brain_region=brain_region, group_by='learning')
        filter_first_combined_group_means = group_animal_means(filter_first_all_animal_means, group_by='learning')
        plot_evoked_bar_graph_combined(filter_first_combined_group_means, brain_region, 'theta', title_suffix=' Three Periods Filter First')

        filter_second_animals_data = collect_animals_data(conditions, brain_region, 'theta', event_boundary=(0, 0.3), num_periods=3, filter_first=False, rms=False)
        plot_evoked_by_period_single_animal(filter_second_animals_data[discriminator_animal], discriminator_animal)

        filter_second_all_animal_means = compute_individual_statistics(filter_second_animals_data, conditions)
        perform_paired_wilcoxon_tests(filter_second_animals_data, freq_band='theta', brain_region=brain_region, group_by='learning')
        filter_second_combined_group_means = group_animal_means(filter_second_all_animal_means, group_by='learning')
        plot_evoked_bar_graph_combined(filter_second_combined_group_means, brain_region, 'theta', title_suffix=' Three Periods Filter Second')
       
            

        

    
    
    # for brain_region in ['pl', 'bla', 'vhip']:
    #     # First, plot group spectrum lines using low frequencies (original grouping).
    #     animals_data = collect_animals_data(conditions, brain_region, 'low frequencies', event_boundary=(0, 0.3))
    #     plot_group_spectrum_lines(animals_data, brain_region)
    #     # Also plot combined (learning-only) spectrum lines.
    #     all_animal_means_lines = compute_individual_statistics(animals_data, conditions)
    #     combined_lines_group_means = group_animal_means(all_animal_means_lines, group_by='learning')
    #     plot_group_spectrum_lines_combined(combined_lines_group_means, brain_region)
        
    #     for freq_name in ['theta', 'gamma', 'high gamma']:
    #         # For bar graphs and stats.
    #         bar_animals_data = collect_animals_data(conditions, brain_region, freq_name, event_boundary=(0, 0.3))
    #         all_animal_means = compute_individual_statistics(bar_animals_data, conditions)
            
    #         # Perform Wilcoxon tests.
    #         # For grouping by sex and learning:
    #         perform_paired_wilcoxon_tests(bar_animals_data, freq_band=freq_name, brain_region=brain_region, group_by='sex_learning')

    #         # For grouping by learning only (ignoring sex):
    #         perform_paired_wilcoxon_tests(bar_animals_data, freq_band=freq_name, brain_region=brain_region, group_by='learning')
            
    #         # Original grouped plots (sex + learning)
    #         group_means = group_animal_means(all_animal_means, group_by='sex_learning')
    #         plot_evoked_bar_graph(group_means, brain_region, freq_name)
            
    #         # Combined (learning only) bar graphs.
    #         combined_group_means = group_animal_means(all_animal_means, group_by='learning')
    #         plot_evoked_bar_graph_combined(combined_group_means, brain_region, freq_name)
            
    #         # For heat maps.
    #         freq_plot_name = 'low frequencies' if freq_name=='theta' else freq_name
    #         heat_map_animals_data = collect_animals_data(conditions, brain_region, freq_plot_name, event_boundary=(-0.1, 0.3))
    #         all_animal_means_heat = compute_individual_statistics(heat_map_animals_data, conditions)
    #         heat_map_group_means = group_animal_means(all_animal_means_heat, group_by='sex_learning')
    #         plot_evoked_heatmaps(heat_map_group_means, brain_region, freq_plot_name)
            
    #         combined_heat_map_group_means = group_animal_means(all_animal_means_heat, group_by='learning')
    #         plot_evoked_heatmaps_combined(combined_heat_map_group_means, brain_region, freq_plot_name)

# =============================================================================
# Run the main processing
# =============================================================================

if __name__ == "__main__":
    main()