#!/usr/bin/env python3
import os
import numpy as np
import subprocess
import tempfile
import scipy.signal as signal
import scipy.io
import matplotlib.pyplot as plt

# =============================================================================
# Helper functions for filtering, MATLAB interfacing, etc.
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
    with open(script_filename, "w") as f:
        f.write(matlab_script)
    
    cmd = [matlab_path, "-nodisplay", "-nosplash", "-r", f"run('{os.path.abspath(script_filename)}')"]
    subprocess.run(cmd, check=True)
    os.remove(script_filename)

# gamma FFT 2048,Fs 2000,moving win in samples 500,overlap in samples 480, 2
# 2048,2000,1000,980,2 
def run_matlab_mtcsg(data, matlab_path, brain_region,
                     animal, period_number, period_type, params=('2048', '2000', '1000', '980', '2'),
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
    
    base_filename = f"{brain_region}_{animal}_period{period_number}_{period_type}_mtcsg_{'_'.join(params)}"
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
    Processes one event's spectrogram as follows:
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
# Plotting functions
# =============================================================================

def plot_evoked_heatmaps(group_means, brain_region, freq_band):
    """
    Plots evoked heat maps for 12 conditions arranged as follows:
      - Rows correspond to learning styles:
           Row 1: Discriminators
           Row 2: Generalizers
           Row 3: Bad Learners
      - Within each row, the four columns are:
           Column 1: Male CS+ (tone_plus)
           Column 2: Male CS– (tone_minus)
           Column 3: Female CS+ (tone_plus)
           Column 4: Female CS– (tone_minus)
    
    For each (learning, sex) pair, a common color scale is computed from both CS+ and CS– data,
    so that the two subplots in the pair share the same vmin/vmax. Each subplot uses the 'jet'
    colormap and overlays a vertical translucent gray patch from 0 to 0.05 s.
    
    Parameters:
      group_means: dict with keys like "male_discriminator", "male_generalizer", etc. Each contains
                   sub-dicts for conditions (e.g. "tone_plus" and "tone_minus") with:
                       - "mean_spec": 2D evoked spectrogram (n_freq x n_time)
                       - "rel_time": 1D relative time axis (in seconds)
                       - "f": 1D frequency vector (already restricted to desired range)
      brain_region: string (e.g., 'vhip') to include in the title.
    """
    learning_styles = ['discriminator', 'generalizer', 'bad_learner']
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
            if row == nrows - 1:
                ax.set_xlabel("Time (s)")
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel("Frequency (Hz)")
            else:
                ax.set_yticklabels([])
    
    fig.suptitle(f"{brain_region.upper()} {freq_band.title()} Evoked Heat Maps by Sex, Learning, and CS Condition", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_evoked_bar_graph(group_means, brain_region, freq_band):
    """
    Plots evoked bar values for each group defined by sex and learning style,
    separately for CS+ (tone_plus; with hatch) and CS– (tone_minus).
    
    Parameters:
      group_means: dict with keys like "male_bad_learner", etc. For each group,
                   group_means[group] is a dict with keys "tone_plus" and "tone_minus"
                   containing:
                       - "mean_bar": evoked value,
                       - "std_bar": error bar value,
                       - "rel_time": ..., "f": ...
      brain_region: string (e.g., 'vhip') for labeling.
      freq_band: string indicating the frequency band (affects axis labeling).
    """
    sexes = ['male', 'female']
    learning_styles = ['bad_learner', 'discriminator', 'generalizer']
    colors = {'bad_learner': 'r', 'discriminator': 'g', 'generalizer': 'pink'}
    
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
    
    fig.suptitle(f'{brain_region.upper()} Group Evoked {freq_band.title()} Power by Sex, Learning, and CS Condition', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_group_spectrum_lines(all_animal_means, brain_region, spec_label='Power', 
                              conditions=('tone_plus', 'tone_minus', 'pretone')):
    """
    Plots group spectra (raw, non-evoked) as line plots arranged in a 2x3 grid:
      - Rows correspond to sex (male and female)
      - Columns correspond to learning style (bad_learner, discriminator, generalizer)
    
    For each group, the spectrum for each condition is computed by averaging the raw spectrogram
    (i.e., 'event_spectrogram') over the time axis for each animal, and then averaging over animals.
    Three lines are plotted:
       - CS+ (tone_plus) in red,
       - CS– (tone_minus) in blue,
       - and Pretone in gray.
       
    A translucent error band (standard error over animals) is also added.
    
    Parameters:
      all_animal_means: dict keyed by animal, each value is a dict with condition keys that contain:
                           - 'event_spectrogram': 2D array (n_freq x n_time)
                           - 'f': 1D frequency vector
      brain_region: string (e.g., 'bla') for labeling.
      spec_label: label for the y-axis.
      conditions: tuple of condition keys to plot.
    """

    # Define grouping info.
    learning_styles = ['bad_learner', 'discriminator', 'generalizer']
    sexes = ['male', 'female']
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
            
            # Compute y-axis ticks: limit to at most 6 ticks and force them to be multiples of 5 or 10.
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
                # Fallback: use the largest candidate.
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

def collect_animals_data(conditions, brain_region, freq_band, event_boundary):
    main_dir = '/Users/katie/likhtik/AS'
    valid_animals = ['As111', 'As107', 'As112', 'As106', 'As105','As108','As110','As113']
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
        
        # Process the entire brain region time series once:
        # processed_signal = divide_by_rms(filter_60_hz(NS3_data[ch, :], fs))
        
        for period_idx, tone_on in enumerate(tone_period_times):
            # Define extraction windows for tone and pretone segments.
            if freq_band in ['theta', 'low frequencies']:
                tone_seg_start = tone_on - 1
                tone_seg_end   = tone_on + 31
                pretone_seg_start = tone_on - 32
                pretone_seg_end   = tone_on
            else:
                tone_seg_start = tone_on - 1.125
                tone_seg_end   = tone_on + 31.125
                pretone_seg_start = tone_on - 32.25
                pretone_seg_end   = tone_on
    
            tone_idx_start = int(round(tone_seg_start * fs))
            tone_idx_end   = int(round(tone_seg_end * fs))
            pretone_idx_start = int(round(pretone_seg_start * fs))
            pretone_idx_end   = int(round(pretone_seg_end * fs))
            
            # try:
            #     tone_data = processed_signal[tone_idx_start:tone_idx_end]
            #     pretone_data = processed_signal[pretone_idx_start:pretone_idx_end]
            # except Exception as e:
            #     print(f"Error extracting data for {animal} event {period_idx}: {e}")
            #     continue

            try:
                tone_data = NS3_data[ch, tone_idx_start:tone_idx_end]
                pretone_data = NS3_data[ch, pretone_idx_start:pretone_idx_end]
            except Exception as e:
                print(f"Error extracting data for {animal} event {period_idx}: {e}")
                continue
            
            tone_data = divide_by_rms(filter_60_hz(tone_data, fs))
            pretone_data = divide_by_rms(filter_60_hz(pretone_data, fs))
            
            try:
                S_tone, f_tone, t_tone = run_matlab_mtcsg(tone_data, matlab_path, brain_region, animal, period_idx, 'tone', mtcsg_args)
                S_pretone, f_pretone, t_pretone = run_matlab_mtcsg(pretone_data, matlab_path, brain_region, animal, period_idx, 'pretone', mtcsg_args)
            except Exception as e:
                print(f"Error computing mtcsg for {animal} event {period_idx}: {e}")
                continue
            

            try:
                # Now call process_event_segment without discard_start/discard_end.
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
                'learning': {'As105': 'discriminator','As106': 'bad_learner','As107': 'generalizer','As108': 'generalizer',
                             'As110': 'bad_learner','As111': 'generalizer','As112': 'generalizer','As113': 'generalizer'}[animal],
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
                'learning': {'As105': 'discriminator','As106': 'bad_learner','As107': 'generalizer','As108': 'generalizer',
                             'As110': 'bad_learner','As111': 'generalizer','As112': 'generalizer','As113': 'generalizer'}[animal],
                'sex': ('male' if animal in (['As107', 'As108'] + ['As105', 'As106'])
                        else 'female')
            }
            
            data_by_within_subject_condition[tone_condition].append(tone_event)
            data_by_within_subject_condition[pretone_condition].append(pretone_event)
            data_by_within_subject_condition['pretone'].append(pretone_event)
        
        animals_data[animal] = data_by_within_subject_condition
    
    return animals_data

def compute_group_statistics(animals_data, conditions):
    """
    Computes group-level statistics by averaging per-animal measures.
    
    For each animal and condition, it computes the mean bar value and spectrogram,
    and then for tone conditions, computes evoked measures by subtracting the pretone baseline.
    
    Returns:
      group_means: dict keyed by group (e.g., "male_discriminator") containing statistics for each condition.
    """
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
        for cond in ['tone_plus', 'tone_minus']:
            evoked_bar = animal_means[cond]['mean_bar'] - animal_means['pretone']['mean_bar']
            evoked_spec = animal_means[cond]['mean_spec'] - animal_means['pretone']['mean_spec']
            animal_means[cond]['evoked_bar'] = evoked_bar
            animal_means[cond]['evoked_spec'] = evoked_spec
        all_animal_means[animal] = animal_means
    
    group_animals = {}
    for animal, animal_means in all_animal_means.items():
        info = animal_info[animal]
        group_key = f"{info['sex']}_{info['learning']}"
        group_animals.setdefault(group_key, []).append(animal_means)
    
    group_means = {}
    for group_key, animal_means_list in group_animals.items():
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

            def get_mean_and_std(vals, axis=None, operation=np.array):
                if not vals:
                    return None, None
                vals = operation(vals)     
                m = np.mean(vals, axis=axis)
                std = np.std(vals, axis=axis)
                
                return m, std
            
            mean_bar, std_bar = get_mean_and_std(bar_vals)
            mean_evoked_bar, std_evoked_bar = get_mean_and_std(evoked_bar_vals)

            mean_spec, std_spec = get_mean_and_std(spec_vals, operation=np.stack, axis=0)
            mean_evoked_spec, std_evoked_spec = get_mean_and_std(evoked_spec_vals, operation=np.stack, axis=0)

            group_means[group_key][cond] = {
                'mean_bar': mean_bar,
                'evoked_bar': mean_evoked_bar,
                'std_bar': std_bar,
                'std_evoked_bar': std_evoked_bar,
                'mean_spec': mean_spec,
                'evoked_spec': mean_evoked_spec,
                'std_spec': std_spec,
                'std_evoked_spec': std_evoked_spec,
                'rel_time': rel_time,
                'f': f_vec
            }
    return group_means

# =============================================================================
# Main processing function: iterate over animals and events, build data structures
# =============================================================================

def main():
    conditions = ['tone_plus', 'tone_minus', 'pretone_plus', 'pretone_minus', 'pretone']
    
    for brain_region in ['pl', 'bla', 'vhip']:

        animals_data = collect_animals_data(conditions, brain_region, 'low frequencies', event_boundary=(0, 0.3))
        plot_group_spectrum_lines(animals_data, brain_region)
    
        for freq_name in ['theta', 'gamma', 'high gamma']:

            # PlPperot bar graphs.
            bar_animals_data = collect_animals_data(conditions, brain_region, freq_name, event_boundary=(0, 0.3))
            bar_graph_group_means = compute_group_statistics(bar_animals_data, conditions)
            plot_evoked_bar_graph(bar_graph_group_means, brain_region, freq_name)
    
            # Plot heat maps.
            if freq_name == 'theta':
                freq_name = 'low frequencies'
            heat_map_animals_data = collect_animals_data(conditions, brain_region, freq_name, event_boundary=(-0.1, 0.3))
            heat_map_group_means = compute_group_statistics(heat_map_animals_data, conditions)
            plot_evoked_heatmaps(heat_map_group_means, brain_region, freq_name)

# =============================================================================
# Run the main processing
# =============================================================================

if __name__ == "__main__":
    main()