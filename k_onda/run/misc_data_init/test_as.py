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

def run_matlab_mtcsg(data, matlab_path, brain_region,
                     animal, period_number, period_type,
                     param1=2048, param2=2000, param3=500, param4=480, param5=2,
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
      param1, param2, param3, param4, param5: parameters for mtcsg (defaults match the original call).
      output_dir: directory where S, f, and t are saved.
    
    Returns:
      S, f, t as numpy arrays.
    """
    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)
    
    # Build a base filename that encodes all the identifiers.
    base_filename = f"{brain_region}_{animal}_period{period_number}_{period_type}_mtcsg_{param1}_{param2}_{param3}_{param4}_{param5}"
    S_file_name = os.path.join(output_dir, base_filename + '_S.txt')
    f_file_name = os.path.join(output_dir, base_filename + '_f.txt')
    t_file_name = os.path.join(output_dir, base_filename + '_t.txt')
    
    # If the output files already exist, load and return them.
    if os.path.exists(S_file_name) and os.path.exists(f_file_name) and os.path.exists(t_file_name):
        S = np.loadtxt(S_file_name)
        f = np.loadtxt(f_file_name)
        t = np.loadtxt(t_file_name)
        return S, f, t
    
    # Otherwise, prepare to run MATLAB.
    # Save the data to a temporary file.
    seg_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    seg_file_name = seg_file.name
    np.savetxt(seg_file_name, data, fmt='%.6f')
    seg_file.close()
    
    # Create a temporary MATLAB script that:
    #   - Adds the required path.
    #   - Loads the data.
    #   - Calls mtcsg with the given parameters.
    #   - Saves S, f, and t to the designated files.
    matlab_script = f"""
addpath(genpath('/Users/katie/likhtik/software'));
data = load('{seg_file_name}');
[S, f, t] = mtcsg(data, {param1}, {param2}, {param3}, {param4}, {param5});
save('-ascii','{S_file_name}','S');
save('-ascii','{f_file_name}','f');
save('-ascii','{t_file_name}','t');
exit;
"""
    script_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.m')
    script_file_name = script_file.name
    script_file.write(matlab_script)
    script_file.close()
    
    # Call MATLAB to run the script.
    cmd = [matlab_path, "-nodisplay", "-nosplash", "-r", f"run('{os.path.abspath(script_file_name)}')"]
    subprocess.run(cmd, check=True)
    
    # Load the results.
    S = np.loadtxt(S_file_name)
    f = np.loadtxt(f_file_name)
    t = np.loadtxt(t_file_name)
    
    # (Optionally, remove the temporary files for the data and MATLAB script.)
    # os.remove(seg_file_name)
    # os.remove(script_file_name)
    
    return S, f, t
# =============================================================================
# Function to process one event's mtcsg output into an averaged pip spectrogram
# =============================================================================

def process_event_segment(S, f, t, discard_start=0.65, discard_end=30.75):
    """
    Processes one event's spectrogram (from mtcsg) as follows:
      1. Restrict the time axis to [discard_start, discard_end].
      2. Define pip onsets such that for each pip we extract the window 
         [pip_onset - 0.1, pip_onset + 0.3].  
         Here, the real pip onset is taken as 0.75 seconds (the first pip), 
         so the first extraction window is [0.65, 1.05] s.
         Pip onsets are then every 1 second, i.e. 0.75, 1.75, …, 29.75 s.
      3. For each pip, extract the corresponding slice of S.
      4. Average across all pips to yield an event spectrogram.
      5. Restrict the frequency axis to 0–20 Hz.
      6. Compute a bar value by averaging the event spectrogram in the theta band (4–8 Hz).
    
    Returns a tuple: (event_spectrogram, rel_time, f_restricted, bar_value)
      - event_spectrogram: 2D array (n_freq x n_time) where time is relative (should span –0.1 to +0.3 s)
      - rel_time: 1D array for the relative time axis (from –0.1 to +0.3 s)
      - f_restricted: 1D array of frequencies (0–20 Hz)
      - bar_value: scalar average over the theta band (4–8 Hz) in the event spectrogram.
    """
    # Restrict t and S to the desired overall window.
    valid_time_inds = (t >= discard_start) & (t <= discard_end)
    t_valid = t[valid_time_inds]
    S_valid = S[:, valid_time_inds]
    
    # Define pip onsets.
    # Now the first pip onset is at 0.75 s (so the window is [0.75-0.1, 0.75+0.3] = [0.65, 1.05]).
    pip_onsets = np.arange(0.75, 30.75, 1)  # yields 30 pips: 0.75, 1.75, ..., 29.75
    
    pip_slices = []
    rel_time_common = None  # to hold the relative time axis from the first pip
    
    for onset in pip_onsets:
        # For each pip, extract S for times between (onset - 0.1) and (onset + 0.3).
        win_inds = np.where((t_valid >= onset - 0.1) & (t_valid <= onset + 0.3))[0]
        if win_inds.size == 0:
            continue  # skip if no data
        S_pip = S_valid[:, win_inds]
        t_pip = t_valid[win_inds]
        # Define a relative time axis (relative to this pip onset)
        rel_time = t_pip - onset
        if rel_time_common is None:
            rel_time_common = rel_time  # assume all pips yield the same grid
        pip_slices.append(S_pip)
    
    if len(pip_slices) == 0:
        raise ValueError("No pip slices extracted for event.")
    
    # Average across pips.
    pip_stack = np.stack(pip_slices, axis=0)  # shape: (n_pips, n_freq, n_time)
    event_S = np.mean(pip_stack, axis=0)  # shape: (n_freq, n_time)
    
    # Restrict frequency axis to 0–20 Hz.
    freq_inds = np.where((f >= 0) & (f <= 20))[0]
    event_S_restricted = event_S[freq_inds, :]
    f_restricted = f[freq_inds]
    
    # Compute bar value: average power over theta band (4–8 Hz)
    theta_inds = np.where((f_restricted >= 3.8) & (f_restricted <= 8.2))[0]
    bar_value = np.mean(event_S_restricted[theta_inds, :])
    
    return event_S_restricted, rel_time_common, f_restricted, bar_value


# =============================================================================
# Plotting functions
# =============================================================================



def plot_evoked_heatmaps(group_means, brain_region):
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
    Only the bottom row subplots show an x-axis label ("Time (s)") and only the leftmost column shows
    a y-axis label ("Frequency (Hz)").
    
    Parameters:
      group_means: dict with keys such as "male_discriminator", "male_generalizer", "male_bad_learner",
                   "female_discriminator", "female_generalizer", "female_bad_learner". For each group,
                   group_means[group] is a dict with keys "tone_plus" and "tone_minus" (and possibly "pretone").
                   For tone conditions, each entry should contain:
                       - "mean_spec": 2D evoked spectrogram (n_freq x n_time)
                       - "rel_time": 1D relative time axis (in seconds)
                       - "f": 1D frequency vector (in Hz, already restricted to 0–20 Hz)
      brain_region: string (e.g., 'vhip') to be included in the overall title.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Define ordering.
    learning_styles = ['discriminator', 'generalizer', 'bad_learner']
    sexes = ['male', 'female']
    stimulus_types = ['tone_plus', 'tone_minus']  # CS+ and CS–

    # Layout: 3 rows (learning styles) x 4 columns (male: CS+, CS–; female: CS+, CS–).
    nrows = len(learning_styles)
    ncols = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 12))
    
    # For each learning style and sex pair, compute a common color scale (vmin, vmax) from both conditions.
    pair_scales = {}  # keys: (learning, sex)
    for learning in learning_styles:
        for sex in sexes:
            group_key = f"{sex}_{learning}"
            pair_vmin = np.inf
            pair_vmax = -np.inf
            for stim in stimulus_types:
                if group_key in group_means and stim in group_means[group_key]:
                    data = group_means[group_key][stim]
                    spec = data.get('mean_spec', None)
                    if spec is not None:
                        pair_vmin = min(pair_vmin, np.min(spec))
                        pair_vmax = max(pair_vmax, np.max(spec))
            if pair_vmin == np.inf or pair_vmax == -np.inf:
                pair_scales[(learning, sex)] = (None, None)
            else:
                pair_scales[(learning, sex)] = (pair_vmin, pair_vmax)

    # Loop over rows (learning styles) and columns.
    # The column order: 
    #   col0: male tone_plus, col1: male tone_minus, col2: female tone_plus, col3: female tone_minus.
    for row, learning in enumerate(learning_styles):
        for col in range(ncols):
            ax = axs[row, col]
            if col < 2:
                sex = 'male'
            else:
                sex = 'female'
            stim = 'tone_plus' if (col % 2 == 0) else 'tone_minus'
            group_key = f"{sex}_{learning}"
            title_str = f"{group_key.capitalize()} - {'CS+' if stim=='tone_plus' else 'CS-'}"
            vmin, vmax = pair_scales.get((learning, sex), (None, None))
            
            if group_key not in group_means or stim not in group_means[group_key]:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14)
                ax.set_title(title_str + "\n(n/a)")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 20)
            else:
                data = group_means[group_key][stim]
                spec = data.get('mean_spec', None)
                if spec is None:
                    ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                            ha="center", va="center", fontsize=14)
                    ax.set_title(title_str + "\n(n/a)")
                else:
                    rel_time = data['rel_time']
                    f_vec = data['f']
                    if vmin is not None and vmax is not None:
                        im = ax.imshow(spec, aspect='auto', origin='lower',
                                       extent=[rel_time[0], rel_time[-1], f_vec[0], f_vec[-1]],
                                       cmap='jet', vmin=vmin, vmax=vmax)
                    else:
                        im = ax.imshow(spec, aspect='auto', origin='lower',
                                       extent=[rel_time[0], rel_time[-1], f_vec[0], f_vec[-1]],
                                       cmap='jet')
                    ax.set_title(title_str)
                    # Overlay a vertical translucent gray patch from 0 to 0.05 s.
                    ax.axvspan(0, 0.05, color='gray', alpha=0.5)
                    fig.colorbar(im, ax=ax)
            # Only display x-axis label on bottom row.
            if row == nrows - 1:
                ax.set_xlabel("Time (s)")
            else:
                ax.set_xticklabels([])
            # Only display y-axis label on leftmost column.
            if col == 0:
                ax.set_ylabel("Frequency (Hz)")
            else:
                ax.set_yticklabels([])
    
    fig.suptitle(f"{brain_region.upper()} Evoked Heat Maps by Sex, Learning, and CS Condition", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =============================================================================
# Main processing function: iterate over animals and events, build data structures
# =============================================================================

def main():
    # Parameters and directories.
    main_dir = '/Users/katie/likhtik/AS'
    valid_animals = ['As105','As106','As107','As108','As110','As111','As112','As113']
    matlab_path = '/Applications/MATLAB_R2022a.app/bin/matlab'
    brain_region = 'vhip'
    
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

    animal_info ={
        'As105': {'sex': 'male', 'learning': 'discriminator'},
        'As106': {'sex': 'male', 'learning': 'bad_learner'},
        'As107': {'sex': 'male', 'learning': 'generalizer'},
        'As108': {'sex': 'male', 'learning': 'generalizer'},
        'As110': {'sex': 'female', 'learning': 'bad_learner'},
        'As111': {'sex': 'female', 'learning': 'generalizer'},
        'As112': {'sex': 'female', 'learning': 'generalizer'},
        'As113': {'sex': 'female', 'learning': 'generalizer'},


    }
    
    # Event parameters.
    tone_on_code = 65503
    expectedToneEvents = 12  # per animal
    
    fs = 2000  # sampling rate (Hz)
    # For mtcsg output, we use a discard window.
    # Now set discard_start to 0.65 so that for the first pip (onset 0.75) we can get data from 0.65.
    discard_start = 0.65
    discard_end   = 30.75
    
    # Data structure: keys for conditions; each will be a list of event dictionaries.
    conditions = ['tone_plus', 'tone_minus', 'pretone_plus', 'pretone_minus', 'pretone']

    animals_data = {}
    
    # Loop over animal subdirectories.
    for animal in valid_animals:
        data_by_within_subject_condition = {cond: [] for cond in conditions}

        animal_path = os.path.join(main_dir, animal)
        if not os.path.isdir(animal_path):
            continue
        print(f"Processing animal: {animal}")
        
        ns3_file = os.path.join(animal_path, "Testing.ns3")
        nev_file = os.path.join(animal_path, "Testing.nev")
        temp_mat_file = os.path.abspath(f"tempdata_{animal}.mat")
        
        # Use MATLAB to load NS3 and NEV.
        try:
            run_matlab_load(ns3_file, nev_file, temp_mat_file, matlab_path)
        except Exception as e:
            print(f"Error loading MATLAB files for {animal}: {e}")
            continue
        
        # Load the MAT file.
        try:
            mat_data = scipy.io.loadmat(temp_mat_file, struct_as_record=False, squeeze_me=True)
            os.remove(temp_mat_file)
        except Exception as e:
            print(f"Error loading MAT file for {animal}: {e}")
            continue
        
        NS3 = mat_data['NS3']
        NEV = mat_data['NEV']
        
        NS3_data = NS3.Data  # assume shape: (electrodes x samples)
        # Get digital IO event info.
        digital_io = NEV.Data.SerialDigitalIO
        tone_timestamps = np.atleast_1d(digital_io.TimeStampSec)
        codes = np.atleast_1d(digital_io.UnparsedData)
        
        tone_indices = np.where(codes == tone_on_code)[0]
        if len(tone_indices) != expectedToneEvents:
            print(f"Warning: {animal} has {len(tone_indices)} tone events (expected {expectedToneEvents}).")
        tone_period_times = tone_timestamps[tone_indices]
        
        # Determine electrode channel based on animal.
        mapping = get_electrode_mapping(animal)
        ch = mapping[brain_region]  
        
        # Process each tone event.
        for period_idx, tone_on in enumerate(tone_period_times):
            # Define extraction windows for tone and pretone segments.
            # Tone segment: [tone_on - 1, tone_on + 31] sec (1 sec padding on each side)
            tone_seg_start = tone_on - 1
            tone_seg_end   = tone_on + 31
            # Pretone segment: [tone_on - 32, tone_on] sec
            pretone_seg_start = tone_on - 32
            pretone_seg_end   = tone_on
            
            tone_idx_start = int(round(tone_seg_start * fs))
            tone_idx_end   = int(round(tone_seg_end * fs))
            pretone_idx_start = int(round(pretone_seg_start * fs))
            pretone_idx_end   = int(round(pretone_seg_end * fs))
            
            # Extract raw data from NS3 for this event.
            try:
                tone_data = NS3_data[ch, tone_idx_start:tone_idx_end]
                pretone_data = NS3_data[ch, pretone_idx_start:pretone_idx_end]
            except Exception as e:
                print(f"Error extracting data for {animal} event {period_idx}: {e}")
                continue
            
            # Filter and normalize.
            tone_data = divide_by_rms(filter_60_hz(tone_data, fs))
            pretone_data = divide_by_rms(filter_60_hz(pretone_data, fs))
            
            # Call MATLAB to compute mtcsg for each segment.
            # data, matlab_path,
            #          animal, period_number, period_type,
            #          param1=2048, param2=2000, param3=500, param4=480, param5=2,
            #          output_dir='/Users/katie/likhtik/data/temp'):
            try:
                S_tone, f_tone, t_tone = run_matlab_mtcsg(tone_data, matlab_path, brain_region, animal, period_idx, 'tone')
                S_pretone, f_pretone, t_pretone = run_matlab_mtcsg(pretone_data, matlab_path, brain_region, animal, period_idx, 'pretone')
            except Exception as e:
                print(f"Error computing mtcsg for {animal} event {period_idx}: {e}")
                continue
            
            # Process each event segment to get event spectrogram (averaged over pips) and bar value.
            try:
                event_tone_S, rel_time, f_restricted, tone_bar = process_event_segment(
                    S_tone, f_tone, t_tone, discard_start, discard_end)
                event_pretone_S, rel_time, f_restricted, pretone_bar = process_event_segment(
                    S_pretone, f_pretone, t_pretone, discard_start, discard_end)
            except Exception as e:
                print(f"Error processing event segmentation for {animal} event {period_idx}: {e}")
                continue
            
            # Determine condition assignment based on event number.
            cs_minus_ids = [0, 2, 4, 5, 8, 11]
            if period_idx in cs_minus_ids:
                tone_condition = 'tone_minus'
                pretone_condition = 'pretone_minus'
            else:
                tone_condition = 'tone_plus'
                pretone_condition = 'pretone_plus'
            
            # Build event data dictionaries.
            tone_event = {
                'event_spectrogram': event_tone_S,  # shape: (n_freq, n_rel_time)
                'rel_time': rel_time,
                'f': f_restricted,
                'bar_value': tone_bar,
                'animal': animal,
                'event_index': period_idx,
                'learning': learning_dict[animal],
                'sex': ('male' if animal in male_stressed else
                        'male' if animal in male_non_stressed else
                        'female' if animal in female_stressed else
                        'female' if animal in female_non_stressed else 'unknown')
            }
            pretone_event = {
                'event_spectrogram': event_pretone_S,
                'rel_time': rel_time,
                'f': f_restricted,
                'bar_value': pretone_bar,
                'animal': animal,
                'event_index': period_idx,
                'learning': learning_dict[animal],
                'sex': ('male' if animal in male_stressed else
                        'male' if animal in male_non_stressed else
                        'female' if animal in female_stressed else
                        'female' if animal in female_non_stressed else 'unknown')
            }
            
            # Append event data to the appropriate condition lists.
            data_by_within_subject_condition[tone_condition].append(tone_event)
            data_by_within_subject_condition[pretone_condition].append(pretone_event)
            data_by_within_subject_condition['pretone'].append(pretone_event)
        
        animals_data[animal] = data_by_within_subject_condition
    
    # =============================================================================
    # Compute group-level statistics.
    # For each condition, compute mean and SEM for the bar values and the event spectrograms.
    # =============================================================================
    # First, accumulate the per-animal means in a dictionary keyed by animal.
    all_animal_means = {}   # key: animal; value: animal_means dictionary
    for animal, data_by_within_subject_condition in animals_data.items():
        animal_means = {}
        for cond in conditions:
            events = data_by_within_subject_condition[cond]
            if len(events) == 0:
                continue
            # Gather bar values and compute their mean.
            bar_vals = np.array([ev['bar_value'] for ev in events])
            mean_bar = np.mean(bar_vals)
            # Stack event spectrograms and compute the mean.
            spec_stack = np.stack([ev['event_spectrogram'] for ev in events], axis=0)
            mean_spec = np.mean(spec_stack, axis=0)
            animal_means[cond] = {
                'mean_bar': mean_bar,
                'mean_spec': mean_spec,
                'rel_time': events[0]['rel_time'],  # assume same for all events
                'f': events[0]['f']
            }
        # Now, for the tone conditions, compute the evoked measures by subtracting the pretone baseline.
        # (Here we assume that the pretone condition key is simply 'pretone'.)
        for cond in ['tone_plus', 'tone_minus']:
            evoked_bar = animal_means[cond]['mean_bar'] - animal_means['pretone']['mean_bar']
            # NOTE: In your snippet you had:
            #    evoked_spec = animal_means[cond]['mean_spec'] - animal_means['pretone']['evoked_spec']
            # If you want the evoked spectrogram to be (tone_spec - pretone_spec), it might be:
            evoked_spec = animal_means[cond]['mean_spec'] - animal_means['pretone']['mean_spec']
            animal_means[cond]['evoked_bar'] = evoked_bar
            animal_means[cond]['evoked_spec'] = evoked_spec
        all_animal_means[animal] = animal_means

    # Next, group the animals by their between-subject factors.
    # We'll build a dictionary keyed by group name, e.g. "male_bad_learner", "male_generalizer", etc.
    group_animals = {}
    for animal, animal_means in all_animal_means.items():
        # Get group membership for this animal.
        info = animal_info[animal]  # animal_info must be defined elsewhere
        group_key = f"{info['sex']}_{info['learning']}"
        if group_key not in group_animals:
            group_animals[group_key] = []
        group_animals[group_key].append(animal_means)

    # Now compute group-level means and standard deviations.
    # We will compute these for each condition and for the evoked measures.
    group_means = {}
    for group_key, animal_means_list in group_animals.items():
        group_means[group_key] = {}
        # For each condition, we collect the evoked_bar values and evoked_spec matrices across animals.
        for cond in ['tone_plus', 'tone_minus', 'pretone']:
            # Initialize lists to accumulate values.
            bar_vals = []   # scalar values from each animal
            spec_vals = []  # spectrogram matrices from each animal
            rel_time = None
            f_vec = None
            # Iterate over animals in the group.
            for am in animal_means_list:
                if cond not in am:
                    continue
                # For tone conditions, we use the evoked measure.
                if cond in ['tone_plus', 'tone_minus']:
                    bar_vals.append(am[cond]['evoked_bar'])
                    spec_vals.append(am[cond]['evoked_spec'])
                else:
                    # For pretone, just use the raw mean.
                    bar_vals.append(am[cond]['mean_bar'])
                    spec_vals.append(am[cond]['mean_spec'])
                # Record time and frequency axes (assumed the same for all animals).
                if rel_time is None:
                    rel_time = am[cond]['rel_time']
                if f_vec is None:
                    f_vec = am[cond]['f']
            if len(bar_vals) == 0:
                continue
            # Compute the group mean and standard deviation for bar values.
            bar_vals = np.array(bar_vals)
            mean_bar = np.mean(bar_vals)
            std_bar = np.std(bar_vals)
            # Stack the spectrograms and compute the elementwise mean and standard deviation.
            spec_stack = np.stack(spec_vals, axis=0)  # shape: (n_animals, n_freq, n_time)
            mean_spec = np.mean(spec_stack, axis=0)
            std_spec = np.std(spec_stack, axis=0)
            # Save these values in the group_means dictionary.
            group_means[group_key][cond] = {
                'mean_bar': mean_bar,
                'std_bar': std_bar,
                'mean_spec': mean_spec,
                'std_spec': std_spec,
                'rel_time': rel_time,
                'f': f_vec
            }

    #plot_evoked_bar_graph(group_means, brain_region)
    plot_evoked_heatmaps(group_means, brain_region)




def plot_evoked_heatmaps(group_means, brain_region):
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
    Only the bottom row subplots show an x-axis label ("Time (s)") and only the leftmost column shows
    a y-axis label ("Frequency (Hz)").
    
    Parameters:
      group_means: dict with keys such as "male_discriminator", "male_generalizer", "male_bad_learner",
                   "female_discriminator", "female_generalizer", "female_bad_learner". For each group,
                   group_means[group] is a dict with keys "tone_plus" and "tone_minus" (and possibly "pretone").
                   For tone conditions, each entry should contain:
                       - "mean_spec": 2D evoked spectrogram (n_freq x n_time)
                       - "rel_time": 1D relative time axis (in seconds)
                       - "f": 1D frequency vector (in Hz, already restricted to 0–20 Hz)
      brain_region: string (e.g., 'vhip') to be included in the overall title.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Define ordering.
    learning_styles = ['discriminator', 'generalizer', 'bad_learner']
    sexes = ['male', 'female']
    stimulus_types = ['tone_plus', 'tone_minus']  # CS+ and CS–

    # Layout: 3 rows (learning styles) x 4 columns (male: CS+, CS–; female: CS+, CS–).
    nrows = len(learning_styles)
    ncols = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 12))
    
    # For each learning style and sex pair, compute a common color scale (vmin, vmax) from both conditions.
    pair_scales = {}  # keys: (learning, sex)
    for learning in learning_styles:
        for sex in sexes:
            group_key = f"{sex}_{learning}"
            pair_vmin = np.inf
            pair_vmax = -np.inf
            for stim in stimulus_types:
                if group_key in group_means and stim in group_means[group_key]:
                    data = group_means[group_key][stim]
                    spec = data.get('mean_spec', None)
                    if spec is not None:
                        pair_vmin = min(pair_vmin, np.min(spec))
                        pair_vmax = max(pair_vmax, np.max(spec))
            if pair_vmin == np.inf or pair_vmax == -np.inf:
                pair_scales[(learning, sex)] = (None, None)
            else:
                pair_scales[(learning, sex)] = (pair_vmin, pair_vmax)

    # Loop over rows (learning styles) and columns.
    # The column order: 
    #   col0: male tone_plus, col1: male tone_minus, col2: female tone_plus, col3: female tone_minus.
    for row, learning in enumerate(learning_styles):
        for col in range(ncols):
            ax = axs[row, col]
            if col < 2:
                sex = 'male'
            else:
                sex = 'female'
            stim = 'tone_plus' if (col % 2 == 0) else 'tone_minus'
            group_key = f"{sex}_{learning}"
            title_str = f"{group_key.capitalize()} - {'CS+' if stim=='tone_plus' else 'CS-'}"
            vmin, vmax = pair_scales.get((learning, sex), (None, None))
            
            if group_key not in group_means or stim not in group_means[group_key]:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14)
                ax.set_title(title_str + "\n(n/a)")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 20)
            else:
                data = group_means[group_key][stim]
                spec = data.get('mean_spec', None)
                if spec is None:
                    ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                            ha="center", va="center", fontsize=14)
                    ax.set_title(title_str + "\n(n/a)")
                else:
                    rel_time = data['rel_time']
                    f_vec = data['f']
                    if vmin is not None and vmax is not None:
                        im = ax.imshow(spec, aspect='auto', origin='lower',
                                       extent=[rel_time[0], rel_time[-1], f_vec[0], f_vec[-1]],
                                       cmap='jet', vmin=vmin, vmax=vmax)
                    else:
                        im = ax.imshow(spec, aspect='auto', origin='lower',
                                       extent=[rel_time[0], rel_time[-1], f_vec[0], f_vec[-1]],
                                       cmap='jet')
                    ax.set_title(title_str)
                    # Overlay a vertical translucent gray patch from 0 to 0.05 s.
                    ax.axvspan(0, 0.05, color='gray', alpha=0.5)
                    fig.colorbar(im, ax=ax)
            # Only display x-axis label on bottom row.
            if row == nrows - 1:
                ax.set_xlabel("Time (s)")
            else:
                ax.set_xticklabels([])
            # Only display y-axis label on leftmost column.
            if col == 0:
                ax.set_ylabel("Frequency (Hz)")
            else:
                ax.set_yticklabels([])
    
    fig.suptitle(f"{brain_region.upper()} Evoked Heat Maps by Sex, Learning, and CS Condition", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
# =============================================================================
# Run the main processing
# =============================================================================

if __name__ == "__main__":
    main()