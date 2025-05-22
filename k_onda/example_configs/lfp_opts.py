import numpy as np

freqs = np.arange(1, 30, 1)

SPECTRUM_OPTS = {
    'kind_of_data': 'lfp', 
    'calc_type': 'power',
    'calc_method': 'mne',
    'brain_regions': ['bla'], 
    'frequency_bands': [(0, 20)], 
    'load_spectrogram': False,
    'power_arg_set': {'sfreq': 2000, 'freqs': freqs, 'decim': 20, 'n_cycles': freqs * 0.5,
                      'time_bandwidth': 2, 'output': 'power'},
    'time_type': 'continous', 
    'frequency_type': 'continuous', 
    'bin_size': .01,
    'periods': {'stim': {'event_pre_post': (0, .3)}},
    }


SPECTRUM_PLOT_SPEC = {
    'plot_type': 'peristimulus_power_spectrum',
    'section': {
        'divisions': [{'divider_type': 'period_type', 'members': ['stim']}],
        'label': {'x_bottom': {'text': 'Seconds'},
                  'y_left': {'text': 'Power'}},
        'legend': {'colorbar': {'share': 'global', 'position': 'right'}},
        'aesthetics': {'indicator': {'type': 'patch', 'when': [0, 0.05]}}
        }}


LFP_OPTS = {
    'procedure': 'make_plots',
    'io_opts': {'paths': {'out': './lfp'}, 'read_opts': {'lfp_file_load': 'neo'}},
    'calc_opts': SPECTRUM_OPTS,
    'plot_spec': SPECTRUM_PLOT_SPEC
}