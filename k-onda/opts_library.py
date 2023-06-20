
PSTH_OPTS = {'data_type': 'psth', 'pre_stim': 0.05, 'post_stim': 0.65, 'bin_size': 0.01, 'trials': (0, 150)}

AUTOCORR_OPTS = {'data_type': 'autocorr', 'pre_stim': 0.0, 'post_stim': 30.0, 'bin_size': 0.01, 'trials': (0, 150, 30),
                 'max_lag': 99}

SPECTRUM_OPTS = {'data_type': 'spectrum', 'pre_stim': 0.0, 'post_stim': 30.0, 'bin_size': 0.01, 'trials': (0, 150, 30),
                 'max_lag': 99, 'freq_range': (3, 60)}

SPREADSHEET_OPTS = {'path': '/Users/katie/likhtik/data/firing_rates.csv', 'data_type': 'psth', 'time': 'continuous',
                    'pre_stim': 0.0, 'post_stim': 1.00, 'bin_size': 0.01, 'trials': (0, 150)}

GRAPH_OPTS = {'graph_dir': '/Users/katie/likhtik/data/graphs', 'units_in_fig': 4, 'tick_step': 0.1, 'sem': True,
              'footer': True, 'equal_y_scales': True}

AC_KEYS = {
    'group':  ['group_by_animal_by_unit_by_trials', 'group_by_animal_by_unit_by_rates', 'group_by_animal_by_rates',
               'group_by_rates'],
    'animal': ['animal_by_unit_by_trials', 'animal_by_unit_by_rates', 'animal_by_rates'],
    'unit': ['unit_by_trials', 'unit_by_rates']
}

ALL_AC_METHODS = ['np', 'ml', 'pd']

AC_METHODS = ['np']
