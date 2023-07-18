
PSTH_OPTS = {'data_type': 'psth', 'pre_stim': 0.05, 'post_stim': 0.65, 'bin_size': 0.01, 'trials': (0, 150),
             'adjustment': 'normalized', 'average_method': 'mean', 'base': ''}

PROPORTION_OPTS = {'data_type': 'proportion', 'pre_stim': 0.05, 'post_stim': 0.65, 'bin_size': 0.01,
                   'trials': (0, 150), 'adjustment': 'normalized', 'base': 'trial'}

AUTOCORR_OPTS = {'data_type': 'autocorr', 'pre_stim': 0.0, 'post_stim': 30.0, 'bin_size': 0.01, 'trials': (0, 150, 30),
                 'max_lag': 99}

SPECTRUM_OPTS = {'data_type': 'spectrum', 'pre_stim': 0.0, 'post_stim': 30.0, 'bin_size': 0.01, 'trials': (0, 150, 30),
                 'max_lag': 99, 'freq_range': (3, 60)}

SPREADSHEET_OPTS = {'path': '/Users/katie/likhtik/data', 'data_type': 'psth', 'base': 'trial',
                    'adjustment': 'normalized', 'time': 'continuous', 'pre_stim': 0.0, 'post_stim': .70,
                    'bin_size': 0.01, 'trials': (0, 150), 'row_type': 'trial', 'num_bins': 70}

GROUP_STAT_PROPORTION_OPTS = {'data_type': 'proportion', 'base': 'trial', 'adjustment': 'normalized',
                              'time': 'continuous', 'pre_stim': 0.0, 'post_stim': 0.70, 'bin_size': 0.01,
                              'trials': (0, 150), 'row_type': 'unit', 'post_hoc_bin_size': 1,
                              'path': '/Users/katie/likhtik/data', 'post_hoc_type': 'beta',
                              'group_colors': {'control': '#76BD4E', 'stressed': '#F2A354'}}

GROUP_STAT_PSTH_OPTS = {'data_type': 'psth', 'adjustment': 'normalized', 'time': 'continuous', 'pre_stim': 0.0,
                        'post_stim': 0.70, 'bin_size': 0.01, 'trials': (0, 150), 'row_type': 'trial',
                        'post_hoc_bin_size': 1, 'path': '/Users/katie/likhtik/data', 'post_hoc_type': 'lmer',
                        'group_colors': {'control': '#76BD4E', 'stressed': '#F2A354'}}

GRAPH_OPTS = {'graph_dir': '/Users/katie/likhtik/data/graphs', 'units_in_fig': 4, 'tick_step': 0.1, 'sem': True,
              'footer': True, 'equal_y_scales': True, 'group_colors': {'control': '#76BD4E', 'stressed': '#F2A354'},
              'force_recalc': False}

FIGURE_1_OPTS = {'data_path': '/Users/katie/likhtik/data/single_cell_data', 'animal_id': 'IG156',
                 'cluster_ids': [21, 27], 'electrodes_for_waveform': [[9, 11], [8]], 'electrodes_for_feature': [1, 8],
                 'el_inds': [1, 1], 'pc_inds': [1, 0], 'sem': False, 'equal_y_scales': True, 'tick_step': 0.2,
                 'neuron_type_colors': {'IN': '#5679C7', 'PN': '#C75B56'}, 'annot_coords': (-0.11, 1.1),
                 'group_colors': {'control': '#76BD4E', 'stressed': '#F2A354'}, 'hist_color': '#9678D3', 'force_recalc':
                 False}

AC_KEYS = {
    'group':  ['group_by_animal_by_unit_by_trials', 'group_by_animal_by_unit_by_rates', 'group_by_animal_by_rates',
               'group_by_rates'],
    'animal': ['animal_by_unit_by_trials', 'animal_by_unit_by_rates', 'animal_by_rates'],
    'unit': ['unit_by_trials', 'unit_by_rates']
}

ALL_AC_METHODS = ['np', 'ml', 'pd']

AC_METHODS = ['np']
