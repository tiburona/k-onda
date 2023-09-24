
PSTH_OPTS = {'data_class': 'spike', 'data_type': 'psth', 'pre_stim': 0.05, 'post_stim': 0.65, 'bin_size': 0.01, 'trials': (0, 150),
             'adjustment': 'normalized', 'average_method': 'mean', 'base': '', 'time_type': 'continuous',
             'data_path': '/Users/katie/likhtik/data', 'row_type': 'trial'}

PROPORTION_OPTS = {'data_class': 'spike', 'data_type': 'proportion', 'pre_stim': 0.05, 'post_stim': 0.65, 'bin_size': 0.01,
                   'trials': (0, 150), 'adjustment': 'normalized', 'base': 'trial', 'time': 'continuous',
                       'row_type': 'trial'}

AUTOCORR_OPTS = {'data_class': 'spike', 'data_type': 'autocorr', 'pre_stim': 0.0, 'post_stim': 30.0, 'bin_size': 0.01, 'trials': (0, 150, 30),
                 'max_lag': 99}

SPECTRUM_OPTS = {'data_class': 'spike', 'data_type': 'spectrum', 'pre_stim': 0.0, 'post_stim': 30.0, 'bin_size': 0.01, 'trials': (0, 150, 30),
                 'max_lag': 99, 'freq_range': (3, 60)}

SPREADSHEET_OPTS = {'data_class': 'spike', 'data_path': '/Users/katie/likhtik/data', 'data_type': 'psth', 'base': 'trial',
                    'pre_stim': 0.0, 'post_stim': .65, 'adjustment': 'none',
                    'bin_size': 0.01, 'trials': (0, 150), 'row_type': 'period', 'pretone_trials': True,
                    'periods': list(range(5)), 'period_types': ['pretone', 'tone']}

GROUP_STAT_PROPORTION_OPTS = {'data_class': 'spike', 'data_type': 'proportion', 'base': 'trial', 'adjustment': 'normalized',
                              'time_type': 'continuous', 'pre_stim': 0.0, 'post_stim': 0.70, 'bin_size': 0.01,
                              'trials': (0, 150), 'row_type': 'unit', 'post_hoc_bin_size': 1,
                              'data_path': '/Users/katie/likhtik/data', 'post_hoc_type': 'beta',
                              'group_colors': {'control': '#76BD4E', 'stressed': '#F2A354'}}

GROUP_STAT_PSTH_OPTS = {'data_class': 'spike', 'data_type': 'psth', 'adjustment': 'normalized', 'time_type': 'continuous', 'pre_stim': 0.0,
                        'post_stim': 0.70, 'bin_size': 0.01, 'trials': (0, 150), 'row_type': 'trial',
                        'post_hoc_bin_size': 1, 'data_path': '/Users/katie/likhtik/data', 'post_hoc_type': 'lmer',
                        'group_colors': {'control': '#76BD4E', 'stressed': '#F2A354'}}

GRAPH_OPTS = {'graph_dir': '/Users/katie/likhtik/data/graphs', 'units_in_fig': 4, 'tick_step': 0.1, 'sem': False,
              'footer': True, 'equal_y_scales': True, 'group_colors': {'control': '#76BD4E', 'stressed': '#F2A354'},
              'force_recalc': False}

ROSE_PLOT_OPTS = {'graph_dir': '/Users/katie/likhtik/data/graphs', 'units_in_fig': 4, 'sem': False,
              'footer': False, 'equal_y_scales': True, 'group_colors': {'control': '#76BD4E', 'stressed': '#F2A354'},
              'force_recalc': False, 'superimpose': False}


FIGURE_1_OPTS = {'data_class': 'spike', 'data_path': '/Users/katie/likhtik/data/single_cell_data', 'animal_id': 'IG156',
                 'cluster_ids': [21, 27], 'electrodes_for_waveform': [[9, 11], [8]], 'electrodes_for_feature': [1, 8],
                 'el_inds': [1, 1], 'pc_inds': [1, 0], 'sem': False, 'equal_y_scales': True, 'tick_step': 0.2,
                 'neuron_type_colors': {'IN': '#5679C7', 'PN': '#C75B56'}, 'annot_coords': (-0.11, 1.1),
                 'group_colors': {'control': '#76BD4E', 'stressed': '#F2A354'}, 'hist_color': '#9678D3', 'force_recalc':
                 False}

LFP_OPTS = {'data_class': 'lfp', 'data_path': '/Users/katie/likhtik/data', 'time_type': 'period', 'brain_region': 'hpc',
            'frequency_type': 'block', 'data_type': 'mrl', 'fb': ['theta_1', 'theta_2'],
            'row_type': 'period', 'pretone_trials': True, 'pre_stim': 0, 'post_stim': 0.3, 'trials': (0, 150),
            'frequency_band': 'theta_1', 'period_types': ['pretone', 'tone']}

HEAT_MAP_DATA_OPTS = {'data_class': 'lfp', 'data_path': '/Users/katie/likhtik/data', 'time_type': 'period',
                      'brain_region': 'hpc', 'fb': ['gamma'], 'frequency_type': 'continuous', 'data_type': 'mrl',
                      'row_type': 'frequency_bin', 'pretone_trials': True, 'pre_stim': 0, 'post_stim': 0.3,
                      'trials': (0, 150), 'adjustment': 'relative', 'frequency_band': (0, 70)}


BEHAVIOR_OPTS = {'data_class': 'behavior', 'data_type': 'percent_freezing', 'row_type': 'period',
                 'period_types': ['pretone', 'tone']}

AC_KEYS = {
    'group':  ['group_by_animal_by_unit_by_trials', 'group_by_animal_by_unit_by_rates', 'group_by_animal_by_rates',
               'group_by_rates'],
    'animal': ['animal_by_unit_by_trials', 'animal_by_unit_by_rates', 'animal_by_rates'],
    'unit': ['unit_by_trials', 'unit_by_rates']
}

ALL_AC_METHODS = ['np', 'ml', 'pd']

AC_METHODS = ['np']

