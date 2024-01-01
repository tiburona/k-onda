STANDARD_ANIMALS = ['IG160', 'IG163', 'IG176', 'IG178', 'IG180', 'IG154', 'IG156', 'IG158', 'IG177', 'IG179']
HPC_REPLICATION_ANIMALS = ['IG162', 'IG171', 'IG173', 'IG176', 'IG155', 'IG174', 'IG175', 'IG179']

PSTH_OPTS = {'data_class': 'spike', 'data_type': 'psth', 'pre_stim': 0.05, 'post_stim': 0.65, 'bin_size': 0.01, 'events': (0, 300),
             'adjustment': 'normalized', 'average_method': 'mean', 'base': '', 'time_type': 'continuous',
             'data_path': '/Users/katie/likhtik/data', 'row_type': 'event', 'levels': ['unit']}

PROPORTION_OPTS = {'data_class': 'spike', 'data_type': 'proportion', 'pre_stim': 0.05, 'post_stim': 0.65,
                   'bin_size': 0.01, 'trials': (0, 150), 'adjustment': 'normalized', 'base': 'trial',
                   'time': 'continuous', 'row_type': 'trial'}

AUTOCORR_OPTS = {'data_class': 'spike', 'data_type': 'autocorr', 'pre_stim': 0.0, 'post_stim': 30.0, 'bin_size': 0.01,
                 'tone_events': (0, 300, 30), 'pretone_events': (0, 300, 30), 'max_lag': 99, 'block_types': ['tone']}

SPECTRUM_OPTS = {'data_class': 'spike', 'data_type': 'spectrum', 'pre_stim': 0.0, 'post_stim': 30.0, 'bin_size': 0.01,
                 'tone_events': (0, 300, 30), 'pretone_events': (0, 300, 30), 'max_lag': 99, 'freq_range': (3, 60),
                 'block_types': ['tone']}

SPREADSHEET_OPTS = {'data_class': 'spike', 'data_path': '/Users/katie/likhtik/data', 'data_type': 'psth', 'base': 'trial',
                    'pre_stim': 0.0, 'post_stim': .05, 'adjustment': 'none', 'bin_size': 0.01, 'trials': (0, 150),
                    'row_type': 'trial', 'periods': list(range(5)), 'period_types': ['pretone', 'tone'],
                    'selected_animals': STANDARD_ANIMALS, 'time_type': 'continuous'}

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

LFP_OPTS = {'data_class': 'lfp', 'time_type': 'block', 'frequency_bands': ['theta_1'], 'data_type': 'power',
            'brain_regions': ['bla', 'il', 'bf'],  'frequency_type': 'block', 'row_type': 'event',
            'pre_stim': 0, 'post_stim': .30, 'period_types': ['pretone', 'tone'],
            'power_deviation': False, 'wavelet': False, 'data_path': '/Users/katie/likhtik/CH_for_katie_less_conservative'}

HEAT_MAP_DATA_OPTS = {'data_class': 'lfp', 'data_path': '/Users/katie/likhtik/data', 'time_type': 'period',
                      'brain_region': 'hpc', 'fb': ['gamma'], 'frequency_type': 'continuous', 'data_type': 'mrl',
                      'row_type': 'frequency_bin', 'pre_stim': 0, 'post_stim': 0.3,
                      'trials': (0, 150), 'adjustment': 'relative', 'frequency_band': (0, 70)}


BEHAVIOR_OPTS = {'data_class': 'behavior', 'data_type': 'percent_freezing', 'row_type': 'period',
                 'period_types': ['pretone', 'tone'], 'selected_animals': STANDARD_ANIMALS,
                 'data_path': '/Users/katie/likhtik/data'}

CAROLINA_OPTS = {'data_class': 'spike', 'data_path': '/Users/katie/likhtik/CH_for_katie_less_conservative',
                 'data_type': 'psth', 'base': 'trial', 'pre_stim': .05, 'post_stim': .65, 'adjustment': 'normalized',
                 'bin_size': 0.01, 'events': (0, 300), 'row_type': 'trial', 'periods': list(range(5)),
                 'period_types': ['pretone', 'tone']}


CAROLINA_GRAPH_OPTS = {'graph_dir': '/Users/katie/likhtik/CH_for_katie_less_conservative/graphs', 'units_in_fig': 4,
                       'tick_step': .1, 'sem': False, 'footer': False, 'equal_y_scales': True,
                       'group_colors': {'control': '#9F9FA3', 'arch': '#32B44A'}, 'force_recalc': False,
                       'neuron_type_colors': {'PV': '#5679C7', 'ACH': '#C75B56'}, 'animal_id': 'CH272',
                       'cluster_ids': [10, 101], 'electrodes_for_waveform': [[7], [3, 5]],
                       'electrodes_for_feature': [13, 15], 'el_inds': [1, 1], 'pc_inds': [0, 1],
                       'annot_coords': (-0.11, 1.1), 'data_path': '/Users/katie/likhtik/CH_for_katie_less_conservative',
                       'normalize_waveform': True, 'block_order': ['pretone', 'tone']}

CAROLINA_GROUP_STAT_OPTS = {'data_class': 'spike', 'data_type': 'psth', 'adjustment': 'normalized',
                            'time_type': 'continuous', 'pre_stim': 0.0, 'post_stim': 0.70, 'bin_size': 0.01,
                            'trials': (0, 150), 'row_type': 'trial', 'post_hoc_bin_size': 1,
                            'data_path': '/Users/katie/likhtik/data', 'post_hoc_type': 'lmer', 'group_colors':
                                {'control': '#76BD4E', 'arch': '#F2A354'}, 'sem_level': 'unit'}

SPONTANEOUS_OPTS = {'data_class': 'spike', 'data_path': '/Users/katie/likhtik/CH_for_katie_less_conservative',
                    'data_type': 'spontaneous_firing', 'bin_size': 0.1, 'selected_animals': ['CH272', 'CH274', 'CH275'],
                    'spontaneous': 120}


SPONTANEOUS_GRAPH_OPTS = {'graph_dir': '/Users/katie/likhtik/CH_for_katie_less_conservative/graphs', 'units_in_fig': 4, 'tick_step': 100,
                          'sem': False, 'footer': False, 'equal_y_scales': True,
                          'group_colors': {'control': '#9F9FA3', 'arch': '#32B44A'}, 'force_recalc': False,
                          'neuron_type_colors': {'PV': '#5679C7', 'ACH': '#C75B56'}}


CROSS_CORR_OPTS = {'data_class': 'spike', 'data_type': 'correlogram', 'pre_stim': 0, 'post_stim': 1,
                   'adjustment': 'none', 'bin_size': 0.001, 'events': (0, 300), 'periods': list(range(10)),
                   'period_types': ['pretone', 'tone'], 'unit_pairs': ['ACH,PV'], 'max_lag': .05}

SPONTANEOUS_MRL_OPTS = {'data_class': 'lfp', 'data_type': 'mrl', 'bin_size': 0.01, 'spontaneous': (0, 120),
                        'events': (0, 300),  'frequency_bands': ['theta_1'], 'brain_regions': ['bla', 'il', 'bf'],
                        'sem_level': 'mrl_calculator'}

CAROLINA_MRL_OPTS = {'data_class': 'lfp', 'data_type': 'mrl', 'bin_size': 0.01, 'pre_stim': 0, 'post_stim': .3,
                     'events': (0, 300),  'frequency_bands': ['theta_1'], 'brain_regions': ['bla', 'il', 'bf'],
                     'sem_level': 'mrl_calculator'}


TEST_RUNNER_OPTS = {'data_opts': PSTH_OPTS, 'graph_opts': CAROLINA_GRAPH_OPTS}

TEST_LFP_OPTS = {'data_class': 'lfp', 'time_type': 'block', 'frequency_bands': ['theta_1'], 'data_type': 'power',
            'brain_regions': ['bla', 'il', 'bf'],  'frequency_type': 'block', 'row_type': 'event',
            'pre_stim': 0, 'post_stim': .30, 'power_deviation': False, 'wavelet': False,
                 'matlab_configuration': {'matlab_path': '/Applications/MATLAB_R2022a.app/bin/matlab'}}


