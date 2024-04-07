STANDARD_ANIMALS = ['IG160', 'IG163', 'IG176', 'IG178', 'IG180', 'IG154', 'IG156', 'IG158', 'IG177', 'IG179']



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
                    'row_type': 'event', 'periods': list(range(5)), 'period_types': ['pretone', 'tone'],
                    'selected_animals': STANDARD_ANIMALS, 'time_type': 'continuous'}



GRAPH_OPTS = {'graph_dir': '/Users/katie/likhtik/data/graphs', 'units_in_fig': 4, 'tick_step': 0.1, 'sem': False,
              'footer': True, 'equal_y_scales': True, 'group_colors': {'control': '#76BD4E', 'stressed': '#F2A354'},
              'force_recalc': False}

ROSE_PLOT_OPTS = {'graph_dir': '/Users/katie/likhtik/data/graphs', 'units_in_fig': 4, 'sem': False,
              'footer': False, 'equal_y_scales': True, 'group_colors': {'control': '#76BD4E', 'stressed': '#F2A354'},
              'force_recalc': False, 'superimpose': False}


FIGURE_1_OPTS = {'data_class': 'spike', 'data_path': '/Users/katie/likhtik/IG_INED_Safety_Recall', 'animal_id': 'IG180',
                 'unit_ids': [10, 2], 'electrodes_for_waveform': [[11], [3]], 'electrodes_for_feature': [1, 3],
                 'el_inds': [0, 0], 'pc_inds': [0, 1], 'sem': False, 'equal_y_scales': True, 'tick_step': 0.2,
                 'neuron_type_colors': {'IN': '#5679C7', 'PN': '#C75B56'}, 'annot_coords': (-0.11, 1.1),
                 'group_colors': {'control': '#76BD4E', 'defeat': '#F2A354'}, 'hist_color': '#9678D3', 'force_recalc':
                 False}


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
                     'frequency_bands': ['theta_1'], 'brain_regions': ['bla', 'il', 'bf'],
                     'sem_level': 'mrl_calculator'}

# PFC_THETA_POWER_ANIMALS = [
#     'IG160', 'IG163', 'IG171', 'IG176', 'IG180', 'INED04', 'INED16', 'INED18', 'IG154', 'IG156', 'IG158', 'IG172',
#     'IG174', 'IG175', 'IG177', 'IG179', 'INED07', 'INED06', 'INED09', 'INED11', 'INED12'
# ]

PFC_THETA_POWER_ANIMALS = [
    'IG160', 'IG163', 'IG171', 'IG176', 'IG180', 'INED04', 'INED16', 'INED18', 'IG156', 'IG158', 'IG172',
    'IG174', 'IG175', 'IG177', 'IG179', 'INED07', 'INED06', 'INED09', 'INED11', 'INED12'
]

BLA_THETA_POWER_ANIMALS = [
    'IG160', 'IG161', 'IG162', 'IG163', 'IG171', 'IG173', 'IG176', 'IG178', 'IG180', 'INED04', 'INED05', 'INED17',
    'INED18', 'IG154', 'IG155', 'IG156', 'IG158', 'IG172', 'IG174', 'IG175', 'IG179', 'INED01', 'INED07', 'INED09',
    'INED12'
]

HPC_THETA_POWER_ANIMALS = ['IG162', 'IG171', 'IG173', 'IG176', 'IG155', 'IG174', 'IG175', 'IG179']

HPC_MRL_ANIMALS = list(set(HPC_THETA_POWER_ANIMALS + STANDARD_ANIMALS))


GRAPH_OPTS = {'graph_dir': '/Users/katie/likhtik/IG_INED_SAFETY_RECALL', 'units_in_fig': 4, 'tick_step': 0.1,
              'sem': False, 'footer': True, 'equal_y_scales': True, 'equal_color_scales': 'within_group',
              'group_colors': {'control': '#76BD4E', 'defeat': '#F2A354'}, 'period_colors':
                  {'pretone': 'gray', 'tone': 'black'}, 'period_order': ['pretone', 'tone']
              }

MATLAB_CONFIG = {
                'path_to_matlab': '/Applications/MATLAB_R2022a.app/bin/matlab',
                'paths_to_add': [], 'recursive_paths_to_add': ['/Users/katie/likhtik/software'],
                'base_directory': '/Users/katie/likhtik/data/temp'}

LFP_OPTS = {'data_class': 'lfp', 'time_type': 'block', 'frequency_bands': ['theta_1'], 'data_type': 'power',
            'brain_regions': ['pl', 'bla', 'hpc'],  'frequency_type': 'block', 'row_type': 'event',
            'blocks': {'tone': range(5), 'pretone': range(5)}, 'power_deviation': False, 'collapse_sem_data': True,
            'events': {'pretone': {'pre_stim': 0, 'post_stim': 1}, 'tone': {'pre_stim': 0, 'post_stim': .3}},
            'rules': {'brain_region': {'pl': [('selected_animals', PFC_THETA_POWER_ANIMALS)],
                                       'bla': [('selected_animals', BLA_THETA_POWER_ANIMALS)],
                                       'hpc': [('selected_animals', HPC_THETA_POWER_ANIMALS)]}},
            'matlab_configuration': MATLAB_CONFIG}

SPECTROGRAM_OPTS = {'data_class': 'lfp', 'time_type': 'block', 'frequency_bands': ['theta_1'], 'data_type': 'power',
                    'brain_regions': ['pl', 'bla', 'hpc'],  'evoked': True, 'power_arg_set': (2048, 2000, 1000, 980, 2),
                    'blocks': {'tone': [0]}, 'power_deviation': False, 'collapse_sem_data': True,
                    'levels': ['group', 'animal'], 'events': {'pretone': {'pre_stim': .05, 'post_stim': .3},
                                                              'tone': {'pre_stim': .05, 'post_stim': .3}},
                    'rules': {'brain_region': {'pl': [('selected_animals', PFC_THETA_POWER_ANIMALS)],
                                               'bla': [('selected_animals', BLA_THETA_POWER_ANIMALS)],
                                               'hpc': [('selected_animals', HPC_THETA_POWER_ANIMALS)]}},
                    'matlab_configuration': MATLAB_CONFIG
                    }

# MRL_OPTS = {'data_class': 'lfp', 'time_type': 'block', 'frequency_bands': ['theta_1'], 'data_type': 'mrl',
#             'brain_regions': ['pl', 'bla', 'hpc'],  'frequency_type': 'block',
#             'blocks': {'tone': range(5), 'pretone': range(5)},
#             'events': {'pretone': {'pre_stim': 0, 'post_stim': 1}, 'tone': {'pre_stim': 0, 'post_stim': .3}},
#             'rules': {'brain_region': {'pl': [('selected_animals', STANDARD_ANIMALS)],
#                                        'bla': [('selected_animals', STANDARD_ANIMALS)],
#                                        'hpc': [('selected_animals', HPC_THETA_POWER_ANIMALS)]}},
#             }

POWER_SPREADSHEET_OPTS = {
    'data_class': 'lfp', 'time_type': 'block', 'frequency_bands': ['theta_1'], 'data_type': 'power',
    'brain_regions': ['pl', 'bla', 'hpc'],  'frequency_type': 'continuous', 'row_type': 'event',
    'time_type': 'continuous',  'power_arg_set': (2048, 2000, 1000, 980, 2),
    'blocks': {'tone': range(5), 'pretone': range(5)}, 'power_deviation': False, 'collapse_sem_data': True,
    'events': {'pretone': {'pre_stim': .05, 'post_stim': .3}, 'tone': {'pre_stim': .05, 'post_stim': .3}},
    'rules': {'brain_region': {'pl': [('selected_animals', PFC_THETA_POWER_ANIMALS)],
                               'bla': [('selected_animals', BLA_THETA_POWER_ANIMALS)],
                               'hpc': [('selected_animals', HPC_THETA_POWER_ANIMALS)]}},
    'matlab_configuration': MATLAB_CONFIG
}

TEST_SPECTROGRAM_OPTS = {
    'data_class': 'lfp', 'frequency_bands': ['theta_1', 'theta_2'], 'data_type': 'power', 
    'row_type': 'event', 'frequency_type': 'block',
    'brain_regions': ['hpc', 'bla', 'pl'], 'power_deviation': False, 'time_type': 'block', 
    'periods': {'pretone': range(5), 'tone': range(5)}, 'power_arg_set': (2048, 2000, 1000, 980, 2),
    'filter': 'spectrum_estimation', 'levels': ['group', 'animal'], 'store': 'pkl',
    'validate_events': True,
    'events': {
        'pretone': {'pre_stim': .15, 'post_stim': .3}, 'tone': {'pre_stim': .15, 'post_stim': 3}
        },
    'rules': {
        'brain_region': 
        {'pl': [('inclusion_rule', {'animal': [['identifier', 'in', PFC_THETA_POWER_ANIMALS]]})],
         'bla': [('inclusion_rule', {'animal': [['identifier', 'in', BLA_THETA_POWER_ANIMALS]]})], 
         'hpc': [('inclusion_rule', {'animal': [['identifier', 'in', HPC_THETA_POWER_ANIMALS]]})]
                               }},
    'matlab_configuration': MATLAB_CONFIG, 'frequency_type': 'block'
    }


NEURON_QUALITY = ['1', '2a', '2b', '2ab']

PROPORTION_OPTS = {
    'data_class': 'spike', 'data_type': 'proportion', 'bin_size': 0.01, 'adjustment': 'normalized',
    'average_method': 'mean', 'base': 'event', 'time_type': 'continuous', 'row_type': 'event', 
    'levels': ['group', 'animal'], 'periods': {'tone': range(5)},
    'inclusion_rule': {'unit': [['quality', 'in', NEURON_QUALITY]], 
                       'animal': [['identifier', 'in', STANDARD_ANIMALS]]},
    'events': {'pretone': {'pre_stim': 0.0, 'post_stim': .7}, 'tone': {'pre_stim': .0, 'post_stim': .7}},
    }

PSTH_OPTS = {'data_class': 'spike', 'data_type': 'psth', 'bin_size': 0.01, 'adjustment': 'none',
             'average_method': 'mean', 'time_type': 'continuous', 'row_type': 'event', 
             'periods': {'pretone': range(5), 'tone': range(5)}, 
             'inclusion_rule': {'unit': [['quality', 'in', NEURON_QUALITY]], 
                                'animal': [['identifier', 'in', STANDARD_ANIMALS]]},
             'events': {'pretone': {'pre_stim': 0.0, 'post_stim': .7}, 'tone': {'pre_stim': .0, 'post_stim': .7}}}

MRL_INCLUSION = {'unit': [['quality', 'in', NEURON_QUALITY]], 'animal': [['identifier', 'in', STANDARD_ANIMALS]]}

MRL_OPTS = {'data_class': 'lfp', 'time_type': 'block', 'frequency_bands': ['theta_1', 'theta_2'], 
            'data_type': 'mrl', 'brain_regions': ['pl', 'bla', 'hpc'],  'frequency_type': 'block',
            'periods': {'tone': range(5), 'pretone': range(5)}, 'row_type': 'mrl_calculator',
            'validate_events': True,
            'events': {
                'pretone': {'pre_stim': 0, 'post_stim': 1}, 
                'tone': {'pre_stim': 0, 'post_stim': 1}},
            'power_arg_set': (2048, 2000, 1000, 980, 2), 'matlab_configuration': MATLAB_CONFIG,
            'rules': {
                'brain_region': 
                {
                    'pl': [('inclusion_rule', MRL_INCLUSION)],
                    'bla': [('inclusion_rule', MRL_INCLUSION)],
                    'hpc': [('inclusion_rule', {'unit': [['quality', 'in', NEURON_QUALITY]], 
                                                'animal': [['identifier', 'in', HPC_MRL_ANIMALS]]})
                                                ]
                }                   
                    }
           } # 


BLA_PL_INCLUSION = {'animal': [['identifier', 'in', list(set(BLA_THETA_POWER_ANIMALS) 
                                                         & set(PFC_THETA_POWER_ANIMALS))]]}
PL_HPC_INCLUSION = {'animal': [['identifier', 'in', list(set(HPC_THETA_POWER_ANIMALS) 
                                                         & set(PFC_THETA_POWER_ANIMALS))]]}
BLA_HPC_INCLUSION = {'animal': [['identifier', 'in', list(set(BLA_THETA_POWER_ANIMALS) 
                                                         & set(HPC_THETA_POWER_ANIMALS))]]}


VALIDATION_DATA_OPTS = {
    'data_class': 'lfp',
    'data_type': 'power',
    'brain_regions': ['pl', 'bla', 'hpc'],
    'power_arg_set': (2048, 2000, 1000, 980, 2), 
    'matlab_configuration': MATLAB_CONFIG,
    'frequency_band': (0, 8),
    'threshold': 20,
    'events': {
                'pretone': {'pre_stim': 0, 'post_stim': 1}, 
                'tone': {'pre_stim': 0, 'post_stim': 1}}
}

COHERENCE_OPTS = {'data_class': 'lfp', 'time_type': 'block', 'frequency_bands': ['theta_1', 'theta_2'], 
            'data_type': 'coherence',  'frequency_type': 'block', 'validate_events': True,
            'coherence_region_sets': ['bla_pl', 'bla_hpc', 'pl_hpc'], 
            'periods': {'tone': range(5), 'pretone': range(5)}, 'row_type': 'coherence_calculator',
            'power_arg_set': (2048, 2000, 1000, 980, 2), 'matlab_configuration': MATLAB_CONFIG,
            'rules': {
                'coherence_region_set': 
                {
                    'bla_pl': [('inclusion_rule', BLA_PL_INCLUSION)],
                    'bla_hpc': [('inclusion_rule', BLA_HPC_INCLUSION)],
                    'hpc_pl': [('inclusion_rule', PL_HPC_INCLUSION)]
                }                   
                    }
           } # 



GROUP_STAT_PROPORTION_OPTS = {
    'data_class': 'spike', 'data_type': 'proportion', 'base': 'event', 'adjustment': 'normalized', 
    'time_type': 'continuous', 'bin_size': 0.01, 'row_type': 'period', 'post_hoc_bin_size': 1, 'periods': {'tone': range(5)},
    'events': {
        'pretone': {'pre_stim': 0.0, 'post_stim': .7}, 'tone': {'pre_stim': .0, 'post_stim': .7}
        }, 'post_hoc_type': 'beta', 'group_colors': {'control': '#76BD4E', 'defeat': '#F2A354'},
        'inclusion_rule': {'unit': [['quality', 'in', NEURON_QUALITY]], 'animal': [['identifier', 'in', STANDARD_ANIMALS]]}, 
    'data_path': '/Users/katie/likhtik/IG_INED_Safety_Recall', 'periods': {'tone': range(5)},
    'period_type_regressor': True}

GROUP_STAT_PSTH_OPTS = {'data_class': 'spike', 'data_type': 'psth', 'adjustment': 'normalized', 'time_type': 'continuous', 
                        'bin_size': 0.01, 'row_type': 'period', 'post_hoc_bin_size': 1, 'period_type_regressor': True,
                        'events': {'pretone': {'pre_stim': 0.0, 'post_stim': .7}, 'tone': {'pre_stim': .0, 'post_stim': .7}}, 
                        'post_hoc_type': 'beta', 'periods': {'tone': range(5)},
                        'post_hoc_type': 'lmer', 'group_colors': {'control': '#76BD4E', 'defeat': '#F2A354'}, 
                        'data_path': '/Users/katie/likhtik/IG_INED_Safety_Recall', 
                        'inclusion_rule': {'unit': [['quality', 'in', NEURON_QUALITY]], 'animal': [['identifier', 'in', STANDARD_ANIMALS]]}}


# TEST_RUNNER_OPTS = {'data_opts': TEST_SPECTROGRAM_OPTS, 'graph_opts': GRAPH_OPTS}

# TEST_RUNNER_OPTS = {'data_opts': TEST_SPECTROGRAM_OPTS}

TEST_RUNNER_OPTS = [{'data_class': 'behavior', 'data_type': 'percent_freezing',
                     'periods': {'pretone': range(5), 'tone': range(5)}, 'row_type': 'period'}, 
                     TEST_SPECTROGRAM_OPTS, PSTH_OPTS, COHERENCE_OPTS, MRL_OPTS]

PSTH_OPTS = {'data_class': 'spike', 'data_type': 'psth', 'bin_size': 0.01, 'adjustment': 'normalized',
             'average_method': 'mean', 'time_type': 'continuous', 'row_type': 'event', 
             'periods': {'tone': range(5)}, 
             'inclusion_rule': {'unit': [['quality', 'in', NEURON_QUALITY]], 
                                'animal': [['identifier', 'in', STANDARD_ANIMALS]]},
             'events': {'pretone': {'pre_stim': 0.0, 'post_stim': .65}, 'tone': {'pre_stim': .0, 'post_stim': .65}}}

TEST_RUNNER_OPTS = [PSTH_OPTS]

#TEST_RUNNER_OPTS = {'data_opts': COHERENCE_OPTS, 'graph_opts': GRAPH_OPTS}

#TEST_RUNNER_OPTS = [MRL_OPTS]



