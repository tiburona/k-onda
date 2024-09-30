STANDARD_ANIMALS = ['IG160', 'IG163', 'IG176', 'IG178', 'IG180', 'IG154', 'IG156', 'IG158', 'IG177', 'IG179']



PROPORTION_OPTS = {'kind_of_data': 'spike', 'data_type': 'proportion', 'pre_stim': 0.05, 'post_stim': 0.65,
                   'bin_size': 0.01, 'trials': (0, 150), 'adjustment': 'normalized', 'base': 'trial',
                   'time': 'continuous', 'row_type': 'trial'}

AUTOCORR_OPTS = {'kind_of_data': 'spike', 'data_type': 'autocorr', 'pre_stim': 0.0, 'post_stim': 30.0, 'bin_size': 0.01,
                 'tone_events': (0, 300, 30), 'pretone_events': (0, 300, 30), 'max_lag': 99, 'block_types': ['tone']}

SPECTRUM_OPTS = {'kind_of_data': 'spike', 'data_type': 'spectrum', 'pre_stim': 0.0, 'post_stim': 30.0, 'bin_size': 0.01,
                 'tone_events': (0, 300, 30), 'pretone_events': (0, 300, 30), 'max_lag': 99, 'freq_range': (3, 60),
                 'block_types': ['tone']}

SPREADSHEET_OPTS = {'kind_of_data': 'spike', 'data_path': '/Users/katie/likhtik/data', 'data_type': 'psth', 'base': 'trial',
                    'pre_stim': 0.0, 'post_stim': .05, 'adjustment': 'none', 'bin_size': 0.01, 'trials': (0, 150),
                    'row_type': 'event', 'periods': list(range(5)), 'period_types': ['pretone', 'tone'],
                    'selected_animals': STANDARD_ANIMALS, 'time_type': 'continuous'}



GRAPH_OPTS = {'graph_dir': '/Users/katie/likhtik/data/graphs', 'units_in_fig': 4, 'tick_step': 0.1, 'sem': False,
              'footer': True, 'equal_y_scales': True, 'group_colors': {'control': '#76BD4E', 'stressed': '#F2A354'},
              'force_recalc': False}

ROSE_PLOT_OPTS = {'graph_dir': '/Users/katie/likhtik/data/graphs', 'units_in_fig': 4, 'sem': False,
              'footer': False, 'equal_y_scales': True, 'group_colors': {'control': '#76BD4E', 'stressed': '#F2A354'},
              'force_recalc': False, 'superimpose': False}


FIGURE_1_OPTS = {'kind_of_data': 'spike', 'data_path': '/Users/katie/likhtik/IG_INED_Safety_Recall', 'animal_id': 'IG180',
                 'unit_ids': [10, 2], 'electrodes_for_waveform': [[11], [3]], 'electrodes_for_feature': [1, 3],
                 'el_inds': [0, 0], 'pc_inds': [0, 1], 'sem': False, 'equal_y_scales': True, 'tick_step': 0.2,
                 'neuron_type_colors': {'IN': '#507DA0', 'PN': '#E76F51'}, 'annot_coords': (-0.11, 1.1),
                 'group_colors': {'control': '#9C89B8', 'defeat': '#F4A261'}, 'hist_color': '#9C89B8', 'force_recalc':
                 False}


HEAT_MAP_DATA_OPTS = {'kind_of_data': 'lfp', 'data_path': '/Users/katie/likhtik/data', 'time_type': 'period',
                      'brain_region': 'hpc', 'fb': ['gamma'], 'frequency_type': 'continuous', 'data_type': 'mrl',
                      'row_type': 'frequency_bin', 'pre_stim': 0, 'post_stim': 0.3,
                      'trials': (0, 150), 'adjustment': 'relative', 'frequency_band': (0, 70)}


BEHAVIOR_OPTS = {'kind_of_data': 'behavior', 'data_type': 'percent_freezing', 'row_type': 'period',
                 'period_types': ['pretone', 'tone'], 'data_path': '/Users/katie/likhtik/data'}

CAROLINA_OPTS = {'kind_of_data': 'spike', 'data_path': '/Users/katie/likhtik/CH_for_katie_less_conservative',
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

CAROLINA_GROUP_STAT_OPTS = {'kind_of_data': 'spike', 'data_type': 'psth', 'adjustment': 'normalized',
                            'time_type': 'continuous', 'pre_stim': 0.0, 'post_stim': 0.70, 'bin_size': 0.01,
                            'trials': (0, 150), 'row_type': 'trial', 'post_hoc_bin_size': 1,
                            'data_path': '/Users/katie/likhtik/data', 'post_hoc_type': 'lmer', 'group_colors':
                                {'control': '#76BD4E', 'arch': '#F2A354'}, 'sem_level': 'unit'}

SPONTANEOUS_OPTS = {'kind_of_data': 'spike', 'data_path': '/Users/katie/likhtik/CH_for_katie_less_conservative',
                    'data_type': 'spontaneous_firing', 'bin_size': 0.1, 'selected_animals': ['CH272', 'CH274', 'CH275'],
                    'spontaneous': 120}


SPONTANEOUS_GRAPH_OPTS = {'graph_dir': '/Users/katie/likhtik/CH_for_katie_less_conservative/graphs', 'units_in_fig': 4, 'tick_step': 100,
                          'sem': False, 'footer': False, 'equal_y_scales': True,
                          'group_colors': {'control': '#9F9FA3', 'arch': '#32B44A'}, 'force_recalc': False,
                          'neuron_type_colors': {'PV': '#5679C7', 'ACH': '#C75B56'}}

CROSS_CORR_OPTS = {'kind_of_data': 'spike', 'data_type': 'correlogram', 'pre_stim': 0, 'post_stim': 1,
                   'adjustment': 'none', 'bin_size': 0.001, 'events': (0, 300), 'periods': list(range(10)),
                   'period_types': ['pretone', 'tone'], 'unit_pairs': ['ACH,PV'], 'max_lag': .05}

SPONTANEOUS_MRL_OPTS = {'kind_of_data': 'lfp', 'data_type': 'mrl', 'bin_size': 0.01, 'spontaneous': (0, 120),
                        'events': (0, 300),  'frequency_bands': ['theta_1'], 'brain_regions': ['bla', 'il', 'bf'],
                        'sem_level': 'mrl_calculator'}

CAROLINA_MRL_OPTS = {'kind_of_data': 'lfp', 'data_type': 'mrl', 'bin_size': 0.01, 'pre_stim': 0, 'post_stim': .3,
                     'frequency_bands': ['theta_1'], 'brain_regions': ['bla', 'il', 'bf'],
                     'sem_level': 'mrl_calculator'}


PFC_THETA_POWER_ANIMALS = [
    'IG160', 'IG163', 'IG171', 'IG176', 'IG180', 'INED04', 'INED16', 'INED18', 'IG156', 'IG158', 'IG172',
    'IG174', 'IG175', 'IG177', 'IG179', 'INED07', 'INED06', 'INED09', 'INED11', 'INED12'
]

BLA_THETA_POWER_ANIMALS = [
    'IG160', 'IG161', 'IG162', 'IG163', 'IG171', 'IG173', 'IG176', 'IG178', 'IG180', 'INED04', 
    'INED05', 'INED17', 'INED18', 'IG154', 'IG155', 'IG156', 'IG158', 'IG172', 'IG174', 'IG175', 
    'IG179', 'INED01', 'INED07', 'INED09', 'INED12'
]

HPC_THETA_POWER_ANIMALS = ['IG162', 'IG171', 'IG173', 'IG176', 'IG155', 'IG174', 'IG175', 'IG179']

HPC_MRL_ANIMALS = list(set(HPC_THETA_POWER_ANIMALS) & set(STANDARD_ANIMALS))
PL_MRL_ANIMALS = list(set(PFC_THETA_POWER_ANIMALS) & set(STANDARD_ANIMALS))
BLA_MRL_ANIMALS = list(set(BLA_THETA_POWER_ANIMALS) & set(STANDARD_ANIMALS))

GRAPH_OPTS = {'graph_dir': '/Users/katie/likhtik/IG_INED_SAFETY_RECALL', 'units_in_fig': 4, 'tick_step': 0.1,
              'sem': False, 'footer': True, 'equal_y_scales': True, 'equal_color_scales': 'within_data_source',
              'group_colors': {'control': '#6C4675', 'defeat': '#F2A354'}, 'period_colors':
                  {'pretone': '#E75480', 'tone': '#76BD4E'}, 'period_order': ['pretone', 'tone']
              }

MATLAB_CONFIG = {
                'path_to_matlab': '/Applications/MATLAB_R2022a.app/bin/matlab',
                'paths_to_add': [], 'recursive_paths_to_add': ['/Users/katie/likhtik/software'],
                'base_directory': '/Users/katie/likhtik/data/temp'}

LFP_OPTS = {'kind_of_data': 'lfp', 'time_type': 'block', 'frequency_bands': ['theta_1', 'theta_2'], 'data_type': 'power',
            'brain_regions': ['pl', 'bla', 'hpc'],  'frequency_type': 'block', 'row_type': 'event',
            'blocks': {'tone': range(5), 'pretone': range(5)}, 'power_deviation': False, 'collapse_sem_data': True,
            'events': {'pretone': {'pre_stim': 0, 'post_stim': 1}, 'tone': {'pre_stim': 0, 'post_stim': .3}},
            'rules': {'brain_region': {'pl': [('selected_animals', PFC_THETA_POWER_ANIMALS)],
                                       'bla': [('selected_animals', BLA_THETA_POWER_ANIMALS)],
                                       'hpc': [('selected_animals', HPC_THETA_POWER_ANIMALS)]}},
            'matlab_configuration': MATLAB_CONFIG}


POWER_SPREADSHEET_OPTS = {
    'kind_of_data': 'lfp', 'time_type': 'block', 'frequency_bands': ['theta_1', 'theta_2'], 'data_type': 'power',
    'brain_regions': ['pl', 'bla', 'hpc'],  'frequency_type': 'continuous', 'row_type': 'event',
    'time_type': 'continuous',  'power_arg_set': (2048, 2000, 1000, 980, 2),
    'blocks': {'tone': range(5), 'pretone': range(5)}, 'power_deviation': False, 'collapse_sem_data': True,
    'events': {'pretone': {'pre_stim': .05, 'post_stim': .3}, 'tone': {'pre_stim': .05, 'post_stim': .3}},
    'rules': {'brain_region': {'pl': [('selected_animals', PFC_THETA_POWER_ANIMALS)],
                               'bla': [('selected_animals', BLA_THETA_POWER_ANIMALS)],
                               'hpc': [('selected_animals', HPC_THETA_POWER_ANIMALS)]}},
    'matlab_configuration': MATLAB_CONFIG
}


SPECTROGRAM_OPTS = {
    'row_type': 'event', 'frequency_type': 'block', 'bin_size': .01, 'level': 'group',
    'brain_regions': ['bla', 'pl', 'hpc'], 'time_type': 'continuous', 
    'periods': {'pretone': range(5), 'tone': range(5)}, 'power_arg_set': (2048, 2000, 500, 480, 2),
    'lost_signal': [.125, .125], 'bin_size': .01, 'lfp_padding': [.625, .625],
    'remove_noise': 'filtfilt', 'store': 'pkl',  'validate_events': True,
    'frequency_bands': [(30, 50), (70, 120)], 'kind_of_data': 'lfp', 'calc_type': 'power',
    'data_path': '/Users/katie/likhtik/IG_INED_SAFETY_RECALL',
    'events': {
        'pretone': {'pre_stim': .15, 'post_stim': .3}, 'tone': {'pre_stim': .15, 'post_stim': .3}
        },
    'rules': {
        'brain_region': 
        {'pl': [('filter', {'animal': {'identifier': ('in', PFC_THETA_POWER_ANIMALS)}})],
         'bla': [('filter', {'animal': {'identifier': ('in', BLA_THETA_POWER_ANIMALS)}})], 
         'hpc': [('filter', {'animal': {'identifier': ('in', HPC_THETA_POWER_ANIMALS)}})]
                               }},
    'matlab_configuration': MATLAB_CONFIG, 
    }

POWER_OPTS = {
    'kind_of_data': 'lfp', 'frequency_bands': [(30, 50), (70,120)], 'data_type': 'power', 
    'row_type': 'event', 'frequency_type': 'block', 'bin_size': .01,
    'brain_regions': ['pl', 'hpc', 'bla'], 'time_type': 'continuous', 
    'periods': {'pretone': range(5), 'tone': range(5)}, 'power_arg_set': (2048, 2000, 500, 480, 2),
    'filter': 'filtfilt', 'store': 'pkl',  'validate_events': True,
    'events': {
        'pretone': {'pre_stim': .15, 'post_stim': .85}, 'tone': {'pre_stim': .85, 'post_stim': .3}
        },
    'rules': {
        'brain_region': 
        {'pl': [('filter', {'animal': {'identifier': ('in', PFC_THETA_POWER_ANIMALS)}})],
         'bla': [('filter', {'animal': {'identifier': ('in', BLA_THETA_POWER_ANIMALS)}})], 
         'hpc': [('filter', {'animal': {'identifier': ('in', HPC_THETA_POWER_ANIMALS)}})]
                               }},
    'matlab_configuration': MATLAB_CONFIG, 
    }

NEURON_QUALITY = ['1', '2a', '2b', '2ab']

PROPORTION_OPTS = {
    'kind_of_data': 'spike', 'data_type': 'proportion', 'bin_size': 0.01, 'adjustment': 'normalized',
    'average_method': 'mean', 'base': 'event', 'time_type': 'continuous', 'row_type': 'event', 
    'levels': ['group', 'animal'], 'periods': {'tone': range(5)},
    'inclusion_rule': {'unit': [['quality', 'in', NEURON_QUALITY]], 
                       'animal': [['identifier', 'in', STANDARD_ANIMALS]]},
    'events': {'pretone': {'pre_stim': 0.05, 'post_stim': .65}, 'tone': {'pre_stim': .05, 'post_stim': .65}},
    }

PSTH_OPTS = {'kind_of_data': 'spike', 'calc_type': 'firing_rates', 'bin_size': 0.01,
             'average_method': 'mean', 'time_type': 'continuous', 
             'periods': {'prelight': range(10), 'light': range(10), 'tone': range(10)}, 'base': 'period'}


MRL_OPTS = {'kind_of_data': 'lfp', 'time_type': 'block', 'frequency_bands': ['theta_1', 'theta_2'], 
            'data_type': 'mrl', 'brain_regions': ['pl', 'bla', 'hpc'],  'frequency_type': 'block',
            'periods': {'tone': range(1), 'pretone': range(1)}, 'row_type': 'mrl_calculator',
            'validate_events': True,
            'events': {
                'pretone': {'pre_stim': 0, 'post_stim': 1}, 
                'tone': {'pre_stim': 0, 'post_stim': 1}},
            'power_arg_set': (2048, 2000, 1000, 980, 2), 'matlab_configuration': MATLAB_CONFIG,
            'rules': {
                'brain_region': 
                {
                    'pl': [('inclusion_rule', {'unit': [['quality', 'in', NEURON_QUALITY]], 
                                                'animal': [['identifier', 'in', PL_MRL_ANIMALS]]})],
                    'bla': [('inclusion_rule', {'unit': [['quality', 'in', NEURON_QUALITY]], 
                                                'animal': [['identifier', 'in', BLA_MRL_ANIMALS]]})],
                    'hpc': [('inclusion_rule', {'unit': [['quality', 'in', NEURON_QUALITY]], 
                                                'animal': [['identifier', 'in', HPC_MRL_ANIMALS]]})]
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
    'kind_of_data': 'lfp',
    'calc_type': 'power',
    'brain_regions': ['pl', 'hpc', 'bla'],
    'power_arg_set': (2048, 2000, 1000, 980, 2), 
    'bin_size': .01, 
    'lfp_padding': [1, 1],
    'lost_signal': [.75, .75],
    'matlab_configuration': MATLAB_CONFIG,
    'frequency_band': (0, 8),
    'threshold': 20,
     'data_path': '/Users/katie/likhtik/IG_INED_Safety_Recall',
    'events': {
                'pretone': {'pre_stim': 0, 'post_stim': 1}, 
                'tone': {'pre_stim': 0, 'post_stim': 1}},
    'rules': {
        'brain_region': 
        {'pl': [('filter', {'animal': {'identifier': ('in', PFC_THETA_POWER_ANIMALS)}})],
         'bla': [('filter', {'animal': {'identifier': ('in', BLA_THETA_POWER_ANIMALS)}})], 
         'hpc': [('filter', {'animal': {'identifier': ('in', HPC_THETA_POWER_ANIMALS)}})]
                               }}
    }

# BLA_PL_INCLUSION = {'animal': [['identifier', 'in', ['IG160', 'IG163', 'IG179']]]}


# VALIDATION_DATA_OPTS = {
#     'kind_of_data': 'lfp',
#     'data_type': 'power',
#     'brain_regions': ['pl', 'bla'],
#     'power_arg_set': (2048, 2000, 1000, 980, 2), 
#     'matlab_configuration': MATLAB_CONFIG,
#     'frequency_band': (0, 8),
#     'threshold': 20,
#     'events': {
#                 'pretone': {'pre_stim': 0, 'post_stim': 1}, 
#                 'tone': {'pre_stim': 0, 'post_stim': 1}},
#     'rules': {
#         'brain_region': 
#         {'pl': [('inclusion_rule', {'animal': [['identifier', 'in', ['IG160', 'IG163', 'IG179']]]})],
#          'bla': [('inclusion_rule', {'animal': [['identifier', 'in', ['IG160', 'IG163', 'IG179']]]})], 
    
#                                }}
# }

'3-6/6-12 for bla-pl.  bla-hpc and pl-hpc , 3-5, 5-12. '
BLA_PL_COHERENCE_BANDS = [(3, 6), (6, 12)]
BLA_HPC_COHERENCE_BANDS = [(3, 5), (5, 12)]
HPC_PL_COHERENCE_BANDS = [(3, 5), (5, 12)]

BLA_HPC_COHERENCE_OPTS = {'kind_of_data': 'lfp', 'time_type': 'block', 
            'data_type': 'coherence',  'frequency_type': 'block', 'validate_events': True,
            'region_set': 'bla_hpc', 
            'data_path': '/Users/katie/likhtik/IG_INED_Safety_Recall',
            'periods': {'tone': range(5), 'pretone': range(5)}, 'row_type': 'coherence_calculator',
            'power_arg_set': (2048, 2000, 1000, 980, 2), 'matlab_configuration': MATLAB_CONFIG,
            'inclusion_rule': BLA_HPC_INCLUSION,
            'frequency_bands': BLA_HPC_COHERENCE_BANDS}

BLA_PL_COHERENCE_OPTS = {'kind_of_data': 'lfp', 'time_type': 'block', 
            'data_type': 'coherence',  'frequency_type': 'block', 'validate_events': True,
            'region_set': 'bla_pl', 
            'data_path': '/Users/katie/likhtik/IG_INED_Safety_Recall',
            'periods': {'tone': range(5), 'pretone': range(5)}, 'row_type': 'coherence_calculator',
            'power_arg_set': (2048, 2000, 1000, 980, 2), 'matlab_configuration': MATLAB_CONFIG,
            'inclusion_rule': BLA_PL_INCLUSION,
            'frequency_bands': BLA_PL_COHERENCE_BANDS}


HPC_PL_COHERENCE_OPTS = {'kind_of_data': 'lfp', 'time_type': 'block', 
            'data_type': 'coherence',  'frequency_type': 'block', 'validate_events': True,
            'region_set': 'hpc_pl', 
            'data_path': '/Users/katie/likhtik/IG_INED_Safety_Recall',
            'periods': {'tone': range(5), 'pretone': range(5)}, 'row_type': 'coherence_calculator',
            'power_arg_set': (2048, 2000, 1000, 980, 2), 'matlab_configuration': MATLAB_CONFIG,
            'inclusion_rule': PL_HPC_INCLUSION,
            'frequency_bands': HPC_PL_COHERENCE_BANDS}

CORRELATION_OPTS = {'kind_of_data': 'lfp', 'time_type': 'continuous', 'frequency_bands': ['theta_1', 'theta_2'], 
            'data_type': 'correlation',  'frequency_type': 'block', 'validate_events': True,
            'region_sets': ['bla_pl', 'bla_hpc', 'hpc_pl'], 'lags': 200, 'bin_size': .01,
            'periods': {'tone': range(5), 'pretone': range(5)}, 'row_type': 'correlation_calculator',
            'power_arg_set': (2048, 2000, 1000, 980, 2), 'matlab_configuration': MATLAB_CONFIG,
            'rules': {
                'region_set': 
                {
                    'bla_pl': [('inclusion_rule', BLA_PL_INCLUSION)],
                    'bla_hpc': [('inclusion_rule', BLA_HPC_INCLUSION)],
                    'hpc_pl': [('inclusion_rule', PL_HPC_INCLUSION)]
                }                   
                    }
           } 


GROUP_STAT_PROPORTION_OPTS = {
    'kind_of_data': 'spike', 'data_type': 'proportion', 'base': 'event', 'adjustment': 'normalized', 
    'time_type': 'continuous', 'bin_size': 0.01, 'row_type': 'period', 'post_hoc_bin_size': 1, 'periods': {'tone': range(5)},
    'events': {
        'pretone': {'pre_stim': 0.0, 'post_stim': .7}, 'tone': {'pre_stim': .0, 'post_stim': .7}
        }, 'post_hoc_type': 'beta', 'group_colors': {'control': '#76BD4E', 'defeat': '#F2A354'},
        'inclusion_rule': {'unit': [['quality', 'in', NEURON_QUALITY]], 'animal': [['identifier', 'in', STANDARD_ANIMALS]]}, 
    'data_path': '/Users/katie/likhtik/IG_INED_Safety_Recall', 'periods': {'tone': range(5)},
    'period_type_regressor': True}


GROUP_STAT_PSTH_OPTS = {'kind_of_data': 'spike', 'data_type': 'psth', 'adjustment': 'normalized', 'time_type': 'continuous', 
                        'bin_size': 0.01, 'row_type': 'event', 'post_hoc_bin_size': 1, 'period_type_regressor': True,
                        'events': {'pretone': {'pre_stim': 0.0, 'post_stim': .7}, 'tone': {'pre_stim': .0, 'post_stim': .7}}, 
                        'periods': {'tone': range(5)}, 'post_hoc_type': 'poisson', 
                        'data_path': '/Users/katie/likhtik/IG_INED_Safety_Recall',
                        'inclusion_rule': {'unit': [['quality', 'in', NEURON_QUALITY]], 'animal': [['identifier', 'in', STANDARD_ANIMALS]]}}

# PSTH_OPTS = {'kind_of_data': 'spike', 'calc_type': 'psth', 'bin_size': 0.01, 'adjustment': 'normalized',
            #  'average_method': 'mean', 'time_type': 'continuous', 'row_type': 'event', 
            #  'periods': {'tone': range(5)}, 'levels': ['group'],
            #  'filters': {'unit': {'quality': ('in', NEURON_QUALITY)}, 
            #             'animal': {'identifier': ('in', STANDARD_ANIMALS)}},
            #  'events': {'pretone': {'pre_stim': 0.05, 'post_stim': .65}, 'tone': {'pre_stim': 0.05, 'post_stim': .65}}}

COUNT_OPTS = {'kind_of_data': 'spike', 'data_type': 'spike_counts', 'bin_size': 0.01, 'adjustment': 'none',
            'time_type': 'continuous', 'row_type': 'event', 
             'periods': {'tone': range(5)}, 
             'inclusion_rule': {'unit': [['quality', 'in', NEURON_QUALITY]], 
                                'animal': [['identifier', 'in', STANDARD_ANIMALS]]},
             'events': {'pretone': {'pre_stim': .0, 'post_stim': .3}, 'tone': {'pre_stim': .0, 'post_stim': .3}}}


CAROLINA_GRAPH_OPTS = {
    'graph_dir': '/Users/katie/likhtik/CH_EXT', 'units_in_fig': 4, 'tick_step': 0.1, 
    'sem': False, 'footer': True, 'equal_y_scales': True, 'equal_color_scales': 'within_group', 
    'colors': {
        'period_type pretone period_group 0': '#000000',
        'period_type pretone period_group 1': '#808080',
        'period_type tone period_group 0': '#82086F',
        'period_type tone period_group 1': '#D52C90' 
    }
    }

PHASE_PHASE_OPTS = {
    'kind_of_data': 'lfp', 'data_type': 'phase_trace', 'time_type': 'block', 
    'frequency_bands': [(5,6)], 'frequency_type': 'continuous',
    'events': {'pretone': {'post_stim': .3}, 'tone': {'post_stim': .3}},
    'region_sets': [ 'bla_il'], 'period_groups': [(0, 2), (18, 20)],
    'periods': {'tone': range(20), 'pretone': range(20)}, 'row_type': 'phase_relationship_calculator',
    'power_arg_set': (2048, 2000, 1000, 980, 2), 'matlab_configuration': MATLAB_CONFIG
            }


#BLA_PL_INCLUSION = {'animal': [['identifier', 'in', ['IG160', 'IG163', 'IG179']]]}

GRANGER_OPTS = {
    'kind_of_data': 'lfp', 'data_type': 'granger_causality', 'time_type': 'block', 
    'frequency_bands': [(0, 20)], 'frequency_type': 'continuous',  'aggregator': 'none',
    'events': {'pretone': {'post_stim': 1}, 'tone': {'post_stim': 1}},
    'region_sets': [ 'bla_pl'], 'validate_events': True, 
    'periods': {'tone': range(5), 'pretone': range(5)}, 'row_type': 'granger_calculator',
    'power_arg_set': (2048, 2000, 1000, 980, 2), 'matlab_configuration': MATLAB_CONFIG,
    'inclusion_rule': BLA_PL_INCLUSION
}


BLA_PL_COHERENCE_RUNNER_OPTS = {'data_opts': BLA_PL_COHERENCE_OPTS, 'graph_opts': GRAPH_OPTS}
HPC_PL_COHERENCE_RUNNER_OPTS = {'data_opts': HPC_PL_COHERENCE_OPTS, 'graph_opts': GRAPH_OPTS}
BLA_HPC_COHERENCE_RUNNER_OPTS = {'data_opts': BLA_HPC_COHERENCE_OPTS, 'graph_opts': GRAPH_OPTS}


#COHERENCE_SPREADSHEET_OPTS = [COHERENCE_OPTS]

SPREADSHEET_OPTS = [BLA_PL_COHERENCE_OPTS, HPC_PL_COHERENCE_OPTS, BLA_HPC_COHERENCE_OPTS]

RUNNER_OPTS = {'calc_opts': PSTH_OPTS, 'graph_opts': GRAPH_OPTS}

