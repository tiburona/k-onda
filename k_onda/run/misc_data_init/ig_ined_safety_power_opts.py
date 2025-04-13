

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




MATLAB_CONFIG = {'path_to_matlab': '/Applications/MATLAB_R2022a.app/bin/matlab',
                'paths_to_add': [], 'recursive_paths_to_add': ['/Users/katie/likhtik/software'],
                'base_directory': '/Users/katie/likhtik/data/temp'}


VALIDATION_DATA_OPTS = {
    'kind_of_data': 'lfp',
    'calc_type': 'power',
    'brain_regions': ['pl', 'hpc', 'bla'],
    'power_arg_set': (2048, 2000, 1000, 980, 2), 
    'bin_size': .01, 
    'lfp_padding': [1, 1],
    'lost_signal': [.25, .25],
    'matlab_configuration': MATLAB_CONFIG,
    'frequency_band': (0, 8),
    'threshold': 20, 
    'data_path': '/Users/katie/likhtik/IG_INED_Safety_Recall',
    'periods': {'tone': {'event_pre_post': (0, 1)}, 'pretone': {'event_pre_post': (0, 1)}},
    'rules': {
        'brain_region': 
        {'pl': [('filter', {'animal': {'identifier': ('in', PFC_THETA_POWER_ANIMALS)}})],
         'bla': [('filter', {'animal': {'identifier': ('in', BLA_THETA_POWER_ANIMALS)}})], 
         'hpc': [('filter', {'animal': {'identifier': ('in', HPC_THETA_POWER_ANIMALS)}})]
                               }}
    }

PREP_OPTS = {
    'calc_opts': VALIDATION_DATA_OPTS,
    'procedure': 'validate_lfp_events'
}


THETA_POWER_OPTS = {
    'kind_of_data': 'lfp', 
    'calc_type': 'power', 
    'brain_regions': ['pl', 'hpc', 'bla'], 
    'frequency_bands': ['theta_1', 'theta_2'], 
    'power_arg_set': (2048, 2000, 1000, 980, 2),
    'lfp_padding': [1, 1],
    'lost_signal': [.25, .25],
    'row_type': 'event', 
    'time_type': 'block', 
    'frequency_type': 'block', 
    'bin_size': .01,
    'concatenation': {'concatenator': 'animal', 'concatenated': 'period', 'dim_xform': 'lambda x: x+1'},
    'filter': 'filtfilt', 
    'store': 'pkl',  
    'validate_events': True,
    'periods': {'tone': {'event_pre_post': (0, .3)}, 'pretone': {'event_pre_post': (0, .3)}},
    'rules': {
        'brain_region': 
        {'pl': [('filter', {'animal': {'identifier': ('in', PFC_THETA_POWER_ANIMALS)}})],
         'bla': [('filter', {'animal': {'identifier': ('in', BLA_THETA_POWER_ANIMALS)}})], 
         'hpc': [('filter', {'animal': {'identifier': ('in', HPC_THETA_POWER_ANIMALS)}})]
                               }},
    'matlab_configuration': MATLAB_CONFIG, 
    }


POWER_PLOT_SPEC = {
    'plot_type': 'line_plot',
    'section': {
        'divisions': [
            {'divider_type': 'group',
             'data_source': 'group',
             'members': ['control', 'defeat'],
             'dim': 1}],
        'label': {
            'y_left': {
                'target': 'subfigure',
                'text': 'Power',
                'space_between': .3,
                'space_within': 0.5, 
                'which': 'absolute_first'}, 
            'title': {
                'target': 'subfigure',
                'text': '{brain_region} {frequency_band} Power',
                'space_between': .1,
                'x': .60},
            },
        'segment': {
            'attr': 'concatenation',
            'divisions':[
                {'divider_type': 'period_type',
                'members': ['pretone', 'tone']}],
            'subfigure': {'hspace': .08, 'wspace': 0},
            'label': {  
                'x_bottom': {'text': 'Period'},
                'x_top': {'text': '{identifier} Group', 'target': 'subfigure'}
                },
            'legend': {'key': { 'loc': 'upper right'},
                       'which': 1},
            'aesthetics': {
                'ax': {
                    'border': {'top': {'visible': 'FFF'}, 
                               'right': {'visible': 'FFF'}},
                    'tick_labels': {'x': {'only_whole_numbers': True}}},
                'conditional': 
                {'period_type': {'pretone': {'marker': {'color': 'pink'}},
                                 'tone': {'marker': {'color': 'green'}}}}}
            }}}


    
POWER_PLOT_OPTS = {
    'procedure': 'make_plots',
    'write_opts': {
        'fname': {'template': '/Users/katie/likhtik/IG_INED_Safety_Recall/power/{brain_region}_{frequency_band}',
                  'fields': ['brain_region', 'frequency_band']}
    },
    'calc_opts': THETA_POWER_OPTS,
    'plot_spec': POWER_PLOT_SPEC
}



