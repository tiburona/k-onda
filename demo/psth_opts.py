psth_plot = {
    'plot_type': 'psth',
    'section': {
        'attr': 'calc',
        'aesthetics': {
            'ax': {'border': {'top': {'visible': 'FFF'}, 'right': {'visible': 'FFF'}}},
            'default': 
                    {'marker': {'color': 'black'},
                    }},
        'label': {'x': 
                  {'text': 'Seconds'}, 
                  'y': {'text': 'Firing Rate (Spikes per Second)'}},        
        'divisions': [
            {
                'divider_type': 'period_type',
                'members': ['stim']
            }]
    }}



PSTH_OPTS = {
    
    'procedure': 'make_plots',
    'plot_spec': psth_plot,
    'write_opts': './psth',
    'calc_opts': {'kind_of_data': 'spike', 'calc_type': 'firing_rates', 
                  'base': 'period', 'bin_size': .5, 
                  'periods': {'light': {'period_pre_post': (1, 0)}} 
    }}
