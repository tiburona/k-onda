psth_plot = {
    'plot_type': 'psth',
    'section': {
        'attr': 'calc',
        'aesthetics': {
            'ax': {'border': {'top': {'visible': 'FFF'}, 'right': {'visible': 'FFF'}}},
            'default': {
                'marker': {'color': '#1f77b4'},
                'indicator': {'type': 'patch', 'when': (0, .05)}}},
        'label': {'x_bottom': {'text': 'Seconds'},
                  'y_left': {'text': 'Firing Rate (Spikes per Second)'},
                  'title': {'text': 'Peristimulus Time Histogram', 'kwargs': {'y': 1.05}}},
        'divisions': [{'divider_type': 'period_type', 'members': ['stim']}]
    }
}

PSTH_OPTS = {
    
    'procedure': 'make_plots',
    'plot_spec': psth_plot,
    'io_opts': {'write_opts': './psth'},
    'calc_opts': {'kind_of_data': 'spike', 'calc_type': 'firing_rates',
                  'base': 'event', 'bin_size': .01, 
                  'periods': {'stim': {'period_pre_post': (1, 0), 'event_pre_post': (.05, 1)}} 
    }}

