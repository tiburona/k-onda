
####### RASTER PLOT ##############

raster_plot = {
    'plot_type': 'raster',
    'section': {
        'divisions': [
             {  'divider_type': 'period_type',
                'members': ['light'],
                'dim': 1
            }],
            'aesthetics': {
                 'ax': {'border': {'top': {'visible': 'FFF'}, 'right': {'visible': 'FFF'}}}
               
            },
            'subfigure': {'hspace': 0, 'wspace': 0}
        ,
        'section': {
             'divisions': [ {
                    'divider_type': 'unit',
                    'data_source': 'unit',
                    'members': 'all_units',
                    'dim': 0
                
        }],
            'attr': 'calc',
            'aesthetics': {
                'default': {
                    'marker': {'colors': 'black'},
                    'ax': {
                        'border': {
                            'all': {'visible': 'FFF'}
                            }}
                },
                'positional': {
                    ('x', 'absolute_last'): {'ax': {'border': {'bottom': {'visible': 'TTT'}}}}}
            },
            'subfigure': {'hspace': 0, 'wspace': 0},
            
            'label': {'y_ax': {
                'text': '{lambda obj: obj.category.lower().replace("good", "unit") + " " + str(obj.experiment_wise_index + 1)}',
                'kwargs': {'rotation': 0}
            }}}
           }}


CH27_UNITS_RASTER_OPTS = {
    'procedure': 'make_plots',
    'plot_spec': raster_plot,
    'write_opts': '/Users/katie/likhtik/ch27/raster',
    'calc_opts': {'kind_of_data': 'spike', 'calc_type': 'spike_train', 'raster_type': 'spike_train', 
                    'base': 'period', 'bin_size': .5, 'sort': {'unit': ('category', 'ascending')},
                    'periods': {'light': {'period_pre_post': (10, 10)}, 'prelight': {'period_pre_post': (10, 10)}},
                    'filter': {'period': {'identifier': ('==', 0)}}}
                    }


######### GROUP PSTH ############

group_psth_plots_whole_axis = {
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
                'members': ['light']
            }]
    }}



CH27_GROUP_PSTH_OPTS = {
    
    'procedure': 'make_plots',
    'plot_spec': group_psth_plots_whole_axis,
    'write_opts': '/Users/katie/likhtik/ch27/psth',
    'calc_opts': {'kind_of_data': 'spike', 'calc_type': 'firing_rates', 
                  'base': 'period', 'bin_size': .5, 
                  'periods': {'light': {'period_pre_post': (10, 10)}} 
    }}


####### WAVEFORM PLOTS #######


units_waveform_plots = {
    'graph_dir': '/Users/katie/likhtik/CH27/',
    'fname': 'units_waveform',
    'plot_type': 'waveform',
    'section': {
           'aesthetics': {
                'default': {
                    'marker': {'color': 'black'},
                    'ax': {
                        'border': {
                            'all': {'visible': 'FFF'}
                            }}
                }
            },
             'label': {'x_ax': {
                'text': '{identifier}',
            }},
                'divisions':  [ {
                    'divider_type': 'unit',
                    'data_source': 'unit',
                    'members': 'all_units',
                    'dim': 0
                
        }],
            }
}


UNITS_WAVEFORM_PLOTS = {
    
    'procedure': 'make_plots',
    'plot_spec': units_waveform_plots,
    'calc_opts': {'kind_of_data': 'spike', 'calc_type': 'waveform',  
                  'base': 'unit',
                   'filter': {'unit': {'category':  ('==', 'good')}}}
    }


####### FIRING RATE BY UNIT ##########

firing_rate_plot = {
    'fname': 'group_percent_change_plot_only_good',
    'section': {
        'aesthetics': {
            'default': {
                'label': {'color': 'black'},
                'ax': {
                    'zero_line': 'false',
                    'border': {
                        'top': {'visible': 'FFF'}, 'right': {'visible': 'FFF'}
                        }}
        }},
        'divisions': [
             {'divider_type': 'unit',
              'data_source': 'unit',
              'members': 'all_units', 
              'dim': 1
                }],
        'layers': [
                {'plot_type': 'categorical_scatter', 
                'aesthetics': {'default': {'cat_width': 4, 'spacing': 2, 'marker': {'color': 'black'}}},
                    'attr': 'scatter'}, 
                {'plot_type': 'categorical_line', 
                'attr': 'mean', 
                'aesthetics': {'default': {'cat_width': 4, 'spacing': 2, 'marker': {'colors': 'blue', 
                                            'linestyles': '--'}}}}],
        'segment': {
            'aesthetics': {
                'conditional': {
                'period_type': {
                    'prelight': {'background_color': ('white', .2)},
                    'light': {'background_color': ('green', .2)}}
        }},
            
            'divisions': [
                {
                    'divider_type': 'period_type',
                    'members': ['prelight', 'light'], 
                    'grouping': 0
                    }]
            }
    }
}

FIRING_RATE_OPTS = {
    'procedure': 'make_plots',
    'plot_spec': firing_rate_plot,
    'calc_opts': {'kind_of_data': 'spike', 'calc_type': 'firing_rates', 'base': 'period',
                    'bin_size': .5, 
                    'filter': {'unit': {'category': ('==', 'good')}},
                    'data_path': '/Users/katie/likhtik/CH27mice'}
}



####### GROUP PERCENT CHANGE ########

new_percent_change_plot = {
    'fname': 'new_group_percent_change_plot_diff_colors',
    
    'section': {
        'divisions': [
            {'divider_type': 'period_type', 
             'members': ['light'], 
             'dim': 1
            }],
            'layers': [
                {'plot_type': 'categorical_scatter', 
                'aesthetics': {'default': {'cat_width': 3, 'spacing': 2, 'marker': {'color': 'black'}},
                                'period_type': {'light': {'background_color': ('green', .2)}}},
                'attr': 'grandchildren_scatter'}, 
                {'plot_type': 'categorical_line', 
                'attr': 'mean',
                'aesthetics': {'default': {'cat_width': 3, 'spacing': 2, 'marker': {'colors': 'blue', 
                                            'linestyles': '--'}}}}],
           'aesthetics': {
            'default': {
                'label': {'color': 'black'},
                'ax': {
                    'zero_line': 'false',
                    'border': {
                        'top': {'visible': 'FFF'}, 'right': {'visible': 'FFF'}
                        }}
        }},
        'segment': {
            
             'divisions': [ {
                    'divider_type': 'animal',
                    'data_source': 'animal',
                    'members': 'all_animals',
                    'dim': 0
                
        }],
         'aesthetics': {
                'conditional': {
                'period_type': {
                    'light': {'background_color': ('green', .2)}}
        }}
            }
            }
    }


CH27_PERCENT_CHANGE_OPTS = {
    'procedure': 'make_plots',
    'plot_spec': new_percent_change_plot,
    'calc_opts': {'kind_of_data': 'spike', 'calc_type': 'firing_rates', 'base': 'period',
                    'bin_size': .5, 'percent_change': {'level': 'unit', 'reference': 'prelight'}}
}