# 1.Group psth showing activity before/during/after light on all trials (-10 to +10s)

# 2.raster with each separate unit or mua we have taking up 1 row, showing activity before/during/after light on trial 1 (these could be dots or lines, -10 to 10s)

# 3.⁠ ⁠Waveform and avg pre & post-light firing rate scatterplot for the 3 single units

# 4.⁠ ⁠Avg percent change with light scatterplot as group data for the available single unit/mua data


percent_change_plot = {
    'graph_dir': '/Users/katie/likhtik/CH27',
    'fname': 'group_percent_change_plot_only_good',
    'plot_spec': {
            'section': {
                'aesthetics': {
                    'border': {'top': 'FFF', 'right': 'FFF'},
                    'default': {'label': {'ax': {'axis': ('', 'Percent Change in Firing Rates')}}}
                    },
                'divisions': {
                'data_source': {
                    'type': 'group', 
                    'members': 'all_groups', 
                    'dim': 1}},
                'segment': {
                    'aesthetics': {
                        'period_type': {
                            'prelight': {'background_color': ('white', .2)},
                            'light': {'background_color': ('green', .2)}}
                },
                    'layers': [
                        {'plot_type': 'categorical_scatter', 
                        'aesthetics': {'default': {'cat_width': 5, 'spacing': .2, 'marker': {'color': 'black'}}},
                            'attr': 'greatgrandchildren_scatter'}, 
                        {'plot_type': 'categorical_line', 
                        'attr': 'mean', 
                        'aesthetics': {'default': {'cat_width': 5, 'spacing': .2, 'marker': {'colors': 'blue', 
                                                    'linestyles': '--'}}}}],
                    'divisions': {
                        'period_type': {
                            'members': ['prelight', 'light'], 
                            'grouping': 0}}
                    }
            }}
}


CH27_PERCENT_CHANGE_OPTS = {
    'procedure': 'make_plots',
    'graph_opts': percent_change_plot,
    'calc_opts': {'kind_of_data': 'spike', 'calc_type': 'firing_rates', 'base': 'period',
                    'bin_size': .01, 'percent_change': {'level': 'unit', 'reference': 'prelight'},
                     'filter': {'unit': {'category':  ('==', 'good')}}}
}



units_percent_change_plots = {
    'graph_dir': '/Users/katie/likhtik/CH27',
    'fname': 'units_percent_change_plots',
    'plot_spec': {
            'section': {
                'aesthetics': {
                    'label': {'ax': {'title': ('identifier',), 'component': {'axis': 'default'}}},
                    'border': {'top': 'FFF', 'right': 'FFF'},
                    '': {}
                    },
                'divisions': {
                'data_source': {
                    'type': 'unit', 
                    'members': 'all_units', 
                    'dim': 1}},
                'segment': {
                    'aesthetics': {
                         
                        'period_type': {
                            'prelight': {'background_color': ('white', .2)},
                            'light': {'background_color': ('green', .2)}}
                },
                    'layers': [
                        {'plot_type': 'categorical_scatter', 
                        'aesthetics': {'default': {'cat_width': 5, 'spacing': .2, 'marker': {'color': 'black'}},
                                       'label': {'ax': {'title': [('identifier',)]}},},
                        'attr': 'scatter'}, 
                        {'plot_type': 'categorical_line', 
                        'attr': 'mean', 
                        'aesthetics': {'default': {'cat_width': 5, 'spacing': .2, 'marker': {'colors': 'blue', 
                                                    'linestyles': '--'}}}}],
                    'divisions': {
                        'period_type': {
                            'members': ['prelight', 'light'], 
                            'grouping': 0}}
                    }
            }}
}

CH27_UNITS_PERCENT_CHANGE_OPTS = {
    'procedure': 'make_plots',
    'graph_opts': units_percent_change_plots,
    'calc_opts': {'kind_of_data': 'spike', 'calc_type': 'firing_rates', 'base': 'period',
                    'bin_size': .01, 'percent_change': {'level': 'unit', 'reference': 'prelight'},
                     'filter': {'unit': {'category':  ('==', 'good')}}}
                    
}

raster_plot = {
    'graph_dir': '/Users/katie/likhtik/CH27',
    'fname': 'raster_plot',
    'plot_spec': {
        'section': {
            'aesthetics': {'ax': {'border': {'top': 'FFF', 'right': 'FFF', 'left': ['T', 'F', 'F']}, 
                                    'share': ['x']},
                            'default': {
                                'label': {
                                    'ax': {'row_labels': 
                                           "lambda x: [s.replace('_', ' ')[2:] for s in x.get_stack(depth=2, attr='identifier').flatten()]"}},
                                'marker': {'colors': 'black'}},
                            
                            },
            'break_axis': {0: [(0, 20), (35, 55)]},
            'attr': 'grandchildren_stack',
            'divisions': {
                'data_source': {
                    'type': 'group',
                    'members': 'all_groups',
                    'dim': 0
                },
                'period_type': {
                    'members': ['light']
                }
        }}},
        
        'plot_type': 'raster'
        }
        
        




CH27_UNITS_RASTER_OPTS = {
    'procedure': 'make_plots',
    'graph_opts': raster_plot,
    'calc_opts': {'kind_of_data': 'spike', 'calc_type': 'raster', 'raster_type': 'spike_train', 
                    'base': 'period', 'bin_size': .5, 'sort_by': ('category', 'descending'),
                    'periods': {'light': {'period_pre_post': (10, 10)}},
                    'filter': {'period': {'identifier': ('==', 0)}}}
                    }
 

group_psth_plots = {
    'graph_dir':  '/Users/katie/likhtik/CH27',
    'plot_type': 'psth',
    'fname': 'group_psth_only_good',
    'plot_spec': {
            'section': {
                'aesthetics': {
                    'ax': {'border': {'top': 'FFF', 'right': 'FFF'}, 'share': ['x']},
                    'default': 
                               {'marker': {'color': 'black'},
                               'label': {'component': {'axis': ('Time (s)', 'Firing Rate (Spikes per Second)',)}}}},
                'break_axis': {0: [(0, 20), (35, 55)]},
                'attr': 'calc',
                'divisions': {
                    'data_source': {
                        'type': 'group',
                        'members': 'all_groups',
                        'dim': 0
                    },
                    'period_type': {
                        'members': ['light']
                    }
            }}}

}

CH27_GROUP_PSTH_OPTS = {
    
    'procedure': 'make_plots',
    'graph_opts': group_psth_plots,
    'calc_opts': {'kind_of_data': 'spike', 'calc_type': 'firing_rates', 'raster_type': 'spike_train', 
                  'base': 'period', 'bin_size': .5, 
                  'periods': {'light': {'period_pre_post': (10, 10)}},
                   'filter': {'unit': {'category':  ('==', 'good')}}}
    }


units_waveform_plots = {
    'graph_dir': '/Users/katie/likhtik/CH27',
    'fname': 'units_waveform',
    'plot_type': 'waveform',
    'plot_spec': {
        'section': {
            'aesthetics': {
                'ax': {'border': {'top': 'FFF', 'right': 'FFF', 'left': 'FFF', 'bottom': 'FFF'}},
                'default': {
                    'marker': {'color': 'black'},
                    'label': {'ax': {'title': [('identifier',)]}}}
                    },
                'divisions': {
                'data_source': {
                    'type': 'unit', 
                    'members': 'all_units', 
                    'dim': 1}},
            }}
}


UNITS_WAVEFORM_PLOTS = {
    
    'procedure': 'make_plots',
    'graph_opts': units_waveform_plots,
    'calc_opts': {'kind_of_data': 'spike', 'calc_type': 'waveform',  
                  'base': 'unit',
                   'filter': {'unit': {'category':  ('==', 'good')}}}
    }


CSV_OPTS = {

    'procedure': 'make_csv',

    'calc_opts': {'kind_of_data': 'spike', 'calc_type': 'firing_rates', 'raster_type': 'spike_train', 
                  'base': 'period', 'bin_size': .01, 'row_type': 'spike_period',
                  'data_path': '/Users/katie/likhtik/CH27mice', 'percent_change': {'level': 'unit', 'reference': 'prelight'}}

}



new_percent_change_plot = {
    'graph_dir': '/Users/katie/likhtik/CH27',
    'fname': 'new_percent_change_plot',
    'plot_spec': {
            'section': {
                'aesthetics': {
                    'ax': {'border': {'top': 'FFF', 'right': 'FFF'}},
                    'default': {'label': {'ax': {'axis': ('', 'Percent Change in Firing Rates')}}}
                    },
                'divisions': {
                'data_source': {
                    'type': 'group', 
                    'members': 'all_groups', 
                    'dim': 1}},
                'segment': {
                    'aesthetics': {
                        'period_type': {
                            'prelight': {'background_color': ('white', .2)},
                            'light': {'background_color': ('green', .2)}}
                },
                    'layers': [
                        {'plot_type': 'categorical_scatter', 
                        'aesthetics': {'default': {'cat_width': 5, 'spacing': .2, 'marker': {'color': 'black'}}},
                            'attr': 'greatgrandchildren_scatter'}, 
                        {'plot_type': 'categorical_line', 
                        'attr': 'mean', 
                        'aesthetics': {'default': {'cat_width': 5, 'spacing': .2, 'marker': {'colors': 'blue', 
                                                    'linestyles': '--'}}}}],
                    'divisions': {
                        'period_type': {
                            'members': ['light'], 
                            'grouping': 0}}
                    }
            }}
}


NEW_PERCENT_CHANGE_OPTS = {
    'procedure': 'make_plots',
    'graph_opts': new_percent_change_plot,
    'calc_opts': {'kind_of_data': 'spike', 'calc_type': 'firing_rates', 'base': 'period',
                    'bin_size': .01, 'percent_change': {'level': 'unit', 'reference': 'prelight'}}
}



animal_percent_change_plot = {
    'graph_dir': '/Users/katie/likhtik/CH27',
    'fname': 'new_percent_change_plot',
    'plot_spec': {
            'section': {
                'aesthetics': {
                    'ax': {'border': {'top': 'FFF', 'right': 'FFF'}},
                    'default': {'label': {'ax': {'axis': ('', 'Percent Change in Firing Rates')}}}
                    },
                'divisions': {
                        'period_type': {
                            'members': ['light'], 
                            }},
                'segment': {
                    'aesthetics': {
                        'period_type': {
                            'prelight': {'background_color': ('white', .3)},
                            'light': {'background_color': ('green', .3)}}
                },
                    'layers': [
                        {'plot_type': 'categorical_scatter', 
                        'aesthetics': {'default': {'cat_width': 3.5, 'spacing': .3, 'marker': {'color': 'black'}}},
                            'attr': 'grandchildren_scatter'}, 
                        {'plot_type': 'categorical_line', 
                        'attr': 'mean', 
                        'aesthetics': {'default': {'cat_width': 3.5, 'spacing': .3, 'marker': {'colors': 'blue', 
                                                    'linestyles': '--'}}}}],
                   'divisions': {
                'data_source': {
                    'type': 'animal', 
                    'members': 'all_animals', 
                    'dim': 1,
                    'grouping': 0}}
                    }
            }}
}




ANIMAL_PERCENT_CHANGE_OPTS = {
    'procedure': 'make_plots',
    'graph_opts': animal_percent_change_plot,
    'calc_opts': {'kind_of_data': 'spike', 'calc_type': 'firing_rates', 'base': 'period',
                    'bin_size': .01, 'percent_change': {'level': 'unit', 'reference': 'prelight'}}
}