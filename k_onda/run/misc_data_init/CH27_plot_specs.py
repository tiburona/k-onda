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





units_percent_change_plots = {
    'graph_dir': '/Users/katie/likhtik/CH27',
    'fname': 'units_firing_rate_plots',
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
                    'bin_size': .01,
                     'filter': {'unit': {'category':  ('==', 'good')}}}
                    
}


group_psth_plots_whole_axis = {
    'plot_type': 'psth',
    'section': {
        'attr': 'calc',
        'aesthetics': {
            'ax': {'border': {'top': {'visible': 'FFF'}, 'right': {'visible': 'FFF'}}},
            'default': 
                    {'marker': {'color': 'black'},
                    }},
        #'break_axis': {0: {'splits': [(-10, 10), (25, 45)], 'dim': 'period_time'}},
        'label': {'x': 
                  {'text': 'Seconds'}, 
                  'y': {'text': 'Firing Rate (Spikes per Second)'}},        
        'divisions': [
            {
                'divider_type': 'period_type',
                'members': ['light']
            }]
    }}

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
 

group_psth_plots = {
    'graph_dir':  '/Users/katie/likhtik/CH27',
    'plot_type': 'psth',
    'fname': 'group_psth_good_and_mua',
    'plot_spec': {
            'section': {
                'aesthetics': {
                    'ax': {'border': {'top': 'FFF', 'right': 'FFF'}, 'share': ['x']},
                    'default': 
                               {'marker': {'color': 'black'},
                               'label': {'component': {'axis': ('Time (s)', 'Firing Rate (Spikes per Second)',)}}}},
                'break_axis': {0: [(0, 20), (35, 55)]},
                'attr': 'calc',
                'divisions': [
                     {
                        'type': 'group',
                        'members': 'all_groups',
                        'dim': 0
                    },
                    {   'divider_type': 'period_type',
                        'members': ['light']
                    }
            ]}}

}



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




# AS_OPTS = {
#     'procedure': 'make_plots',
#     'write_opts': {
#         'fname': {'template': '/Users/katie/likhtik/AS/power_{brain_region}_{frequency_band}',
#                   'fields': ['brain_region', 'frequency_band']}
#     },
#     'calc_opts': AS_POWER_OPTS,
#     'plot_spec': AS_SPECTRUM_PLOT_SPEC
# }

CH27_GROUP_PSTH_OPTS = {
    
    'procedure': 'make_plots',
    'plot_spec': group_psth_plots_whole_axis,
    'write_opts': '/Users/katie/likhtik/ch27/psth',
    'calc_opts': {'kind_of_data': 'spike', 'calc_type': 'firing_rates', 'raster_type': 'spike_train', 
                  'base': 'period', 'bin_size': .5, 
                  'periods': {'light': {'period_pre_post': (10, 10)}, 'prelight': {'period_pre_post': (10, 10)}} 
    }}


units_waveform_plots = {
    'graph_dir': '/Users/katie/likhtik/CH27/psth.png',
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


new_percent_change_plot = {
    'graph_dir': '/Users/katie/likhtik/CH27',
    'fname': 'new_group_percent_change_plot_diff_colors',
    'plot_spec': {
            'section': {
                'aesthetics': {
                    'ax': {'border': {'top': 'FFF', 'right': 'FFF'}},
                    'default': {'label': {'ax': {'axis': ('', 'Percent Change in Firing Rates')}}}
                    },
                'divisions': {
                'period_type': {
                    'members': ['light'], 
                    'dim': 1}},
                'segment': {
                    'layers': [
                        {'plot_type': 'categorical_scatter', 
                        'aesthetics': {'default': {'cat_width': 3, 'spacing': 2, 'marker': {'color': 'black'}},
                                       'period_type': {'light': {'background_color': ('green', .2)}}},
                        'attr': 'grandchildren_scatter',
                        'calc_opts': {'filter': {'unit': {'category':  ('==', 'good')}}}},
                        {'plot_type': 'categorical_scatter', 
                        'aesthetics': {'default': {'cat_width': 3, 'spacing': 2, 'marker': {'facecolor': 'white', 'edgecolor': 'black'}}},
                        'attr': 'grandchildren_scatter',
                        'calc_opts': {'filter': {'unit': {'category':  ('==', 'mua')}}}}, 
                        {'plot_type': 'categorical_line', 
                        'attr': 'mean', 
                        'calc_opts': {'filter': None},
                        'aesthetics': {'default': {'cat_width': 3, 'spacing': 2, 'marker': {'colors': 'blue', 
                                                    'linestyles': '--'}}}}],
                    'divisions': {

                        'data_source': {
                            'type': 'animal',
                            'members': 'all_animals',
                            'dim':1},
                            }
                    }
            }}
}


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