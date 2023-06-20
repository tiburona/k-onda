from copy import deepcopy

from plotters import Plotter
from initialize_experiment import experiment, data_type_context, neuron_type_context

plotter = Plotter(experiment, data_type_context, neuron_type_context)

'''
Helpers for procs.py
'''


def plot(data_opts, graph_opts, levels, n_types=None):

    for level in levels:
        if level == 'animal':
            n_types = n_types or ['PN', 'IN']
            for nt in n_types:
                plotter.plot(data_opts, graph_opts, level, neuron_type=nt)
        else:
            plotter.plot(data_opts, graph_opts, level)


def add_ac_keys_and_plot(levels, data_opts, graph_opts, ac_methods, ac_keys):
    for program in ac_methods:
        for level in levels:
            keys = ac_keys.get(level, [])
            for key in keys:
                ac_data_opts = {'ac_key': key, 'ac_program': program, **data_opts}
                plot(ac_data_opts, graph_opts, [level], n_types=['PN', 'IN'] if level == 'animal' else None)


def assign_vars(variables, defaults):
    for i in range(len(variables)):
        if variables[i] is None:
            variables[i] = defaults[i]
    return variables
