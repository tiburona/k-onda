from opts_library import PSTH_OPTS, AUTOCORR_OPTS, SPECTRUM_OPTS, SPREADSHEET_OPTS, PROPORTION_OPTS, GRAPH_OPTS, \
    AC_KEYS, AC_METHODS
from initialize_experiment import experiment, data_type_context
from proc_helpers import add_ac_keys_and_plot, assign_vars, plot
from spreadsheet import Spreadsheet

"""
Functions in this module, with the assistance of functions imported from proc_helpers, read in values of opts or other 
variables if they are provided and assign the defaults from opts_library if not. 
"""


def plot_autocorr_or_spectrum(levels, data_opts, graph_opts, ac_methods, ac_keys, opts_constant):
    data_opts, graph_opts, ac_methods, ac_keys = assign_vars([data_opts, graph_opts, ac_methods, ac_keys],
                                                             [opts_constant, GRAPH_OPTS, AC_METHODS, AC_KEYS])
    add_ac_keys_and_plot(levels, data_opts, graph_opts, ac_methods, ac_keys)


def plot_psth(levels, psth_opts=None, graph_opts=None):
    psth_opts, graph_opts = assign_vars([psth_opts, graph_opts], [PSTH_OPTS, GRAPH_OPTS])
    plot(psth_opts, graph_opts, levels)


def plot_autocorr(levels, autocorr_opts=None, graph_opts=None, ac_methods=None, ac_keys=None):
    plot_autocorr_or_spectrum(levels, autocorr_opts, graph_opts, ac_methods, ac_keys, AUTOCORR_OPTS)


def plot_spectrum(levels, spectrum_opts=None, graph_opts=None, ac_methods=None, ac_keys=None):
    plot_autocorr_or_spectrum(levels, spectrum_opts, graph_opts, ac_methods, ac_keys, SPECTRUM_OPTS)


def plot_proportion_score(levels, proportion_opts=None, graph_opts=None):
    proportion_opts, graph_opts = assign_vars([proportion_opts, graph_opts], [PROPORTION_OPTS, GRAPH_OPTS])
    plot(proportion_opts, graph_opts, levels)


def make_spreadsheet(spreadsheet_opts=None):
    spreadsheet_opts, = assign_vars([spreadsheet_opts], [SPREADSHEET_OPTS])
    sheet = Spreadsheet(experiment, data_type_context, spreadsheet_opts)
    sheet.make_spreadsheet()



