from copy import deepcopy

from opts_library import PSTH_OPTS, AUTOCORR_OPTS, SPECTRUM_OPTS, SPREADSHEET_OPTS, PROPORTION_OPTS, GRAPH_OPTS, \
    GROUP_STAT_PSTH_OPTS, GROUP_STAT_PROPORTION_OPTS, AC_KEYS, AC_METHODS, FIGURE_1_OPTS, LFP_OPTS, ROSE_PLOT_OPTS
from initialize_experiment import experiment, data_type_context, neuron_type_context, period_type_context, \
    lfp_experiment
from proc_helpers import add_ac_keys_and_plot, assign_vars, plot
from stats import Stats
from plotters import Plotter, MRLPlotter


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
    plot(psth_opts, graph_opts, levels=levels)


def plot_autocorr(levels, autocorr_opts=None, graph_opts=None, ac_methods=None, ac_keys=None):
    plot_autocorr_or_spectrum(levels, autocorr_opts, graph_opts, ac_methods, ac_keys, AUTOCORR_OPTS)


def plot_spectrum(levels, spectrum_opts=None, graph_opts=None, ac_methods=None, ac_keys=None):
    plot_autocorr_or_spectrum(levels, spectrum_opts, graph_opts, ac_methods, ac_keys, SPECTRUM_OPTS)


def plot_proportion_score(levels, proportion_opts=None, graph_opts=None):
    proportion_opts, graph_opts = assign_vars([proportion_opts, graph_opts], [PROPORTION_OPTS, GRAPH_OPTS])
    plot(proportion_opts, graph_opts, levels=levels)


def make_spreadsheet(spreadsheet_opts=None):
    spreadsheet_opts, = assign_vars([spreadsheet_opts], [SPREADSHEET_OPTS])
    stats = Stats(experiment, data_type_context, spreadsheet_opts)
    stats.make_spreadsheet()


def plot_psth_group_stats(group_stat_opts=None, graph_opts=None):
    group_stat_opts, graph_opts = assign_vars([group_stat_opts, graph_opts], [GROUP_STAT_PSTH_OPTS, GRAPH_OPTS])
    plot(group_stat_opts, graph_opts)


def plot_proportion_group_stats(group_stat_opts=None, graph_opts=None):
    group_stat_opts, graph_opts = assign_vars([group_stat_opts, graph_opts], [GROUP_STAT_PROPORTION_OPTS, GRAPH_OPTS])
    plot(group_stat_opts, graph_opts)


def plot_pie_chart(psth_opts=None, graph_opts=None):
    psth_opts, graph_opts = assign_vars([psth_opts, graph_opts], [PSTH_OPTS, GRAPH_OPTS])
    plotter = Plotter(experiment, data_type_context, neuron_type_context, graph_opts=None)
    plotter.plot_unit_pie_chart(psth_opts, graph_opts)


def make_lfp_firing_rate_spreadsheet(spreadsheet_opts=None, lfp_opts=None):
    spreadsheet_opts, lfp_opts = assign_vars([spreadsheet_opts, lfp_opts], [SPREADSHEET_OPTS, LFP_OPTS])
    stats = Stats(experiment, data_type_context, neuron_type_context, lfp_opts)
    df_name = stats.make_dfs((lfp_opts, spreadsheet_opts))
    stats.make_spreadsheet(df_name)


def make_lfp_spreadsheet(lfp_opts=None):
    lfp_opts = assign_vars([lfp_opts], [LFP_OPTS])
    stats = Stats(experiment, data_type_context, neuron_type_context, lfp_opts[0])
    df_name = stats.make_dfs(lfp_opts, )
    stats.make_spreadsheet(df_name)


def make_all_mrl_spreadsheets(lfp_opts=None):
    for brain_region in ['pl', 'bla', 'hpc']:
        my_lfp_opts = assign_vars([lfp_opts], [LFP_OPTS])
        my_lfp_opts[0]['brain_region'] = brain_region  # I found it! it's a pass by reference error
        for fb in ['delta', 'theta_1', 'theta_2', 'delta', 'gamma', 'hgamma']:
            my_lfp_opts[0]['fb'] = [fb]
            for phase_opt in ['wavelet', None]:
                my_lfp_opts[0]['phase'] = phase_opt
                copy_lfp_opts = deepcopy(my_lfp_opts[0])
                stats = Stats(experiment, data_type_context, neuron_type_context, copy_lfp_opts, lfp=lfp_experiment)
                df_name = stats.make_dfs([copy_lfp_opts],)
                stats.make_spreadsheet(df_name)


def make_all_rose_plots(lfp_opts=None, graph_opts=None):
    for brain_region in ['pl', 'bla', 'hpc']:
        my_lfp_opts, graph_opts = assign_vars([lfp_opts, graph_opts], [LFP_OPTS, ROSE_PLOT_OPTS])
        plotter = MRLPlotter(experiment, data_type_context, neuron_type_context, period_type_context,
                              lfp=lfp_experiment)
        my_lfp_opts['brain_region'] = brain_region
        for phase_opt in ['wavelet', None]:
            my_lfp_opts['phase'] = phase_opt
            for fb in ['delta', 'theta_1', 'theta_2', 'delta', 'gamma', 'hgamma']:
                my_lfp_opts['frequency_band'] = fb
                for adjustment in [None, 'relative']:
                    my_lfp_opts['adjustment'] = adjustment
                    copy_lfp_opts = deepcopy(my_lfp_opts)
                    plotter.rose_plot(copy_lfp_opts, graph_opts)


def make_all_mrl_plots(lfp_opts=None, graph_opts=None):
    print('what is happening?')
    for brain_region in ['pl', 'bla', 'hpc']:
        my_lfp_opts, graph_opts = assign_vars([lfp_opts, graph_opts], [LFP_OPTS, ROSE_PLOT_OPTS])
        plotter = MRLPlotter(experiment, data_type_context, neuron_type_context, period_type_context,
                             lfp=lfp_experiment)
        my_lfp_opts['brain_region'] = brain_region
        for phase_opt in ['wavelet', None]:
            my_lfp_opts['phase'] = phase_opt
            for fb in ['delta', 'theta_1', 'theta_2', 'delta', 'gamma', 'hgamma']:
                my_lfp_opts['frequency_band'] = fb
                for adjustment in [None]:
                    print("hmm")
                    my_lfp_opts['adjustment'] = adjustment
                    copy_lfp_opts = deepcopy(my_lfp_opts)
                    plotter.mrl_vals_plot(copy_lfp_opts, graph_opts)