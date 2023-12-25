import os
from copy import deepcopy

from opts_library import PSTH_OPTS, AUTOCORR_OPTS, SPECTRUM_OPTS, SPREADSHEET_OPTS, PROPORTION_OPTS, GRAPH_OPTS, \
    GROUP_STAT_PSTH_OPTS, GROUP_STAT_PROPORTION_OPTS, AC_KEYS, AC_METHODS, FIGURE_1_OPTS, LFP_OPTS, ROSE_PLOT_OPTS, \
    HEAT_MAP_DATA_OPTS, BEHAVIOR_OPTS, CAROLINA_OPTS, CAROLINA_GRAPH_OPTS, CAROLINA_GROUP_STAT_OPTS, SPONTANEOUS_OPTS, \
    SPONTANEOUS_GRAPH_OPTS, CROSS_CORR_OPTS, SPONTANEOUS_MRL_OPTS, CAROLINA_MRL_OPTS
from initialize_experiment import Initializer
from stats import Stats
from plotters import Plotter, MRLPlotter, NeuronTypePlotter, PeriStimulusPlotter, GroupStatsPlotter

"""
Functions in this module, with the assistance of functions imported from proc_helpers, read in values of opts or other 
variables if they are provided and assign the defaults from opts_library if not. 
"""

INIT_CONFIG = os.getenv('INIT_CONFIG')
ANALYSIS_CONFIG_FILE = os.getenv('ANALYSIS_CONFIG')
INIT_CONFIG = '/Users/katie/likhtik/CH_for_katie_less_conservative/init_config.json'
initializer = Initializer(INIT_CONFIG)
experiment = initializer.init_experiment()


###
#  Proc Helper Functions
###

def plot(data_opts, graph_opts, levels=None, neuron_types=None, sig_markers=True):
    if levels is None:
        plotter = GroupStatsPlotter(experiment)
        plotter.initialize(data_opts, graph_opts)
        plotter.plot_group_stats(sig_markers=sig_markers)
        return
    plotter = PeriStimulusPlotter(experiment)
    for level in levels:
        if level == 'animal':
            n_types = neuron_types or experiment.neuron_types
            for nt in n_types:
                plotter.plot(data_opts, graph_opts, level, neuron_type=nt)
        else:
            plotter.plot(data_opts, graph_opts, level)


def assign_vars(variables, defaults):
    for i in range(len(variables)):
        if variables[i] is None:
            variables[i] = defaults[i]
    return variables


###
#  Procs
###


def plot_psth(levels, psth_opts=None, graph_opts=None):
    psth_opts, graph_opts = assign_vars([psth_opts, graph_opts], [PSTH_OPTS, CAROLINA_GRAPH_OPTS])
    plot(psth_opts, graph_opts, levels=levels)


def plot_autocorr(levels, data_opts=None, graph_opts=None):
    my_data_opts, graph_opts = assign_vars([data_opts, graph_opts], [AUTOCORR_OPTS, CAROLINA_GRAPH_OPTS])
    for ac_key in my_data_opts['ac_keys']:
        my_data_opts['ac_key'] = ac_key
        data_opts_copy = deepcopy(my_data_opts)
        plot(data_opts_copy, graph_opts, levels=levels)


def plot_spectrum(levels, data_opts=None, graph_opts=None):
    my_data_opts, graph_opts = assign_vars([data_opts, graph_opts], [SPECTRUM_OPTS, CAROLINA_GRAPH_OPTS])
    for ac_key in my_data_opts['ac_keys']:
        my_data_opts['ac_key'] = ac_key
        data_opts_copy = deepcopy(my_data_opts)
        plot(data_opts_copy, graph_opts, levels=levels)


def plot_proportion_score(levels, proportion_opts=None, graph_opts=None):
    proportion_opts, graph_opts = assign_vars([proportion_opts, graph_opts], [PROPORTION_OPTS, GRAPH_OPTS])
    plot(proportion_opts, graph_opts, levels=levels)


def make_spreadsheet(spreadsheet_opts=None):
    spreadsheet_opts, = assign_vars([spreadsheet_opts], [SPREADSHEET_OPTS])
    stats = Stats(experiment, spreadsheet_opts)
    stats.make_spreadsheet()


def plot_psth_group_stats(group_stat_opts=None, graph_opts=None):
    group_stat_opts, graph_opts = assign_vars([group_stat_opts, graph_opts], [GROUP_STAT_PSTH_OPTS, GRAPH_OPTS])
    plot(group_stat_opts, graph_opts)


def plot_proportion_group_stats(group_stat_opts=None, graph_opts=None):
    group_stat_opts, graph_opts = assign_vars([group_stat_opts, graph_opts], [GROUP_STAT_PROPORTION_OPTS, GRAPH_OPTS])
    plot(group_stat_opts, graph_opts)


def plot_pie_chart(psth_opts=None, graph_opts=None):
    psth_opts, graph_opts = assign_vars([psth_opts, graph_opts], [PSTH_OPTS, GRAPH_OPTS])
    plotter = Plotter(experiment, graph_opts=None)
    plotter.plot_unit_pie_chart(psth_opts, graph_opts)


def make_lfp_firing_rate_spreadsheet(spreadsheet_opts=None, lfp_opts=None):
    spreadsheet_opts, lfp_opts = assign_vars([spreadsheet_opts, lfp_opts], [SPREADSHEET_OPTS, LFP_OPTS])
    stats = Stats(experiment, lfp_opts)
    df_name = stats.make_dfs((lfp_opts, spreadsheet_opts))
    stats.make_spreadsheet(df_name)


def make_lfp_spreadsheet(lfp_opts=None):
    lfp_opts = assign_vars([lfp_opts], [LFP_OPTS])
    stats = Stats(experiment, lfp_opts[0], lfp=initializer.init_lfp_experiment())
    df_name = stats.make_dfs(lfp_opts, )
    stats.make_spreadsheet(df_name)


def make_all_mrl_spreadsheets(lfp_opts=None):
    for brain_region in ['pl', 'bla', 'hpc']:
        my_lfp_opts = assign_vars([lfp_opts], [LFP_OPTS])
        my_lfp_opts[0]['brain_region'] = brain_region
        for fb in ['delta', 'theta_1', 'theta_2', 'delta', 'gamma', 'hgamma']:
            my_lfp_opts[0]['fb'] = [fb]
            for phase_opt in ['wavelet', None]:
                my_lfp_opts[0]['phase'] = phase_opt
                for adjustment in [None]:
                    my_lfp_opts[0]['adjustment'] = adjustment
                    copy_lfp_opts = deepcopy(my_lfp_opts[0])
                    stats = Stats(experiment, copy_lfp_opts, lfp=initializer.init_lfp_experiment())
                    df_name = stats.make_dfs([copy_lfp_opts], )
                    stats.make_spreadsheet(df_name)


def make_all_rose_plots(lfp_opts=None, graph_opts=None):
    for brain_region in ['pl', 'bla', 'hpc']:
        my_lfp_opts, graph_opts = assign_vars([lfp_opts, graph_opts], [LFP_OPTS, ROSE_PLOT_OPTS])
        plotter = MRLPlotter(experiment, lfp=initializer.init_lfp_experiment())
        my_lfp_opts['brain_region'] = brain_region
        for phase_opt in ['wavelet', None]:
            my_lfp_opts['phase'] = phase_opt
            for fb in ['delta', 'theta_1', 'theta_2', 'delta', 'gamma', 'hgamma']:
                my_lfp_opts['frequency_band'] = fb
                for adjustment in [None, 'relative']:
                    my_lfp_opts['adjustment'] = adjustment
                    copy_lfp_opts = deepcopy(my_lfp_opts)
                    plotter.rose_plot(copy_lfp_opts, graph_opts)


def make_mrl_plots(lfp_opts=None, graph_opts=None):
    my_lfp_opts, graph_opts = assign_vars([lfp_opts, graph_opts], [LFP_OPTS, ROSE_PLOT_OPTS])
    plotter = MRLPlotter(experiment, lfp=initializer.init_lfp_experiment())
    for brain_region in lfp_opts['brain_regions']:
        my_lfp_opts['current_brain_region'] = brain_region
        for fb in my_lfp_opts['fb']:
            my_lfp_opts['current_frequency_band'] = fb
            copy_lfp_opts = deepcopy(my_lfp_opts)
            plotter.mrl_vals_plot(copy_lfp_opts, graph_opts)



def make_mrl_heat_maps(lfp_opts=None, graph_opts=None):
    for brain_region in ['pl', 'bla', 'hpc']:
        my_lfp_opts, graph_opts = assign_vars([lfp_opts, graph_opts], [HEAT_MAP_DATA_OPTS, GRAPH_OPTS])
        plotter = MRLPlotter(experiment, lfp=initializer.init_lfp_experiment())
        my_lfp_opts['brain_region'] = brain_region
        copy_lfp_opts = deepcopy(my_lfp_opts)
        plotter.make_plot(copy_lfp_opts, graph_opts, plot_type='heat_map')


def make_spike_lfp_behavior_spreadsheet(behavior_opts=None, lfp_opts=None, spike_opts=None):
    behavior_opts, my_lfp_opts, psth_opts = assign_vars([behavior_opts, lfp_opts, spike_opts],
                                                        [BEHAVIOR_OPTS, LFP_OPTS, SPREADSHEET_OPTS])
    opts_dicts = []
    for brain_region in ['bla', 'pl', 'hpc']:
        for dtype in ['mrl', 'power']:
            opts_dict = deepcopy(my_lfp_opts)
            opts_dict['brain_region'] = brain_region
            opts_dict['data_type'] = dtype
            if dtype == 'mrl':
                opts_dict['row_type'] = 'period'
                opts_dict['time_type'] = 'block'
            opts_dicts.append(opts_dict)
    stats = Stats(experiment, opts_dicts[0], lfp=initializer.init_lfp_experiment(),
                  behavior=initializer.init_behavior_experiment())
    df_name = stats.make_dfs([psth_opts] + opts_dicts + [behavior_opts])
    stats.make_spreadsheet(df_name, name_suffix='previous_pip_lfp_05_firing_rate')


def test_mrl_post_hoc_results(lfp_opts=None):
    my_lfp_opts = assign_vars([lfp_opts], [LFP_OPTS])
    stats = Stats(experiment, my_lfp_opts[0], lfp=initializer.init_lfp_experiment())
    results = stats.get_post_hoc_results()
    print(results)


def duplicate_itamar_spreadsheet(lfp_opts=None):
    my_lfp_opts = assign_vars([lfp_opts], [LFP_OPTS])
    opts_dicts = []
    for trial_duration in [(0, 1), (0, .3)]:
        opts_dict = deepcopy(my_lfp_opts[0])
        opts_dict['pre_stim'] = trial_duration[0]
        opts_dict['post_stim'] = trial_duration[1]
        opts_dicts.append(opts_dict)
    stats = Stats(experiment, opts_dicts[0], lfp=initializer.init_lfp_experiment(),
                  behavior=initializer.init_behavior_experiment())
    df_name = stats.make_dfs(opts_dicts)
    stats.make_spreadsheet(df_name, name_suffix='duplicate_itamar')


def plot_carolina_psth(levels, psth_opts=None, graph_opts=None):
    psth_opts, graph_opts = assign_vars([psth_opts, graph_opts], [CAROLINA_OPTS, CAROLINA_GRAPH_OPTS])
    plot(psth_opts, graph_opts, levels=levels, n_types=['PV_IN', 'ACH'])


def plot_carolina_group_stats(group_stat_opts=None, graph_opts=None):
    group_stat_opts, graph_opts = assign_vars([group_stat_opts, graph_opts], [CAROLINA_GROUP_STAT_OPTS,
                                                                              CAROLINA_GRAPH_OPTS])
    plot(group_stat_opts, graph_opts, sig_markers=False)


def plot_carolina_scatter(graph_opts=None):
    graph_opts = assign_vars([graph_opts], [CAROLINA_GRAPH_OPTS])[0]
    plotter = NeuronTypePlotter(experiment, graph_opts=graph_opts)
    plotter.scatterplot()


def plot_carolina_waveforms(graph_opts=None):
    graph_opts = assign_vars([graph_opts], [CAROLINA_GRAPH_OPTS])[0]
    plotter = NeuronTypePlotter(experiment, graph_opts=graph_opts)
    plotter.phy_graphs()


def plot_spontaneous_firing(spontaneous_opts=None, graph_opts=None):
    spontaneous_opts, graph_opts = assign_vars([spontaneous_opts, graph_opts],
                                               [SPONTANEOUS_OPTS, SPONTANEOUS_GRAPH_OPTS])
    plotter = PeriStimulusPlotter(experiment, graph_opts=graph_opts)
    plotter.plot(spontaneous_opts, graph_opts, level='group')


def plot_cross_correlations(cross_corr_opts=None, graph_opts=None):
    cross_corr_opts, graph_opts = assign_vars([cross_corr_opts, graph_opts], [CROSS_CORR_OPTS, CAROLINA_GRAPH_OPTS])
    plot(cross_corr_opts, graph_opts, levels=['group'])


def plot_spontaneous_mrl(spontaneous_opts=None, graph_opts=None):
    spontaneous_opts, graph_opts = assign_vars([spontaneous_opts, graph_opts],
                                               [SPONTANEOUS_MRL_OPTS, CAROLINA_GRAPH_OPTS])
    my_lfp_opts, graph_opts = assign_vars([spontaneous_opts, graph_opts], [SPONTANEOUS_MRL_OPTS, CAROLINA_GRAPH_OPTS])
    plotter = MRLPlotter(experiment, lfp=initializer.init_lfp_experiment())
    for brain_region in my_lfp_opts['brain_regions']:
        my_lfp_opts['brain_region'] = brain_region
        for fb in my_lfp_opts['frequency_bands']:
            my_lfp_opts['frequency_band'] = fb
            copy_lfp_opts = deepcopy(my_lfp_opts)
            plotter.mrl_vals_plot(copy_lfp_opts, graph_opts)


def plot_cross_correlations_by_unit_pair(cross_corr_opts=None, graph_opts=None):
    cross_corr_opts, graph_opts = assign_vars([cross_corr_opts, graph_opts], [CROSS_CORR_OPTS, CAROLINA_GRAPH_OPTS])
    plot(cross_corr_opts, graph_opts, levels=['unit_pair'])


def make_carolina_mrl_plots(lfp_opts=None, graph_opts=None):
    my_lfp_opts, graph_opts = assign_vars([lfp_opts, graph_opts], [CAROLINA_MRL_OPTS, CAROLINA_GRAPH_OPTS])
    plotter = MRLPlotter(experiment, lfp=initializer.init_lfp_experiment())
    for brain_region in my_lfp_opts['brain_regions']:
        my_lfp_opts['brain_region'] = brain_region
        for fb in my_lfp_opts['frequency_bands']:
            my_lfp_opts['frequency_band'] = fb
            copy_lfp_opts = deepcopy(my_lfp_opts)
            plotter.mrl_vals_plot(copy_lfp_opts, graph_opts)

