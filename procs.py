from copy import deepcopy

from opts_library import PSTH_OPTS, AUTOCORR_OPTS, SPECTRUM_OPTS, SPREADSHEET_OPTS, PROPORTION_OPTS, GRAPH_OPTS, \
    GROUP_STAT_PSTH_OPTS, GROUP_STAT_PROPORTION_OPTS, AC_KEYS, AC_METHODS, FIGURE_1_OPTS, LFP_OPTS, ROSE_PLOT_OPTS, \
    HEAT_MAP_DATA_OPTS, BEHAVIOR_OPTS, CAROLINA_OPTS, CAROLINA_GRAPH_OPTS, CAROLINA_GROUP_STAT_OPTS, SPONTANEOUS_OPTS, \
    SPONTANEOUS_GRAPH_OPTS, CROSS_CORR_OPTS, SPONTANEOUS_MRL_OPTS
from initialize_experiment import experiment, data_type_context, neuron_type_context, period_type_context, \
    lfp_experiment, behavior_experiment
from proc_helpers import add_ac_keys_and_plot, assign_vars, plot
from stats import Stats
from plotters import Plotter, MRLPlotter, NeuronTypePlotter, PeriStimulusPlotter


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
    stats = Stats(experiment, data_type_context, neuron_type_context, lfp_opts[0], lfp=lfp_experiment)
    df_name = stats.make_dfs(lfp_opts, )
    stats.make_spreadsheet(df_name, name_suffix='lfp_power_bla_by_period')


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
                    my_lfp_opts['adjustment'] = adjustment
                    copy_lfp_opts = deepcopy(my_lfp_opts)
                    plotter.mrl_vals_plot(copy_lfp_opts, graph_opts)


def make_mrl_heat_maps(lfp_opts=None, graph_opts=None):
    for brain_region in ['pl', 'bla', 'hpc']:
        my_lfp_opts, graph_opts = assign_vars([lfp_opts, graph_opts], [HEAT_MAP_DATA_OPTS, GRAPH_OPTS])
        plotter = MRLPlotter(experiment, data_type_context, neuron_type_context, period_type_context,
                             lfp=lfp_experiment)
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
    stats = Stats(experiment, data_type_context, neuron_type_context, opts_dicts[0], lfp=lfp_experiment,
                  behavior=behavior_experiment)
    df_name = stats.make_dfs([psth_opts] + opts_dicts + [behavior_opts])
    stats.make_spreadsheet(df_name, name_suffix='previous_pip_lfp_05_firing_rate')


def test_mrl_post_hoc_results(lfp_opts=None):
    my_lfp_opts = assign_vars([lfp_opts], [LFP_OPTS])
    stats = Stats(experiment, data_type_context, neuron_type_context, my_lfp_opts[0], lfp=lfp_experiment)
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
    stats = Stats(experiment, data_type_context, neuron_type_context, opts_dicts[0], lfp=lfp_experiment,
                  behavior=behavior_experiment)
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
    plotter = NeuronTypePlotter(experiment, data_type_context, neuron_type_context, period_type_context,
                                graph_opts=graph_opts)
    plotter.scatterplot()


def plot_carolina_waveforms(graph_opts=None):
    graph_opts = assign_vars([graph_opts], [CAROLINA_GRAPH_OPTS])[0]
    plotter = NeuronTypePlotter(experiment, data_type_context, neuron_type_context, period_type_context,
                                graph_opts=graph_opts)
    plotter.phy_graphs()


def plot_spontaneous_firing(spontaneous_opts=None, graph_opts=None):
    spontaneous_opts, graph_opts = assign_vars([spontaneous_opts, graph_opts],
                                               [SPONTANEOUS_OPTS, SPONTANEOUS_GRAPH_OPTS])
    plotter = PeriStimulusPlotter(experiment, data_type_context, neuron_type_context, period_type_context,
                                  graph_opts=graph_opts)
    plotter.plot(spontaneous_opts, graph_opts, level='group')


def plot_cross_correlations(cross_corr_opts=None, graph_opts=None):
    cross_corr_opts, graph_opts = assign_vars([cross_corr_opts, graph_opts], [CROSS_CORR_OPTS, CAROLINA_GRAPH_OPTS])
    plot(cross_corr_opts, graph_opts, levels=['group'], n_types=['PV_IN', 'ACH'])


def plot_spontaneous_mrl(spontaneous_opts=None, graph_opts=None):
    spontaneous_opts, graph_opts = assign_vars([spontaneous_opts, graph_opts], [SPONTANEOUS_MRL_OPTS, CAROLINA_GRAPH_OPTS])
    for brain_region in ['bla', 'il']:
        my_data_opts = deepcopy(spontaneous_opts)
        my_data_opts['brain_region'] = brain_region
        plotter = MRLPlotter(experiment, data_type_context, neuron_type_context, period_type_context, lfp=lfp_experiment)
        plotter.mrl_vals_plot(spontaneous_opts, graph_opts)







