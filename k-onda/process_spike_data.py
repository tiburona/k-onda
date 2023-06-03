from initialize_experiment import experiment
from logging import log_directory_contents
from plotters import Plotter


base_opts = {'graph_dir': '/Users/katie/likhtik/data/graphs', 'units_in_fig': 4}
psth_opts = {**base_opts, **{'data_type': 'psth', 'pre_stim': 0.05, 'post_stim': 0.65, 'bin_size': 0.01,
                             'trials': (0, 150), 'tick_step': 0.1}}
autocorr_opts = {**base_opts, **{'data_type': 'autocorr', 'pre_stim': 0.0, 'post_stim': 30.0, 'bin_size': 0.01,
                                 'trials': (0, 150, 30), 'max_lag': 99, 'tick_step': .1}}
frequency_opts = {**autocorr_opts, **{'data_type': 'spectrum', 'bin_size': 0.1, 'max_lag': 99, 'up_to_hz': 25,
                                      'tick_step': .1}}

# for opts_type in [autocorr_opts, frequency_opts]:
#     experiment.plot_groups(opts_type)
#     for group in experiment.groups:
#         group.plot_animals(opts_type)
#         for animal in group.animals:
#             animal.plot_units(opts_type)

result = experiment.get_all_autocorrelations(autocorr_opts, method='np', neuron_type='IN')

Plotter(autocorr_opts).plot_groups(experiment.groups, ['PN', 'IN'], ac_info={'method': 'np', 'tag': 'group_by_rates'})

log_directory_contents('/Users/katie/likhtik/data/logdir')