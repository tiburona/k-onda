from initialize_experiment import experiment
from logger import log_directory_contents
from plotters import Plotter


base_opts = {'graph_dir': '/Users/katie/likhtik/data/graphs', 'units_in_fig': 4}
psth_opts = {**base_opts, **{'data_type': 'psth', 'pre_stim': 0.05, 'post_stim': 0.65, 'bin_size': 0.01,
                             'trials': (0, 150), 'tick_step': 0.1}}
autocorr_opts = {**base_opts, **{'data_type': 'autocorr', 'pre_stim': 0.0, 'post_stim': 30.0, 'bin_size': 0.01,
                                 'trials': (0, 150, 30), 'max_lag': 99, 'tick_step': .1}}
spectrum_opts = {**autocorr_opts, **{'data_type': 'spectrum', 'bin_size': 0.01, 'max_lag': 99, 'up_to_hz': 100,
                                      'tick_step': .1}}

# for opts_type in [autocorr_opts, frequency_opts]:
#     experiment.plot_groups(opts_type)
#     for group in experiment.groups:
#         group.plot_animals(opts_type)
#         for animal in group.animals:
#             animal.plot_units(opts_type)



def main():
    all_group_tags = ['group_by_animal_by_unit_by_trials', 'group_by_animal_by_unit_by_rates',
                      'group_by_animal_by_rates', 'group_by_rates']
    all_methods = ['pd', 'np', 'ml']
    #result = experiment.get_all_autocorrelations(autocorr_opts, method='np', neuron_type='IN')
    for meth in all_methods:
        for tag in all_group_tags:
            Plotter(autocorr_opts).plot_groups(
                experiment.groups, ['PN', 'IN'], ac_info={'method': meth, 'mean_correction': 'none', 'tag': tag})
            Plotter(spectrum_opts).plot_groups(
                experiment.groups, ['PN', 'IN'], ac_info={'method': meth, 'mean_correction': 'none', 'tag': tag})
    log_directory_contents('/Users/katie/likhtik/data/logdir')


if __name__ == '__main__':
    main()

