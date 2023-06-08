from initialize_experiment import experiment
from logger import log_directory_contents
from plotters import Plotter

base_opts = {'graph_dir': '/Users/katie/likhtik/data/graphs', 'units_in_fig': 4}
psth_opts = {**base_opts, **{'data_type': 'psth', 'pre_stim': 0.05, 'post_stim': 0.65, 'bin_size': 0.01,
                             'trials': (0, 150), 'tick_step': 0.1}}
autocorr_opts = {**base_opts, **{'data_type': 'autocorr', 'pre_stim': 0.0, 'post_stim': 30.0, 'bin_size': 0.01,
                                 'trials': (0, 150, 30), 'max_lag': 99, 'tick_step': .1}}
spectrum_opts = {**autocorr_opts, **{'data_type': 'spectrum', 'bin_size': 0.01, 'max_lag': 99, 'freq_range': (3, 60),
                                     'tick_step': .1}}


def main():
    all_group_tags = ['group_by_animal_by_unit_by_trials', 'group_by_animal_by_unit_by_rates',
                      'group_by_animal_by_rates', 'group_by_rates']
    all_animal_tags = ['animal_by_unit_by_trials', 'animal_by_unit_by_rates', 'animal_by_rates']
    all_unit_tags = ['unit_by_trials', 'unit_by_rates']
    all_methods = ['np', 'ml', 'pd']
    opts_list = [spectrum_opts]
    neuron_types = ['PN', 'IN']
    for meth in all_methods:
        for opts in opts_list:
            # for group in experiment.groups:
            #     for neuron_type in neuron_types:
            #         for tag in all_animal_tags:
            #             Plotter(opts).plot_animals(
            #                 group, neuron_type=neuron_type,
            #                 ac_info={'method': meth, 'mean_correction': 'none', 'tag': tag})
            #     for animal in group.animals:
            #         for tag in all_unit_tags:
            #             Plotter(opts).plot_units(animal,
            #                                      ac_info={'method': meth, 'mean_correction': 'none', 'tag': tag})
            for tag in all_group_tags:
                Plotter(opts).plot_groups(
                    experiment.groups, neuron_types, ac_info={'method': meth, 'mean_correction': 'none', 'tag': tag})

    log_directory_contents('/Users/katie/likhtik/data/logdir')


if __name__ == '__main__':
    main()
