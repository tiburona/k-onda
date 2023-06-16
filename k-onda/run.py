from initialize_experiment import experiment, all_units
from logger import log_directory_contents
from plotters import Plotter
from spreadsheet import Spreadsheet

base_opts = {'graph_dir': '/Users/katie/likhtik/data/graphs', 'units_in_fig': 4}
psth_opts = {**base_opts, **{'data_type': 'psth', 'pre_stim': 0.05, 'post_stim': 0.65, 'bin_size': 0.01,
                             'trials': (0, 150), 'tick_step': 0.1}}
autocorr_opts = {**base_opts, **{'data_type': 'autocorr', 'pre_stim': 0.0, 'post_stim': 30.0, 'bin_size': 0.01,
                                 'trials': (0, 150, 30), 'max_lag': 99, 'tick_step': .1}}
spectrum_opts = {**autocorr_opts, **{'data_type': 'spectrum', 'bin_size': 0.01, 'max_lag': 99, 'freq_range': (3, 60),
                                     'tick_step': .1}}
spreadsheet_opts = {**base_opts, **{'data_type': 'psth', 'pre_stim': 0.0, 'post_stim': 1.00, 'bin_size': 0.01,
                                    'trials': (0, 150), 'tick_step': 0.1}}


def main():
    all_group_keys = ['group_by_animal_by_unit_by_trials', 'group_by_animal_by_unit_by_rates',
                      'group_by_animal_by_rates', 'group_by_rates']
    all_animal_keys = ['animal_by_unit_by_trials', 'animal_by_unit_by_rates', 'animal_by_rates']
    all_methods = ['np']
    opts_list = [psth_opts, autocorr_opts, spectrum_opts]
    neuron_types = ['PN', 'IN']
    for opts in opts_list:
        for meth in all_methods:
            for group in experiment.groups:
                for neuron_type in neuron_types:
                    for key in all_animal_keys:
                        opts_with_ac = {**opts, **{'ac_program': meth, 'ac_key': key}}
                        Plotter(opts_with_ac).plot_animals(group, neuron_type=neuron_type, sem=True)
                for animal in group.animals:
                    for key in ['unit_by_trials', 'unit_by_rates']:
                        opts_with_ac = {**opts, **{'ac_program': meth, 'ac_key': key}}
                        sem = True if 'trials' in key else False
                        Plotter(opts_with_ac).plot_units(animal, sem=sem)
            for key in all_group_keys:
                opts_with_ac = {**opts, **{'ac_program': meth, 'ac_key': key}}
                Plotter(opts_with_ac).plot_groups(experiment.groups, neuron_types, sem=True)

    log_directory_contents('/Users/katie/likhtik/data/logdir')

# def main():
#     sheet = Spreadsheet()
#     sheet.make_spreadsheet(spreadsheet_opts, all_units, '/Users/katie/likhtik/data/firing_rates_by_unit.csv')


if __name__ == '__main__':
    main()
