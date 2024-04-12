import json
import os

from plotters import PeriStimulusPlotter, GroupStatsPlotter, PiePlotter, NeuronTypePlotter, MRLPlotter, LFPPlotter
from stats import Stats
from initialize_experiment import Initializer

peristimulus_plots = {
    f"plot_{data_type}": {'class': PeriStimulusPlotter, 'method': 'plot'}
    for data_type in [
        'psth', 'proportion', 'autocorrelation', 'spectrum', 'cross_correlation,' 'autocorrelogram'
    ]}

mrl_procs = {meth: {'class': MRLPlotter, 'method': meth} for meth in ['mrl_bar_plot', 'mrl_rose_plot', 'mrl_heat_map']}

other_procedures = {
    'plot_group_stats': {'class': GroupStatsPlotter, 'method': 'plot_group_stats'},
    'make_spreadsheet': {'class': Stats, 'method': 'make_df', 'follow_up': 'make_spreadsheet'},
    'unit_upregulation_pie_chart': {'class': PiePlotter, 'method': 'unit_upregulation_pie_chart'},
    'neuron_type_scatterplot': {'class': NeuronTypePlotter, 'method': 'scatterplot'},
    'plot_waveforms': {'class': NeuronTypePlotter, 'method': 'plot_waveforms'},
    'plot_power': {'class': LFPPlotter, 'method': 'plot_power'},
    'plot_coherence': {'class': LFPPlotter, 'method': 'plot_coherence'},
    'plot_coherence_over_frequencies':  {'class': LFPPlotter, 'method': 'plot_coherence_over_frequencies'},
    'plot_spectrogram': {'class': LFPPlotter, 'method': 'plot_spectrogram'}
}

PROCEDURE_DICT = {**peristimulus_plots, **mrl_procs, **other_procedures}


class Runner:

    def __init__(self, config_file=None, lfp=False, behavior=False):
        self.config = config_file if config_file else os.getenv('INIT_CONFIG')
        self.initializer = Initializer(self.config)
        self.experiment = self.initializer.init_experiment()
        self.lfp = lfp
        self.behavior = behavior
        self.executing_class = None
        self.executing_instance = None
        self.executing_instances = {}
        self.executing_method = None
        self.loop_lists = {}
        self.follow_up_method = None
        self.data_opts = None
        self.graph_opts = None
        self.proc_name = None
        self.current_data_opts = None
        self.preparatory_method = None

    def load_analysis_config(self, opts):
        if isinstance(opts, str):
            try:
                with open(opts, 'r', encoding='utf-8') as file:
                    data = file.read()
                    opts = json.loads(data)
            except FileNotFoundError:
                raise Exception(f"File not found: {opts}")
            except json.JSONDecodeError:
                raise Exception(f"Error decoding JSON from the file: {opts}")
        if isinstance(opts, list):
            self.data_opts = opts
        else:
            self.data_opts = opts.get('data_opts', {})
            self.graph_opts = opts.get('graph_opts', None)

    def prepare(self):
        self.executing_class = PROCEDURE_DICT[self.proc_name]['class']
        self.data_class_kwargs = {dc: getattr(self.initializer, f"init_{dc}_experiment")() 
                                  for dc in ['lfp', 'behavior'] if getattr(self, dc)}

        if self.executing_class.__name__ in self.executing_instances:
            self.executing_instance = self.executing_instances[self.executing_class.__name__]
        else:
            self.executing_instance = self.executing_class(self.experiment, 
                                                           **self.data_class_kwargs)
        method = PROCEDURE_DICT[self.proc_name].get('method')
        if method is None:
            method = self.proc_name
        self.executing_method = getattr(self.executing_instance, method)
        self.follow_up_method = PROCEDURE_DICT[self.proc_name].get('follow_up')

    def read_inputs(self, proc_name, opts):
        self.proc_name = proc_name
        self.load_analysis_config(opts)
        

    def get_loop_lists(self):
        for opt_list_key in ['brain_regions', 'frequency_bands', 'levels', 'unit_pairs', 
                             'neuron_qualities', 'inclusion_rules', 'coherence_region_sets']:
            opt_list = self.current_data_opts.get(opt_list_key)
            if opt_list is not None:
                self.loop_lists[opt_list_key] = opt_list

    def iterate_loop_lists(self, remaining_loop_lists, current_index=0):
        if current_index >= len(remaining_loop_lists):
            self.execute()
            return
        opt_list_key, opt_list = remaining_loop_lists[current_index]
        for opt in opt_list:
            key = opt_list_key[:-1] if opt_list_key != 'neuron_qualities' else 'neuron_quality'
            self.current_data_opts[key] = opt
            self.iterate_loop_lists(remaining_loop_lists, current_index + 1)

    def apply_rules(self):
        rules = self.current_data_opts['rules']
        # Assuming rules is a dictionary like: {'data_type': {'mrl': [('time_type', 'block')]}}
        for data_key, conditions in rules.items():
            if data_key not in self.current_data_opts:
                raise ValueError(f"Key '{data_key}' not found in data_opts")
            for trigger_val, vals_to_assign in conditions.items():
                for target_key, val_to_assign in vals_to_assign:
                    if self.current_data_opts[data_key] == trigger_val:
                        self.current_data_opts[target_key] = val_to_assign

    def validate(self):
        # TODO: update animal selection validation
        all_animal_ids = [animal.identifier for animal in self.experiment.all_animals]
        selected_animals = self.current_data_opts.get('selected_animals')
        if selected_animals is not None and not all([id in all_animal_ids for id in selected_animals]):
            raise ValueError("Missing animals")
        if self.current_data_opts.get('adjustment') != 'none':
            if self.current_data_opts.get('evoked'):
                raise ValueError("It does not make sense to set 'evoked' to True and 'adjustment' to anything other "
                                 "than 'none'.  See Analysis Configuration Reference.")
            if not self.current_data_opts.get('periods'):
                raise ValueError("You picked a value for adjustment other than 'none' and haven't specified which "
                                 "periods to include.  This will result in a nonsensical result.  See the Analysis "
                                 "Configuration Reference.")

    def execute(self):
        if self.current_data_opts.get('rules'):
            self.apply_rules()
        self.validate()
        print(f"executing {self.executing_method} with options {self.current_data_opts}")
        if self.graph_opts is not None:
            self.executing_method(self.current_data_opts, self.graph_opts)
        else:
            self.executing_method(self.current_data_opts)

    def run_prep(self, prep):
        prep_fun, prep_opts = prep
        if prep_opts['data_class'] == 'spike':
            prep_executor = self.experiment
        else:
            prep_executor = self.data_class_kwargs[prep_opts['data_class']]

        getattr(prep_executor, prep_fun)(prep_opts)

    def run_all(self, prep=None):

        if prep is not None:
            self.run_prep(prep)

        opts_list = self.data_opts if isinstance(self.data_opts, list) else [self.data_opts]
            
        for opts in opts_list:
            self.current_data_opts = opts
            self.get_loop_lists()
            if self.loop_lists:
                self.iterate_loop_lists(list(self.loop_lists.items()))
            else:
                self.execute()

    def run(self, proc_name, opts, *args, prep=None, **kwargs):
        self.read_inputs(proc_name, opts)
        self.prepare()
        self.run_all(prep=prep)
        if self.follow_up_method is not None:
            getattr(self.executing_instance, self.follow_up_method)(*args, **kwargs)






