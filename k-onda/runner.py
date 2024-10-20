import json
import os


from opts_validator import OptsValidator
from stats import Stats
from layout import Layout
from plotters import ExecutivePlotter
from initialize_experiment import Initializer


class Runner(OptsValidator):

    def __init__(self, config_file=None):
        self.config = config_file if config_file else os.getenv('INIT_CONFIG')
        self.initializer = Initializer(self.config)
        self.experiment = self.initializer.init_experiment()
        self.executing_class = None
        self.executing_instance = None
        self.executing_instances = {}
        self.executing_method = None
        self.loop_lists = {}
        self.follow_up_method = None
        self.calc_opts = None
        self.graph_opts = None
        self.proc_name = None
        self.current_calc_opts = None
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
        self.opts = opts

    def run_main(self, opts):
        if self.proc_name == 'make_figure':
            self.set_executors(Layout, 'make_figure')
            self.executing_method(opts)
        
        elif self.proc_name == 'make_plots':
            self.set_executors(ExecutivePlotter, 'plot')
            calc_opts = opts['calc_opts']
            self.graph_opts = self.opts['graph_opts']
            opts_list = [calc_opts]
            self.run_list(opts_list)

        elif self.proc_name == 'make_csv':
            self.set_executors(Stats, 'make_df')
            self.follow_up_method = 'make_csv'
            opts_list = opts if isinstance(self.opts, list) else [opts]
            self.run_list(opts_list)

        else:
            raise ValueError("Unknown proc name")
        
    def set_executors(self, cls, method):
        self.executing_class = cls
        if self.executing_class.__name__ in self.executing_instances:
            self.executing_instance = self.executing_instances[self.executing_class.__name__]
        else:
            self.executing_instance = self.executing_class(self.experiment)
        self.executing_method = getattr(self.executing_instance, method)

    def get_loop_lists(self):
        for opt_list_key in ['brain_regions', 'frequency_bands', 'levels', 'unit_pairs', 
                             'neuron_qualities', 'inclusion_rules', 'region_sets']:
            opt_list = self.current_calc_opts.get(opt_list_key)
            if opt_list is not None:
                self.loop_lists[opt_list_key] = opt_list

    def iterate_loop_lists(self, remaining_loop_lists, current_index=0):
        if current_index >= len(remaining_loop_lists):
            self.execute()
            return
        opt_list_key, opt_list = remaining_loop_lists[current_index]
        for opt in opt_list:
            key = opt_list_key[:-1] if opt_list_key != 'neuron_qualities' else 'neuron_quality'
            self.current_calc_opts[key] = opt
            self.iterate_loop_lists(remaining_loop_lists, current_index + 1)

    def apply_rules(self):
        rules = self.current_calc_opts['rules']
        if isinstance(rules, dict):
            for rule in rules:
                self.assign_per_rule(*self.parse_natural_language_rule(rule))
        else:          
            # Assuming rules is a dictionary like: {'calc_type': {'mrl': [('time_type', 'block')]}}
            for trigger_k, conditions in rules.items():
                for trigger_v, target_vals in conditions.items():
                    for target_k, target_v in target_vals:
                        self.assign_per_rule(trigger_k, trigger_v, target_k, target_v)
    
    def assign_per_rule(self, trigger_k, trigger_v, target_k, target_v):
        if trigger_k not in self.current_calc_opts:
            raise ValueError(f"Key '{trigger_k}' not found in calc_opts")
        if self.current_calc_opts[trigger_k] == trigger_v:
            self.current_calc_opts[target_k] = target_v

    def parse_natural_language_rule(self, rule):
        # assuming rule is a tuple like ('if brain_region is bla', frequency_band is', 'theta')
        # or ('if brain_region is bla, frequency_bands are',  ['theta_1', 'theta_2'])
        string, target_val = rule
        split_string = '_'.split(string)
        trigger_key = split_string[1]
        trigger_val = split_string[3][:-1]
        target_key = split_string[4]
        return trigger_key, trigger_val, target_key, target_val

    def execute(self):
        if self.current_calc_opts.get('rules'):
            self.apply_rules()
        print(f"executing {self.executing_method} with options {self.current_calc_opts}")
        if self.graph_opts is not None:
            self.executing_method(self.current_calc_opts, self.graph_opts)
        else:
            self.executing_method(self.current_calc_opts)

    def prep(self, prep):
        self.validate_and_load(prep)
        self.executing_method = getattr(self.experiment, self.proc_name)
        self.run_list()
        self.loop_lists = {}

    def run_list(self, opts_list):

        for opts in opts_list:
            self.current_calc_opts = opts
            self.get_loop_lists()
            if self.loop_lists:
                self.iterate_loop_lists(list(self.loop_lists.items()))
            else:
                self.execute()

    def validate_and_load(self, opts):
        self.validate_opts(opts)
        self.proc_name = opts['procedure']
        self.load_analysis_config(opts)

    def run(self, opts, *args, prep=None, **kwargs):
        
        if prep:
            self.prep()
        self.validate_and_load(opts)
        self.run_main(opts)
        if self.follow_up_method is not None:
            self.follow_up_method(*args, **kwargs)
