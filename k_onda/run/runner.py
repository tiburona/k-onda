import json
import os
from copy import deepcopy

from ..spreadsheet import Stats
from ..plotting import ExecutivePlotter, Layout
from .opts_validator import OptsValidator
from .initialize_experiment import Initializer


class Runner(OptsValidator):

    def __init__(self, config_file=None):
        self.config = config_file if config_file else os.getenv('INIT_CONFIG')
        self.initializer = Initializer(self.config)
        self.experiment = self.initializer.init_experiment()
        self.prep = None
        self.opts = None
        self.follow_up = None
        self.executing_class = None
        self.executing_instance = None
        self.executing_instances = {}
        self.executing_method = None
        
    def setup(self, opts, prep=None, follow_up=None):
       
        self.prep = self.load(prep)
        self.opts = self.load(opts)
        if follow_up:
            self.follow_up = self.load(follow_up)
        elif self.opts['proc_name'] == 'make_csv':
            self.follow_up = {'proc_name': 'write_csv', 'calc_opts': {}}
        else:
            self.follow_up = None

    def load(self, opts):
        if isinstance(opts, str):
            try:
                with open(opts, 'r', encoding='utf-8') as file:
                    data = file.read()
                    opts = json.loads(data)
            except FileNotFoundError:
                raise Exception(f"File not found: {opts}")
            except json.JSONDecodeError:
                raise Exception(f"Error decoding JSON from the file: {opts}")
        return opts

    def execute(self, opts):

        executors = {
            'make_plots': (ExecutivePlotter, 'plot'),
            'make_csv': (Stats, 'make_df'),
            'validate_lfp_events': (self.experiment, 'validate_lfp_events'),
            'write_csv': (self.experiment, 'write_csv')
        }
        
        self.set_executors(*executors[opts['proc_name']])

        calc_opts = opts.get('calc_opts', [])
        if isinstance(calc_opts, dict):
            calc_opts = [calc_opts]

        expanded_calc_opts = [
            d 
            for single_calc_opts in calc_opts
            for d in CalcOptsProcessor(single_calc_opts).process()
        ]

        if opts['proc_name'] == 'make_csv':
            self.executing_method(expanded_calc_opts)

        else:
            for each_opts in expanded_calc_opts:
                self.executing_method(each_opts)
            
    def set_executors(self, executor, method):
        if isinstance(executor, type):
            self.executing_class = executor
            if self.executing_class.__name__ in self.executing_instances:
                self.executing_instance = self.executing_instances[self.executing_class.__name__]
            else:
                self.executing_instance = self.executing_class(self.experiment)
        else:
            self.executing_instance = executor
        self.executing_method = getattr(self.executing_instance, method)

    def run(self, opts, prep=None, follow_up=None):

        self.setup(opts, prep, follow_up)
        for opts in [self.prep, self.opts, self.follow_up]:
            if opts:
                self.execute(opts)
        

class CalcOptsProcessor(OptsValidator):

    def __init__(self, calc_opts):
        self.calc_opts_ = calc_opts
        self.current_calc_opts_ = calc_opts

    def process(self):
        loop_lists = self.get_loop_lists()
        opts = self.loop_lists() if loop_lists else [self.apply_rules(self.calc_opts_)]
        return opts
     
    def get_loop_lists(self):
        loop_lists = {
            key: opt_list 
            for key in ['brain_regions', 'frequency_bands', 'levels', 'unit_pairs', 
                        'neuron_qualities', 'inclusion_rules', 'region_sets'] 
                        for opt_list in self.calc_opts_.get(key) 
                        if opt_list is not None}
        return self.iterate_loop_lists(loop_lists)
            
    def iterate_loop_lists(self, remaining_loop_lists, current_index=0, accumulated_results=None):
        # Initialize the results list if not provided
        if accumulated_results is None:
            accumulated_results = []

        # Base case: accumulate the current state as a result
        if current_index >= len(remaining_loop_lists):
            # Optionally apply rules if needed
            if self.current_calc_opts_.get('rules'):
                self.apply_rules()
            
            accumulated_results.append(deepcopy(self.current_calc_opts_))
            return accumulated_results

        # Recursive case: iterate over the current loop
        opt_list_key, opt_list = remaining_loop_lists[current_index]
        for opt in opt_list:
            key = opt_list_key[:-1] if opt_list_key != 'neuron_qualities' else 'neuron_quality'
            self.current_calc_opts_[key] = opt
            # Recursively iterate
            self.iterate_loop_lists(remaining_loop_lists, current_index + 1, accumulated_results)

        return accumulated_results

    def apply_rules(self, calc_opts):
        rules = calc_opts['rules']
        if not isinstance(rules, dict):
            for rule in rules:
                self.assign_per_rule(*self.parse_natural_language_rule(rule))
        else:          
            # Assuming rules is a dictionary like: {'calc_type': {'mrl': [('time_type', 'block')]}}
            for trigger_k, conditions in rules.items():
                for trigger_v, target_vals in conditions.items():
                    for target_k, target_v in target_vals:
                        self.assign_per_rule(calc_opts, trigger_k, trigger_v, target_k, target_v)
    
    def assign_per_rule(self, calc_opts, trigger_k, trigger_v, target_k, target_v):
        if trigger_k not in calc_opts:
            raise ValueError(f"Key '{trigger_k}' not found in calc_opts")
        if calc_opts[trigger_k] == trigger_v:
            calc_opts[target_k] = target_v

    def parse_natural_language_rule(self, rule):
        # assuming rule is a tuple like ('if brain_region is bla', frequency_band is', 'theta')
        # or ('if brain_region is bla, frequency_bands are',  ['theta_1', 'theta_2'])
        string, target_val = rule
        split_string = '_'.split(string)
        trigger_key = split_string[1]
        trigger_val = split_string[3][:-1]
        target_key = split_string[4]
        return trigger_key, trigger_val, target_key, target_val
