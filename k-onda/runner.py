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
        self.calc_spec = None
        self.graph_spec = None
        self.proc_name = None
        self.current_calc_spec = None
        self.preparatory_method = None

    def load_analysis_config(self, spec):
        if isinstance(spec, str):
            try:
                with open(spec, 'r', encoding='utf-8') as file:
                    data = file.read()
                    spec = json.loads(data)
            except FileNotFoundError:
                raise Exception(f"File not found: {spec}")
            except json.JSONDecodeError:
                raise Exception(f"Error decoding JSON from the file: {spec}")
        self.spec = spec

    def run_main(self, spec):
        
        if self.proc_name == 'plot':
            self.set_executors(ExecutivePlotter, 'plot')
            calc_spec = spec.get('calc_spec')
            if calc_spec is None:
                self.executing_method(spec)
            elif isinstance(calc_spec, list):
                self.run_list(calc_spec)
            else:
                self.run_list([calc_spec])

        elif self.proc_name == 'csv':
            self.set_executors(Stats, 'make_df')
            self.follow_up_method = 'make_csv'
            spec_list = spec if isinstance(self.spec, list) else [spec]
            self.run_list(spec_list)

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
        for spec_list_key in ['brain_regions', 'frequency_bands', 'levels', 'unit_pairs', 
                             'neuron_qualities', 'inclusion_rules', 'region_sets']:
            spec_list = self.current_calc_spec.get(spec_list_key)
            if spec_list is not None:
                self.loop_lists[spec_list_key] = spec_list

    def iterate_loop_lists(self, remaining_loop_lists, current_index=0):
        if current_index >= len(remaining_loop_lists):
            self.execute()
            return
        spec_list_key, spec_list = remaining_loop_lists[current_index]
        for spec in spec_list:
            key = spec_list_key[:-1] if spec_list_key != 'neuron_qualities' else 'neuron_quality'
            self.current_calc_spec[key] = spec
            self.iterate_loop_lists(remaining_loop_lists, current_index + 1)

    def apply_rules(self):
        rules = self.current_calc_spec['rules']
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
        if trigger_k not in self.current_calc_spec:
            raise ValueError(f"Key '{trigger_k}' not found in calc_spec")
        if self.current_calc_spec[trigger_k] == trigger_v:
            self.current_calc_spec[target_k] = target_v

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
        if self.current_calc_spec.get('rules'):
            self.apply_rules()
        print(f"executing {self.executing_method} with options {self.current_calc_spec}")
        if self.graph_spec is not None:
            self.executing_method(self.current_calc_spec, self.graph_spec)
        else:
            self.executing_method(self.current_calc_spec)

    def prep(self, prep):
        self.validate_and_load(prep)
        self.executing_method = getattr(self.experiment, self.proc_name)
        self.run_list()
        self.loop_lists = {}

    def run_list(self, spec_list):

        for spec in spec_list:
            self.current_calc_spec = spec
            self.get_loop_lists()
            if self.loop_lists:
                self.iterate_loop_lists(list(self.loop_lists.items()))
            else:
                self.execute()

    def validate_and_load(self, spec):
        self.validate_opts(spec)
        self.proc_name = spec['procedure']
        self.load_analysis_config(spec)

    def run(self, spec, *args, prep=None, **kwargs):
        
        if prep:
            self.prep()
        self.validate_and_load(spec)
        self.run_main(spec)
        if self.follow_up_method is not None:
            self.follow_up_method(*args, **kwargs)
