import json
import os

from plotters import PeriStimulusPlotter, GroupStatsPlotter
from stats import Stats
from initialize_experiment import Initializer

PROCEDURE_DICT = {
    f"plot_{data_type}": {'class': PeriStimulusPlotter, 'data_category': 'spike', 'method': 'plot'}
    for data_type in [
        'psth', 'proportion_score', 'autocorrelation', 'spectrum', 'cross_correlation,' 'autocorrelogram'
    ]}

other_procedures = {
    'plot_group_stats': {'class': GroupStatsPlotter, 'data_category': 'spike', 'method': 'plot_group_stats'},
    'make_spreadsheet': {'class': Stats, 'method': 'make_df', 'follow_up': 'make_spreadsheet'}
}

# TODO: one idea is to make sure the Runner class keeps track of its executing classes with their name, and always checks
# whether one exists before creating a new one.  That way all the created dfs will exist on the same stats instance


class Runner:

    def __init__(self, config_file=None, lfp=False, behavior=False):
        self.config = config_file if config_file else os.getenv('INIT_CONFIG')
        self.lfp = lfp
        self.behavior = behavior
        self.initializer = Initializer(self.config)
        self.experiment = self.initializer.init_experiment()
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

    def read_inputs(self, proc_name, opts):
        self.proc_name = proc_name
        self.load_analysis_config(opts)
        if isinstance(self.data_opts, list):
            for opts in self.data_opts:
                self.prepare(opts)
        else:
            self.prepare(self.data_opts)

    def prepare(self, opts):
        self.current_data_opts = opts
        self.executing_class = PROCEDURE_DICT[self.proc_name]['class']
        kwargs = {dc: getattr(self.initializer, f"init_{dc}_experiment")() for dc in ['lfp', 'behavior']
                  if getattr(self, dc)}
        if self.executing_class.__name__ in self.executing_instances:
            self.executing_instance = self.executing_instances[self.executing_class.__name__]
        else:
            self.executing_instance = self.executing_class(self.experiment, **kwargs)
        self.executing_method = getattr(self.executing_instance, PROCEDURE_DICT[self.proc_name]['method'])
        self.follow_up_method = PROCEDURE_DICT[self.proc_name].get('follow_up')
        self.get_loop_lists()

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

    def get_loop_lists(self):
        for opt_list_key in ['brain_regions', 'frequency_bands', 'levels', 'ac_keys']:
            opt_list = self.current_data_opts.get(opt_list_key)
            if opt_list is not None:
                self.loop_lists[opt_list_key] = opt_list

    def iterate_loop_lists(self, remaining_loop_lists, current_index=0):
        if current_index >= len(remaining_loop_lists):
            self.execute()
            return

        opt_list_key, opt_list = remaining_loop_lists[current_index]
        for opt in opt_list:
            self.data_opts[opt_list_key[:-1]] = opt
            self.iterate_loop_lists(remaining_loop_lists, current_index + 1)

    def run(self, proc_name, opts, *args, **kwargs):
        self.read_inputs(proc_name, opts)
        if self.loop_lists:
            self.iterate_loop_lists(list(self.loop_lists.items()))
        else:
            self.execute()
        if self.follow_up_method is not None:
            getattr(self.executing_instance, self.follow_up_method)(*args, **kwargs)

    def apply_rules(self):
        if 'rules' not in self.data_opts:
            raise ValueError("Expected 'rules' in data_opts")

        rules = self.current_data_opts['rules']
        # Assuming rules is a dictionary like: {'data_type': {'mrl': [('time_type', 'block')]}}
        for data_key, conditions in rules.items():
            if data_key not in self.data_opts:
                raise ValueError(f"Key '{data_key}' not found in data_opts")

            data_value = self.current_data_opts[data_key]
            for condition in conditions:
                if not (isinstance(condition, tuple) and len(condition) == 2):
                    raise ValueError(f"Condition '{condition}' is not a valid tuple of length 2")

                condition_key, condition_value = condition
                if data_value == condition_key:
                    self.current_data_opts[condition_key] = condition_value

    def execute(self):
        if self.current_data_opts.get('rules'):
            self.apply_rules()
        if self.graph_opts is not None:
            self.executing_method(self.current_data_opts, self.graph_opts)
        else:
            self.executing_method(self.current_data_opts)




