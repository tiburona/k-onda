import json
import os

from plotters import PeriStimulusPlotter
from initialize_experiment import Initializer

PROCEDURE_DICT = {
    f"plot_{data_type}": {'class': PeriStimulusPlotter, 'data_category': 'spike', 'method': 'plot'}
    for data_type in [
        'psth', 'proportion_score', 'autocorrelation', 'spectrum', 'cross_correlation,' 'autocorrelogram',
    '']}


class Runner:

    def __init__(self, proc_name, opts, config_file=None):
        self.config = config_file if config_file else os.getenv('INIT_CONFIG')
        self.initializer = Initializer(self.config)
        self.experiment = self.initializer.init_experiment()
        self.proc_name = proc_name
        # If opts is a string, assume it's a file path and try to read from it
        if isinstance(opts, str):
            try:
                with open(opts, 'r', encoding='utf-8') as file:
                    data = file.read()
                    opts = json.loads(data)
            except FileNotFoundError:
                raise Exception(f"File not found: {opts}")
            except json.JSONDecodeError:
                raise Exception(f"Error decoding JSON from the file: {opts}")

        # Safely get data_opts and graph_opts, assuming opts is a dictionary
        self.data_opts = opts.get('data_opts', {})
        self.graph_opts = opts.get('graph_opts', None)
        self.executing_class = None
        self.executing_instance = None
        self.executing_method = None
        self.loop_lists = {}
        self.follow_up_method = None

    def prepare(self):
        self.executing_class = PROCEDURE_DICT[self.proc_name]['class']
        kwargs = {dc: getattr(self.initializer, f"init_{dc}_experiment")() for dc in ['lfp', 'behavior']
                  if dc in self.data_opts.get('data_classes', [])}
        self.executing_instance = self.executing_class(self.experiment, **kwargs)
        self.executing_method = getattr(self.executing_instance, PROCEDURE_DICT[self.proc_name]['method'])
        self.follow_up_method = PROCEDURE_DICT[self.proc_name].get('follow_up')
        for opt_list_key in ['brain_regions', 'frequency_bands', 'levels', 'ac_keys']:
            opt_list = self.data_opts.get(opt_list_key)
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

    def run(self):
        self.prepare()
        if self.loop_lists:
            self.iterate_loop_lists(list(self.loop_lists.items()))
        else:
            self.execute()
        if self.follow_up_method is not None:
            getattr(self.executing_instance, self.follow_up_method)()

    def execute(self):
        if self.graph_opts is not None:
            self.executing_method(self.data_opts, self.graph_opts)
        else:
            self.executing_method(self.data_opts)




