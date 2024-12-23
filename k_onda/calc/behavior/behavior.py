import math

import numpy as np

from k_onda.data import Period, Event


# TODO: give behavior data the hierarchy it needs
class BehaviorPrepMethods:

    @property
    def processed_behavior(self):
        return self._processed_behavior
    
    def behavior_prep(self):
        self.prep_data()
        self.prepare_periods()

    def prep_data(self):
        data_importer = self.load_importer()
        data_file = self.calc_opts['behavior_data_file']
        orig_sampling_rate = data_importer.sampling_rate
        data = data_importer.import_data(data_file, self.identifier)
        self._processed_behavior = self.resample(data, orig_sampling_rate)

    def load_importer(self):
        
        user_module = self.load_user_module(self.calc_opts.get('behavior_data_importer'))
        importer_class = getattr(user_module, 'MyBehaviorImporter', None)
        if importer_class is None:
            raise ValueError("User plugin must define MyBehaviorImporter")
        importer = importer_class()
        return importer
    
    def resample(self, data, orig_sampling_rate):
        new_sampling_rate = 1/self.finest_res
        old_to_new_transform = new_sampling_rate/orig_sampling_rate #100/30.303030 = 3.3
        len_new_data = math.ceil(len(data)*old_to_new_transform)
        time_old, values_old = zip(*enumerate(data))
        time_old = [i/orig_sampling_rate for i in time_old] 
        time_new = [i * self.finest_res for i in range(len_new_data)]
        values_new = np.interp(time_new, time_old, values_old)
        return values_new


class BehaviorMethods:

    def get_behavior(self):
        return self.get_average('get_behavior', stop_at='mrl_calculator')   


class BehaviorPeriod(Period):

    def get_behavior(self):
        return self.resolve_calc_fun('behavior')
        
    def _get_behavior(self):
        import_mode = self.calc_opts.get('import_mode', 'by_animal')
        if import_mode == 'by_animal':
            data = self.processed_data[self.calc_opts.get('behavior_fun')]
            return data[self.universal_res_start:self.universal_res_stop]
        elif import_mode == 'by_period':
            return self.processed_data[self.period_type][self.period.identifier]
        else:
            raise ValueError("Unknown import mode")   


class BehaviorEvent(Event):
    
    def get_behavior(self):
        import_mode = self.calc_opts.get('import_mode', 'by_animal')
        if import_mode == 'by_animal':
            data = self.processed_data[self.calc_opts.get('behavior_fun')]
            return data[self.universal_res_start:self.universal_res_stop]
        elif import_mode == 'by_period':
            data = self.processed_data[self.calc_opts.get('behavior_fun')]
            return data[self.universal_res_start:self.universal_res_stop]
        
    




