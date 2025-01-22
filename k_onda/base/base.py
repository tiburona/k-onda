from collections import defaultdict
from copy import deepcopy
import importlib
import json
import pickle
import os
import string

from k_onda.utils import get_round_decimals
    

class Base:
    
    _experiment = None
    _calc_opts = {}
    _cache = defaultdict(dict)
    _filter = {}
    _selected_conditions = {}
    _selected_period_type = ''
    _selected_period_types = []
    _selected_neuron_type = ''
    _selected_brain_region = ''
    _selected_frequency_band = ''
    _selected_region_set = []
    _calc_mode = 'normal'
    original_periods = None
    selectable_variables = [
        'period_type', 
        'period_types',
        'period_conditions', 
        'period_groups',
        'neuron_type', 
        'conditions', 
        'brain_region', 
        'region_set',
        'frequency_band'
        ]
    
    @property
    def experiment(self):
        return Base._experiment

    @experiment.setter
    def experiment(self, value):
        Base._experiment = value

    @property
    def calc_opts(self):
        return Base._calc_opts  
    
    @calc_opts.setter
    def calc_opts(self, value):
        Base._calc_opts = value
        self.set_filter_from_calc_opts()
        Base._cache = defaultdict(dict)

    @property
    def cache(self):
        return Base._cache
    
    def clear_cache(self):
        Base._cache = defaultdict(dict)

    @property
    def filter(self):
        return Base._filter
    
    @filter.setter
    def filter(self, filter):
        Base._filter = filter

    def set_filter_from_calc_opts(self):
        self.filter = defaultdict(lambda: defaultdict(tuple))
        filters = self.calc_opts.get('filter', {})
        if not filters:
            return
        if isinstance(filters, list):
            for filter in filters:
                self.add_to_filters(self.parse_natural_language_filter(filter))
        else:
            for object_type in filters:
                object_filters = self.calc_opts['filter'][object_type]
                for property in object_filters:
                    self.filter[object_type][property] = object_filters[property]   
            if self.calc_opts.get('validate_events'):
                self.filter['event']['is_valid'] = ('==', True)

    def add_to_filters(self, obj_name, attr, operator, target_val):
         self.filter[obj_name][attr] = (operator, target_val)

    def del_from_filters(self, obj_name, attr):
        del self.filter[obj_name][attr]
    
    @staticmethod
    def parse_natural_language_filter(filter):
        # natural language filter should be a tubple like:
        # ex1: ('for animals, identifier must be in', ['animal1', 'animal2'])
        # ex2L ('for units, quality must be !=', '3')] 
        condition, target_val = filter
        split_condition = condition.split(' ')
        obj_name = split_condition[1][:-2]
        attr = split_condition[3]
        be_index = condition.find('be')
        operator = condition[be_index + 3:]
        return obj_name, attr, operator, target_val
    
    @property
    def kind_of_data(self):
        return self.calc_opts.get('kind_of_data')

    @property
    def calc_type(self):
        return self.calc_opts['calc_type']

    @calc_type.setter
    def calc_type(self, calc_type):
        self.calc_opts['calc_type'] = calc_type

    @property
    def calc_mode(self):
        return self._calc_mode
    
    @calc_mode.setter
    def calc_mode(self, calc_mode):
        self._calc_mode = calc_mode

    @property
    def selected_conditions(self):
        return self._selected_conditions
    
    @selected_conditions.setter
    def selected_conditions(self, conditions):
        Base._selected_conditions = conditions
        self.add_to_filters('animal', 'conditions', '==', conditions)

    @property
    def selected_neuron_type(self):
        return Base._selected_neuron_type

    @selected_neuron_type.setter
    def selected_neuron_type(self, neuron_type):
        Base._selected_neuron_type = neuron_type

    @property
    def selected_period_type(self):
        return Base._selected_period_type
        
    @selected_period_type.setter
    def selected_period_type(self, period_type):
        Base._selected_period_type = period_type

    @property
    def selected_period_types(self):
        return Base._selected_period_types
        
    @selected_period_types.setter
    def selected_period_types(self, period_types):
        Base._selected_period_types = period_types

    @property
    def selected_period_conditions(self):
        return Base._selected_period_conditions
        
    @selected_period_types.setter
    def selected_period_conditions(self, period_conditions):
        Base._selected_period_conditions = period_conditions

    @property
    def selected_period_group(self):
        return tuple(self.calc_opts['periods'][self.selected_period_type])
    
    @selected_period_group.setter
    def selected_period_group(self, period_group):
        self.calc_opts['periods'][self.selected_period_type] = period_group
    
    @property
    def selected_frequency_band(self):
        return self.calc_opts['frequency_band']

    @selected_frequency_band.setter
    def selected_frequency_band(self, frequency_band):
        self.calc_opts['frequency_band'] = frequency_band

    @property
    def selected_brain_region(self):
        return self.calc_opts.get('brain_region')
    
    @selected_brain_region.setter
    def selected_brain_region(self, brain_region):
        self.calc_opts['brain_region'] = brain_region

    @property
    def selected_region_set(self):
        return self.calc_opts.get('region_set')

    @selected_region_set.setter
    def selected_region_set(self, region_set):
        self.calc_opts['region_set'] = region_set

    @property
    def freq_range(self):
        if isinstance(self.selected_frequency_band, type('str')):
            return self.experiment.exp_info['frequency_bands'][self.selected_frequency_band]
        else:
            return self.selected_frequency_band
        
    @property
    def finest_res(self):
        return self.calc_opts.get('finest_res', .01)
    
    @property
    def round_to(self):
        return get_round_decimals(self.finest_res)
        
    def get_data_sources(self, data_object_type=None, identifiers=None, identifier=None):
        if data_object_type is None:
            data_object_type = self.calc_opts['base']
            if data_object_type in ['period', 'event']:
                data_object_type = f"{self.kind_of_data}_{data_object_type}"
        data_sources = getattr(self.experiment, f"all_{data_object_type}s")
        if identifier and 'all' in identifier:
            return data_sources
        if identifiers:
            return [source for source in data_sources if source.identifier in identifiers]
        if identifier:
            return [source for source in data_sources if source.identifier == identifier][0]

    @property
    def pre_event(self):
        return self.get_pre_post(0, 'event')
    
    @property
    def post_event(self):
        return self.get_pre_post(1, 'event')

    @property
    def pre_period(self):
        return self.get_pre_post(0, 'period')
    
    @property
    def post_period(self):
        return self.get_pre_post(1, 'period')
    
    def get_pre_post(self, time, obj_type):
        pt = getattr(self, 'period_type', None)
        if pt is None and hasattr(self, 'parent'):
            pt = getattr(self.parent, 'period_type', None)
        if pt is None:
            pt = self.selected_period_type
        return self.calc_opts.get('periods', {}).get(pt, {}).get(
            f'{obj_type}_pre_post', (0, 0))[time]

    
    @property
    def bin_size(self):
        return self.calc_opts.get('bin_size', .01)
    
    def load(self, calc_name, other_identifiers):
        store = self.calc_opts.get('store', 'pkl')
        data_path = self.calc_opts.get('data_path', self.experiment.exp_info.get('data_path'))
        d = os.path.join(data_path, self.kind_of_data)
        store_dir = os.path.join(d, f"{calc_name}_{store}s")
        for p in [d, store_dir]:
            if not os.path.exists(p):
                os.mkdir(p)
        store_path = os.path.join(store_dir, '_'.join(other_identifiers) + f".{store}")
        if os.path.exists(store_path) and not self.calc_opts.get('force_recalc'):
            with open(store_path, 'rb') as f:
                if store == 'pkl':
                    return_val = pickle.load(f)
                else:
                    return_val = json.load(f)
                return True, return_val, store_path
        else:
            return False, None, store_path

    def save(self, result, store_path):
        store = self.calc_opts.get('store', 'pkl')
        mode = 'wb' if store == 'pkl' else 'w'
        with open(store_path, mode) as f:
            if store == 'pkl':
                return pickle.dump(result, f)
            else:
                result_str = json.dumps([arr.tolist() for arr in result])
                f.write(result_str)

    def load_user_module(file_path):
        # Extract a module name from the file path (e.g., "user_plugin")
        module_name = os.path.splitext(os.path.basename(file_path))[0]

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        else:
            raise ImportError(f"Could not load module from {file_path}")
        
    def construct_path(self, constructor_id):
        constructor = deepcopy(self.experiment.exp_info['path_constructors'][constructor_id])
        return self.fill_fields(constructor)
    
    @staticmethod
    def preprocess_constructor(constructor, placeholder="__"):
        # Replace dots in keys
        processed_constructor = {
            key.replace(".", placeholder): value for key, value in constructor.items()
        }
        # Replace dots in the template
        processed_constructor['template'] = processed_constructor['template'].replace(".", placeholder)
        return processed_constructor
    
    def fill_fields(self, constructor):
        
        if not constructor:
            return
        if isinstance(constructor, str):
            return constructor
        
        for field in constructor['fields']:
            if field in self.selectable_variables:
                constructor[field] = getattr(self, field, getattr(self, f'selected_{field}'))
            elif '|' in field:
                field_type, field_key = field.split('|')
                constructor[field] = getattr(self, f'selected_{field_type}')[field_key]
            else:
                constructor[field] = getattr(self, field)
        
        return constructor['template'].format(**constructor)
