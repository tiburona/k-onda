import numpy as np
from collections import defaultdict
import pickle
import json
import os


class Base:

    _data_opts = {}

    @property
    def data_opts(self):
        return Base._data_opts  
    
    @data_opts.setter
    def data_opts(self, value):
        Base._data_opts = value
        self.set_filter_from_data_opts()

    def set_filter_from_data_opts(self):
        Base.filter = defaultdict(lambda: defaultdict(tuple))
        for object_type in self.data_opts.get('filter', {}):
            filters = self.data_opts['filter'][object_type]
            for property in filters:
                self.filter[object_type][property] = filters[property]   
        if self.data_opts.get('validate_events'):
            self.filter['event']['is_valid'] = ('==', True)

    @property
    def data_type(self):
        return self.data_opts['data_type']

    @data_type.setter
    def data_type(self, data_type):
        self.data_opts['data_type'] = data_type

    @property
    def data_class(self):
        return self.data_opts.get('data_class')

    @property
    def selected_neuron_type(self):
        if not self.filter['unit']['neuron_type']:
            return None
        return self.filter['unit']['neuron_type'][1]

    @selected_neuron_type.setter
    def selected_neuron_type(self, neuron_type):
        self.filter['unit']['neuron_type'] = ('==', neuron_type)

    @property
    def selected_period_type(self):
        if not self.filter['period']['period_type']:
            return None
        return self.filter['period']['period_type'][1]

    @selected_period_type.setter
    def selected_period_type(self, period_type):
        self.filter['period']['period_type'] = ('==', period_type)

    @property
    def current_frequency_band(self):
        return self.data_opts['frequency_band']

    @current_frequency_band.setter
    def current_frequency_band(self, frequency_band):
        self.data_opts['frequency_band'] = frequency_band

    @property
    def current_brain_region(self):
        return self.data_opts.get('brain_region')

    @current_brain_region.setter
    def current_brain_region(self, brain_region):
        self.data_opts['brain_region'] = brain_region

    @property
    def current_region_set(self):
        return self.data_opts.get('region_set')

    @current_region_set.setter
    def current_region_set(self, region_set):
        self.data_opts['region_set'] = region_set

    @property
    def freq_range(self):
        if isinstance(self.current_frequency_band, type('str')):
            return self.experiment.info['frequency_bands'][self.current_frequency_band]
        else:
            return self.current_frequency_band


class Data(Base):

    summarizers = {'psth': np.mean, 'firing_rates': np.mean,
                   'proportion': np.mean, 'spike_count': np.sum}

    def __init__(self):
        self.parent = None
        self.cache = {}

    def get_data(self, data_type=None):
        if data_type is None:
            data_type = self.data_type
        data = getattr(self, f"get_{data_type}")()
        if (self.data_opts.get('evoked') 
            and hasattr(self, 'reference')
            and self.reference is not None):
            data -= self.reference.get_data(data_type)
        return data

    @property
    def data(self):
        if self.data_type in self.cache:
            return self.cache[self.data_type]
        data = self.get_data()
        if self.data_opts.get('evoked'): 
            data -= self.get_reference_data()
        self.cache[self.data_type] = data
        return data

    def include(self, check_ancestors=False):
        return self.select(self.filter, check_ancestors=check_ancestors)
    
    def select(self, filters, check_ancestors=False):
        operations = {
            '==': lambda a, b: a == b,
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b,
            'in': lambda a, b: a in b,
            '!=': lambda a, b: a != b,
            'not in': lambda a, b: a not in b
        }

        if not check_ancestors and self.name not in filters:
            return True
              
        for obj in (self.ancestors if check_ancestors else [self]):
            if obj.name not in filters:
                continue
            obj_filters = filters[obj.name]
            for attr in obj_filters:
                if hasattr(obj, attr):
                    object_value = getattr(obj, attr)
                    operation_symbol, target_value = obj_filters[attr]
                    function = operations[operation_symbol]
                    if not function(object_value, target_value):
                        return False
        return True
            
    def get_average(self, base_method, stop_at='event', axis=0, **kwargs):
        """
        Recursively calculates the average of the values of the computation in the base method on the object's
        descendants.

        Parameters:
        - base_method (str): Name of the method to be called when the recursion reaches the base case.
        - stop_at (str): The 'name' attribute of the base case object upon which `base_method` should be called.
        - axis (int or None): Specifies the axis across which to compute the mean.
        - **kwargs: Additional keyword arguments to be passed to the base method.

        Returns:
        float or np.array: The mean of the data values from the object's descendants.
        """
        if not self.include():
            if self.name == 'event':
                print('yay')
            return float('nan')
        
        if self.name == stop_at:  # we are at the base case and will call the base method
            if hasattr(self, base_method) and callable(getattr(self, base_method)):
                return getattr(self, base_method)(**kwargs)
            else:
                raise ValueError(f"Invalid base method: {base_method}")

        else: 
            if not len(self.children):
                return float('nan')
            
            child_vals = [child.get_average(base_method, axis=axis, stop_at=stop_at, **kwargs) 
                          for child in self.children]
        
            child_vals = [val for val in child_vals if not self.is_nan(val)]
                
            if len(child_vals) and isinstance(child_vals[0], dict):
                # Initialize defaultdict to automatically start lists for new keys
                result_dict = defaultdict(list)
    
                # Aggregate values from each dictionary under their corresponding keys
                for child_val in child_vals:
                    for key, value in child_val.items():
                        result_dict[key].append(value)

                # Calculate average of the list of values for each key
                return_dict = {key: self.take_average(values, axis) for key, values in result_dict.items()}
                return return_dict
                    
            else:
                return self.take_average(child_vals, axis)
            
    @staticmethod        
    def take_average(vals, axis):
        if axis is None:  # compute mean over all dimensions
            return np.nanmean(np.array(vals))
        else:  # compute mean over provided dimension
            return np.nanmean(np.array(vals), axis=axis)
    
    @staticmethod
    def extend_into_bins(sources, extend_by):
        if 'frequency' in extend_by:
            sources = [freq_bin for src in sources for freq_bin in src.frequency_bins]
        if 'time' in extend_by:
            sources = [time_bin for src in sources for time_bin in src.time_bins]
        return sources
            
    def get_median(self, stop_at=None, extend_by=None):
        if not stop_at:
            stop_at = self.data_opts.get('stop_at')
        vals_to_summarize = self.get_descendants(stop_at=stop_at)
        vals_to_summarize = self.extend_into_bins(vals_to_summarize, extend_by)
        return np.median([obj.data for obj in vals_to_summarize])

    def is_nan(self, value):
        if isinstance(value, float) and np.isnan(value):
            return True
        elif isinstance(value, np.ndarray) and np.all(np.isnan(value)):
            return True
        elif isinstance(value, dict) and all([self.is_nan(val) for val in value.values()]):
            return True                                
        else:
            return False

    @property
    def hierarchy(self):
        if self.name == 'experiment':
            return [self.name]
        if hasattr(self, 'parent'):
            return self.parent.hierarchy + [self.name]
 
    @property
    def sampling_rate(self):
        if self.name == 'experiment':
            return self._sampling_rate
        else:
            return self.experiment.sampling_rate
        
    @property
    def lfp_sampling_rate(self):
        if self.name == 'experiment':
            return self._lfp_sampling_rate
        else:
            return self.experiment.lfp_sampling_rate

    @property
    def ancestors(self):
        return self.get_ancestors()  

    @property
    def descendants(self):
        return self.get_descendants()    
    
    def get_ancestors(self):
       
        if not hasattr(self, 'parent'):
            return []
        if not hasattr(self.parent, 'ancestors'):
            return [self]
        return [self] + self.parent.get_ancestors()
    
    def get_descendants(self, stop_at=None, descendants=None):
   
        if descendants is None:
            descendants = []
        if self.name == stop_at or not hasattr(self, 'children'):
            descendants.append(self)
        else:
            if hasattr(self, 'children') and self.children:
                for child in self.children:
                    child.get_descendants(descendants=descendants)
        return descendants
    
    def get_reference_data(self, data_type=None):
        if data_type is None:
            data_type = self.data_type
        if hasattr(self, 'reference'):
            return getattr(self, 'reference').get_data(data_type)
        else:
            period_type = self.selected_period_type
            reference_period_type = self.experiment.info['reference'][period_type]
            self.period_type = reference_period_type
            reference_data = self.get_data(data_type)
            self.period_type = period_type
            return reference_data
        
    def load(self, calc_name, other_identifiers):
        store = self.data_opts.get('store', 'pkl')
        d = os.path.join(self.data_opts['data_path'], self.data_class)
        store_dir = os.path.join(d, f"{calc_name}_{store}s")
        for p in [d, store_dir]:
            if not os.path.exists(p):
                os.mkdir(p)
        store_path = os.path.join(store_dir, '_'.join(other_identifiers) + f".{store}")
        if os.path.exists(store_path) and not self.data_opts.get('force_recalc'):
            with open(store_path, 'rb') as f:
                if store == 'pkl':
                    return_val = pickle.load(f)
                else:
                    return_val = json.load(f)
                return True, return_val, store_path
        else:
            return False, None, store_path

    def save(self, result, store_path):
        store = self.data_opts.get('store', 'pkl')
        mode = 'wb' if store == 'pkl' else 'w'
        with open(store_path, mode) as f:
            if store == 'pkl':
                return pickle.dump(result, f)
            else:
                result_str = json.dumps([arr.tolist() for arr in result])
                f.write(result_str)

    # def get_data(self, data_type=None):
        
    #     if data_type is None:
    #         data_type = self.data_type

    #     is_calculated = self.experiment.calculated_data.get(data_type, False)

    #     if not is_calculated:
    #         level = self.base_levels[self.data_type]
    #         for ent in getattr(self.experiment, f"all_{level}s"):
    #             getattr(ent, f"get_{self.data_type}")()
        
    #     if self.name == level:
    #         return self.data[self.data_type]
    #     else:
    #         return self.summarize()

   


        
    # def summarize(self, data, axis=0):

    #     summary_func = np.mean
        
    #     for vals in data:
    #         if isinstance(vals[0], dict):
    #             vals = [v for val in vals for k, v in val.items() if not self.filter_out(k)]
    #             return summary_func(self.summarize(vals), axis=axis)
    #         else:
    #             return [val for val in data.values()]

    
    # def summarize_from_data_frame(self, summarizer):
    #     # TODO I need to figure this out
    #     summarizer = self.summarizers[self.data_type]
    #     data = self.experiment.data_frames[self.data_type]
    #     data = self.filter_data_frame(data)
    #     for ancestor in self.ancestors:
    #         data = data[data[ancestor.name] == ancestor.identifier]

    #     if self.data_opts.get('time_type') == 'continuous':
    #         continuous_time = True
    #     else:
    #         continuous_time = False
        
    #     if self.data_opts.get('frequency_type') == 'continuous':
    #         continuous_freq = True
    #     else:
    #         continuous_freq = False
        
    #     for level in reversed(self.hierarchy):
    #         # group by all levels below this one, successively
    #         if level == self.name:
    #             break
    #         group_levels = self.hierarchy[0:self.hierarchy.find(level)]
    #         if self.data_opts.get('time_type') == 'continuous':
    #             group_levels.append('time')
    #         if self.data_opts.get('frequency_type') == 'continuous':
    #             group_levels.append('frequency')
    #         grouped_data = data.groupby(group_levels)
    #         data = grouped_data.apply(summarizer).reset_index()

    #     # TODO: going to have to add brain region and frequency to values string
    #     if continuous_time and continuous_freq:
    #         value = data.pivot(index='frequency', columns='time', values=self.data_type).values
    #     elif continuous_time or continuous_freq:
    #         value = data[self.data_type].values
    #     else:
    #         value = data[self.data_type].iloc[0]

    #     return value
    
    # def filter_data_frame(self, data):
    #     # common filters: neuron_type, period_type, frequency, neuron quality
    #     # expected form of filter
    #     # {'period_type': 'tone'}

    #     for key, val in self.current_filters.items():
    #         data = data[data[key] == val]  
    #         return data


