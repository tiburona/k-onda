import numpy as np
from collections import defaultdict
from collections.abc import Iterable

def is_iterable(obj):
    return isinstance(obj, Iterable)

class Base:

    _data_opts = {}
    filters = defaultdict(lambda: defaultdict(tuple))
    cache_objects = []

    @property
    def data_opts(self):
        return Base._data_opts  
    
    @data_opts.setter
    def data_opts(self, value):
        Base._data_opts = value
        self.set_filters_from_data_opts()

    def set_filters_from_data_opts(self):
        for object_type in self.data_opts.get('filters', {}):
            new_object_type_filters = self.data_opts['filters'][object_type]
            existing_object_type_filters = self.filters[object_type]
            merged_dict = {k: v for d in [existing_object_type_filters, new_object_type_filters] 
                           for k, v in d.items()}
            self.filters[object_type] = merged_dict

            


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
        if not self.filters['unit']['neuron_type']:
            return None
        return self.filters['unit']['neuron_type'][1]

    @selected_neuron_type.setter
    def selected_neuron_type(self, neuron_type):
        self.filters['unit']['neuron_type'] = ('==', neuron_type)

    @property
    def selected_period_type(self):
        if not self.filters['period']['period_type']:
            return None
        return self.filters['period']['period_type'][1]

    @selected_period_type.setter
    def selected_period_type(self, period_type):
        self.filters['period']['period_type'] = ('==', period_type)
      

    @property
    def selected_frequency_band(self):
        return self.current_filters['frequency_band']

    @selected_frequency_band.setter
    def selected_frequency_band(self, frequency_band):
        self.current_filters['frequency_band'] = frequency_band

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


class Data(Base):

    summarizers = {'psth': np.mean, 'firing_rates': np.mean,
                   'proportion': np.mean, 'spike_count': np.sum}

    def __init__(self):
        self.parent = None
        self.children = []

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
        data = self.get_data()
        if self.data_opts.get('evoked'): 
            data -= self.get_reference_data()
        return data

    def include(self):

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

        filters = self.filters.get(self.name)
        if not filters:
            return True

        for attr in filters:
            if hasattr(self, attr):
                object_value = getattr(self, attr)
                operation_symbol, target_value = filters[attr]
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
        if self.name == stop_at:  # we are at the base case and will call the base method
            if hasattr(self, base_method) and callable(getattr(self, base_method)):
                return getattr(self, base_method)(**kwargs)
            else:
                raise ValueError(f"Invalid base method: {base_method}")

        else: 
            if not self.include():
                return float('nan')
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
            return self.experiment._sampling_rate
    
    @property
    def ancestors(self):
        return self.get_ancestors()      
    
    def get_ancestors(self):
       
        if not hasattr(self, 'parent'):
            return []
    
        if not hasattr(self.parent, 'ancestors'):
            return [self]

        # Include the current object and then get the ancestors of its parent
        return [self] + self.parent.ancestors

    def get_time_bins(self, data):
        if len(list(enumerate(data))) == 71:
            a = 'foo'
        time_bins = [TimeBin(i, data_point, self) for i, data_point in enumerate(data)]
        return time_bins

    def get_frequency_bins(self, data):
        pass        
    
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

   

    @staticmethod        
    def take_average(vals, axis):
        if axis is None:  # compute mean over all dimensions
            return np.nanmean(np.array(vals))
        else:  # compute mean over provided dimension
            return np.nanmean(np.array(vals), axis=axis)

        
    def summarize(self, data, axis=0):

        summary_func = np.mean
        
        for vals in data:
            if isinstance(vals[0], dict):
                vals = [v for val in vals for k, v in val.items() if not self.filter_out(k)]
                return summary_func(self.summarize(vals), axis=axis)
            else:
                return [val for val in data.values()]

    
    def summarize_from_data_frame(self, summarizer):
        # TODO I need to figure this out
        summarizer = self.summarizers[self.data_type]
        data = self.experiment.data_frames[self.data_type]
        data = self.filter_data_frame(data)
        for ancestor in self.ancestors:
            data = data[data[ancestor.name] == ancestor.identifier]

        if self.data_opts.get('time_type') == 'continuous':
            continuous_time = True
        else:
            continuous_time = False
        
        if self.data_opts.get('frequency_type') == 'continuous':
            continuous_freq = True
        else:
            continuous_freq = False
        
        for level in reversed(self.hierarchy):
            # group by all levels below this one, successively
            if level == self.name:
                break
            group_levels = self.hierarchy[0:self.hierarchy.find(level)]
            if self.data_opts.get('time_type') == 'continuous':
                group_levels.append('time')
            if self.data_opts.get('frequency_type') == 'continuous':
                group_levels.append('frequency')
            grouped_data = data.groupby(group_levels)
            data = grouped_data.apply(summarizer).reset_index()

        # TODO: going to have to add brain region and frequency to values string
        if continuous_time and continuous_freq:
            value = data.pivot(index='frequency', columns='time', values=self.data_type).values
        elif continuous_time or continuous_freq:
            value = data[self.data_type].values
        else:
            value = data[self.data_type].iloc[0]

        return value
    
    def filter_data_frame(self, data):
        # common filters: neuron_type, period_type, frequency, neuron quality
        # expected form of filter
        # {'period_type': 'tone'}

        for key, val in self.current_filters.items():
            data = data[data[key] == val]  
            return data


class SpikeMethods:
 
    def get_psth(self):
        return self.get_average('get_psth', stop_at=self.data_opts.get('base', 'event'))
    
    def get_firing_rates(self):
        if self.name == 'period' and self.is_relative:
            a = 'foo'
        return self.get_average('get_firing_rates', stop_at=self.data_opts.get('base', 'event'))
        

class TimeBin(Data):
    name = 'time_bin'

    def __init__(self, i, val, parent):
        self.parent = parent
        self.identifier = i
        self.val = val
        self.period = None
        if self.period:
            self.period_type = self.period.period_type

    @property
    def data(self):
        return self.val

    @property
    def time(self):
        if self.data_type == 'correlation':
            ts = np.arange(-self.data_opts['lags'], self.data_opts['lags'] + 1) / self.sampling_rate
        else:
            pre_stim, post_stim = [self.data_opts['events'][self.parent.period_type][val] 
                                for val in ['pre_stim', 'post_stim']]
            ts = np.arange(-pre_stim, post_stim, self.data_opts['bin_size'])
        
        # Round the timestamps to the nearest 10th of a microsecond
        ts = np.round(ts, decimals=7)

        return ts[self.identifier]