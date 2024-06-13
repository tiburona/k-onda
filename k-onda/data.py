from context import experiment_context
from utils import get_ancestors, get_descendants
import numpy as np
from math_functions import sem
from utils import cache_method, find_ancestor_attribute
from collections import defaultdict


class Base:
    
    _global_neuron_types = None  

    context = experiment_context

    @classmethod
    def set_global_neuron_types(cls, types):
        cls._global_neuron_types = types

    @property
    def neuron_types(self):
        return self.__class__._global_neuron_types

    @property
    def data_opts(self):
        return self.context.vals.get('data')

    @data_opts.setter
    def data_opts(self, opts):
        self.context.set_val('data', opts)

    @property
    def data_type(self):
        return self.data_opts['data_type']

    @data_type.setter
    def data_type(self, data_type):
        self.update_data_opts([(['data_type'], data_type)])

    @property
    def data_class(self):
        return self.data_opts.get('data_class')

    @property
    def selected_neuron_type(self):
        return self.context.vals.get('neuron_type')

    @selected_neuron_type.setter
    def selected_neuron_type(self, neuron_type):
        self.context.set_val('neuron_type', neuron_type)

    @property
    def selected_period_type(self):
        return self.context.vals.get('period_type')

    @selected_period_type.setter
    def selected_period_type(self, period_type):
        self.context.set_val('period_type', period_type)

    @property
    def current_frequency_band(self):
        return self.data_opts.get('frequency_band')

    @current_frequency_band.setter
    def current_frequency_band(self, frequency_band):

        self.update_data_opts([(['frequency_band'], frequency_band)])

    @property
    def current_brain_region(self):
        return self.data_opts.get('brain_region')

    @current_brain_region.setter
    def current_brain_region(self, brain_region):
        self.update_data_opts([(['brain_region'], brain_region)])

    def update_data_opts(self, reassignments):
        for path, value in reassignments:
            current_level = self.data_opts
            for key in path[:-1]:
                if key not in current_level or not isinstance(current_level[key], dict):
                    current_level[key] = {}
                current_level = current_level[key]
            current_level[path[-1]] = value

        self.data_opts = self.data_opts  # Reassign to trigger the setter


class Data(Base):

    _global_sampling_rate = None  # Class variable to hold the sampling rate

    @classmethod
    def set_global_sampling_rate(cls, rate):
        cls._global_sampling_rate = rate

    @property
    def sampling_rate(self):
        if hasattr(self, '_sampling_rate'):
            return self._sampling_rate
        else:
            return self.__class__._global_sampling_rate

    def __iter__(self):
        for child in self.children:
            yield child

    @property
    def children(self):
        if self._children is None:
            return self._children
        return [child for child in self._children if child.is_valid]

    @property
    def data(self):
        """
        Returns:
        float or np.array: The mean of the data values from the object's descendants.
        """
        return getattr(self, f"get_{self.data_type}")()

    
    @property
    def is_valid(self):  
        for ancestor in self.ancestors:
            if ancestor is self:
                pass
            else:
                if not ancestor.is_valid:
                    return False
        inclusion_criteria = self.get_inclusion_criteria()[self.name]
        if hasattr(self, 'validator'):
            inclusion_criteria += [lambda x: x.validator()]
        return all([criterion(self) for criterion in inclusion_criteria])

    @property
    def mean_data(self):
        return np.mean(self.data)
    
    @property
    def sum_data(self):
        return np.sum(self.data)

    @property
    def sd_data(self):
        return np.std(self.data)

    @property
    def ancestors(self):
        return get_ancestors(self)

    @property
    def descendants(self):
        return get_descendants(self)

    @property
    def sampling_rate(self):
        if hasattr(self, '_sampling_rate'):
            return self._sampling_rate
        elif self.parent is not None:
            return self.parent.sampling_rate
        else:
            return None

    @property
    def is_relative(self):
        if hasattr(self, '_is_relative'):
            return self._is_relative
        elif self.parent is not None:
            return self.parent.is_relative
        else:
            return None

    @property
    def reference(self):
        if hasattr(self, 'period') and not self.period.reference_period_type:
            return None
        elif hasattr(self, 'reference_period_type') and not self.reference_period_type:
            return None
        else:
            if self.name == 'period':
                return [prd for prd in self.parent.periods[self.reference_period_type] 
                        if self is prd.target_period][0]
            if self.name == 'mrl_calculator':
                return [calc for calc in self.parent.mrl_calculators[self.reference_period_type]
                        if self is calc.period.target and self.unit is calc.unit][0]
            if self.name == 'event':
                return self.parent.reference
        return None

    @property
    def sem(self):
        return self.get_sem(collapse_sem_data=True)

    @property
    def sem_envelope(self):
        return self.get_sem(collapse_sem_data=False)

    @property
    def scatter(self):
        return self.get_scatter_points()

    @property
    def num_bins_per_event(self):
        bin_size = self.data_opts.get('bin_size')
        pre_stim, post_stim = (self.data_opts['events'][self.period_type].get(opt) 
                               for opt in ['pre_stim', 'post_stim'])
        return int((pre_stim + post_stim) / bin_size)

    @property
    def current_reference_period_type(self):
        exp = self.find_experiment()
        return [period for period in exp.all_periods 
                if period.period_type == self.selected_period_type][0].reference_period_type
    
    def get_child_by_identifier(self, identifier):
        return [child for child in self.children if child.identifier == identifier][0]
    
    @cache_method
    def get_inclusion_criteria(self):
        criteria = defaultdict(list)

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

        def make_criteria_func(name, attribute, value, operation, modifier=None, mattr=None, 
                               mop=None, mval=None):
            def criteria_func(x):
                
                # Ensure ancestors are not excluded based on descendants
                if name != x.name and (name not in x.hierarchy or 
                                       x.hierarchy[x.name] <= x.hierarchy[name]):
                    return True
                # Apply modifier if present
                elif modifier and not mop(find_ancestor_attribute(x, name, mattr), mval):
                    return True
                else:
                    return operation(find_ancestor_attribute(x, name, attribute), value)
            return criteria_func

        if self.data_opts.get('inclusion_rule'):
            for name, rules in self.data_opts['inclusion_rule'].items():
                for rule in rules:
                    if len(rule) == 3:
                        attribute, relationship, value = rule
                        modifier = mattr = mrel = mval = None
                    elif len(rule) == 7:
                        attribute, relationship, value, modifier, mattr, mrel, mval = rule
                    else:
                        raise ValueError('Unknown rule format')

                    if relationship not in operations:
                        raise ValueError('Unknown operation')

                    operation = operations[relationship]
                    mop = operations[mrel] if modifier else None
                    criteria_func = make_criteria_func(name, attribute, value, operation, modifier, 
                                                       mattr, mop, mval)
                    criteria[name].append(criteria_func)

        return criteria
      
    def refer(self, data, stop_at='', is_spectrum=False):
        if (  # all the conditions in which reference data should not be subtracted
                not self.data_opts.get('evoked') or
                self.period_type == self.current_reference_period_type or
                (stop_at and stop_at != self.name) or
                (self.data_type == 'spectrum' and not is_spectrum)
        ):
            return data
        if self.reference.name == 'period':
            ref_data = self.reference.data
        else:
            ref_data = self.reference.period.data
        mean_ref_data = np.mean(ref_data) if ref_data.shape[0] == 1 else np.mean(ref_data, axis=1)[:, np.newaxis]
        data -= mean_ref_data
        return data

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

        else:  # recursively call
            child_vals = [child.get_average(base_method, axis=axis, stop_at=stop_at, **kwargs) for child in
                          self.children]
            # Filter out nan values and arrays that are all NaN
            child_vals_filtered = [x for x in child_vals
                                   if not (isinstance(x, np.ndarray) and np.isnan(x).all())
                                   and not (isinstance(x, float) and np.isnan(x))]
            if axis is None:  # compute mean over all dimensions
                return np.nanmean(np.array(child_vals_filtered))
            else:  # compute mean over provided dimension
                return np.nanmean(np.array(child_vals_filtered), axis=axis)
            

    def get_sum(self, base_method, axis=0, stop_at='period'):
        if self.name == stop_at:  # we are at the base case and will call the base method
            if hasattr(self, base_method) and callable(getattr(self, base_method)):
                return getattr(self, base_method)()
            else:
                raise ValueError(f"Invalid base method: {base_method}")

        else:  # recursively call
            child_vals = [child.get_sum(base_method, axis=axis, stop_at=stop_at) for child in
                          self.children]
            # Filter out nan values and arrays that are all NaN
            child_vals_filtered = [x for x in child_vals
                                   if not (isinstance(x, np.ndarray) and np.isnan(x).all())
                                   and not (isinstance(x, float) and np.isnan(x))]
            if axis is None:
                return np.sum(np.array(child_vals_filtered))
            else:
                return np.sum(np.array(child_vals_filtered), axis=axis)
            

    @cache_method
    def get_sem(self, collapse_sem_data=False):
        """
        Calculates the standard error of an object's data. If object's data is a vector, it will always return a float.
        If object's data is a matrix, the `collapse_sem_data` argument will determine whether it returns the standard
        error of its children's average data points or whether it computes the standard error over children maintaining
        the original shape of children's data, as you would want, for instance, if graphing a standard error envelope
        around firing rate over time.
        """

        if self.data_opts.get('sem_level'):
            sem_children = get_descendants(self, level=self.data_opts.get('sem_level'))
        else:
            sem_children = self.children

        data = []
        for child in sem_children:
            cd = child.data
            if (isinstance(cd, np.ndarray) and (np.isnan(cd).all() or not(len(cd)))
                ) or (isinstance(cd, float) and np.isnan(cd)):
                continue
            else:
                data_to_append = np.mean(cd) if collapse_sem_data else cd
                data.append(data_to_append)
        return sem(data)

    @cache_method
    def get_median(self, stop_at='event', extend_by=None, select_by=None):
        def collect_vals(obj, vals=None):
            if vals is None:
                vals = []
            if obj.name == stop_at or not hasattr(obj, 'children'):
                if extend_by is not None:
                    sources = expand_sources(obj, extend_by)
                    sources = [src for src in sources if select_sources(src, select_by)]
                    vals.extend(sources)
                else:
                    vals.append(obj)
            else:
                if hasattr(obj, 'children') and obj.children:
                    [collect_vals(child, vals) for child in obj.children if select_sources(child, select_by)]
            return vals

        def expand_sources(obj, extension):
            sources = [obj]
            if extension:
                if 'frequency' in extension:
                    sources = [freq_bin for src in sources for freq_bin in src.frequency_bins]
                if 'time' in extension:
                    sources = [time_bin for src in sources for time_bin in src.time_bins]
            return sources

        def select_sources(obj, selection):
            if selection is None: return True
            for name, key, val in selection:
                for ancestor in obj.ancestors: 
                    if name == ancestor.name and hasattr(ancestor, key) and getattr(ancestor, key) != val:
                        return False
            return True

        vals_to_summarize = collect_vals(self)

        return np.median([obj.data for obj in vals_to_summarize])

    @cache_method
    def get_scatter_points(self):
        """Returns a list of points of the data values for an object's children for use on, e.g. a bar graph"""
        if not self.children:
            return []
        return [np.nanmean(child.data) for child in self.children]

    def find_experiment(self):
        if self.name == 'experiment':
            return self
        elif hasattr(self, 'experiment'):
            return self.experiment
        elif hasattr(self, 'parent'):
            return self.parent.find_experiment()
        else:
            return None


class TimeBin(Data):
    name = 'time_bin'

    def __init__(self, i, val, parent):
        self.parent = parent
        self.identifier = i
        self.val = val
        self.hierarchy = parent.hierarchy
        self.position = self.get_position_in_period_time_series()
        self.period = None
        for ancestor in self.ancestors:
            if ancestor.name == 'period':
                self.period = ancestor
                break
            elif hasattr(ancestor, 'period'):
                self.period = ancestor.period
                break
        if self.period:
            self.period_type = self.period.period_type

    @property
    def data(self):
        return self.val

    @property
    def time(self):
        if self.data_type == 'correlation':
            ts = np.arange(-self.data_opts['lags'], self.data_opts['lags'] + 1)/self.sampling_rate
        else:
            pre_stim, post_stim = [self.data_opts['events'][self.parent.period_type][val] 
                                   for val in ['pre_stim', 'post_stim']]
            ts = np.arange(-pre_stim, post_stim, self.data_opts['bin_size'])
        return ts[self.identifier]
        

    def get_position_in_period_time_series(self):
        if self.parent.name == 'event':
            self.parent.num_bins_per_event * self.parent.identifier + self.identifier
        else:
            return self.identifier


