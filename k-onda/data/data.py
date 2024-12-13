import numpy as np
from collections import defaultdict


from base.base import Base
from utils.utils import cache_method, always_last, operations
from utils.math_functions import sem

# TODO a lot of methods in here need to deal with dictionaries for the granger case,
# like sem, mean, etc.


class Data(Base):

    def __init__(self):
        self.parent = None

    @property
    def name(self):
        return self._name

    def get_calc(self, calc_type=None):
        if calc_type is None:
            calc_type = self.calc_type
            self.calc_mode = 'normal'  # the other cases set calc_mode later
            return getattr(self, f"get_{calc_type}")()
        if self.calc_opts.get('percent_change'):
            return self.percent_change
        elif self.calc_opts.get('evoked'):
            return self.evoked
        
    def resolve_calc_fun(self, calc_type):
        stop_at=self.calc_opts.get('base', 'event')
        if self.name == stop_at:
            return getattr(self, f"_get_{calc_type}")()
        else:
            return self.get_average(f"get_{calc_type}", stop_at=stop_at)
    
    @property
    def calc(self):
        return self.get_calc()
    
    def fetch_opts(self, list_of_opts=None):
        if list_of_opts is not None:
            return (self.calc_opts.get(opt) for opt in list_of_opts)
        
    def include(self, check_ancestors=True):
        return self.select(self.filter, check_ancestors=check_ancestors)
    
    def active(self):
        return self.include() and self in self.parent.children
    
    @property
    def included_children(self):
        if hasattr(self, 'children'):
            return [child for child in self.children if child.include()]
        else:
            return None
    
    @property
    def has_children(self):
        return len(self.children)
    
    @property
    def has_grandchildren(self):
        return any([child.has_children for child in self.children])
    
    @property
    def parent_identifier(self):
        try:
            return self.parent.identifier
        except AttributeError:
            return None
        
    @property
    def grandparent_identifier(self):
        try:
            return self.ancestors[-3].identifier
        except IndexError:
            return None
        
    def get_child_by_identifier(self, id):
        return [child for child in self.children if child.identifier == id][0]
            
    def select(self, filters, check_ancestors=False):

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
    
    @cache_method
    def get_average(self, base_method, stop_at='event', level=0, axis=0, *args, **kwargs):
        """
        Recursively calculates the average of the values of the computation in the base method on the object's
        descendants.

        Parameters:
        - base_method (str): Name of the method called when the recursion reaches the base case.
        - stop_at (str): `name` attribute of the base case object the `base_method` is called on.
        - level (int): a counter for how deep recursion has descended; limits the cache.
        - axis (int or None): Specifies the axis across which to compute the mean.
        - **kwargs: Additional keyword arguments to be passed to the base method.

        Returns:
        float or np.array: The mean of the data values from the object's descendants.
        """

        if not self.include():
            return float('nan')
                   
        if self.name == stop_at:  # we are at the base case and will call the base method
            if hasattr(self, base_method) and callable(getattr(self, base_method)):
                return getattr(self, base_method)(*args, **kwargs)
            else:
                raise ValueError(f"Invalid base method: {base_method}")

        if not len(self.children):
            return float('nan')
        
        child_vals = []

        for child in self.children:
            if not child.include():
                continue
            child_val = child.get_average(
                base_method, level=level+1, stop_at=stop_at, axis=axis, **kwargs)
            if not self.is_nan(child_val):
                child_vals.append(child_val)
            
        if len(child_vals) and isinstance(child_vals[0], dict):
            # Initialize defaultdict to automatically start lists for new keys
            result_dict = defaultdict(list)

            # Aggregate values from each dictionary under their corresponding keys
            for child_val in child_vals:
                for key, value in child_val.items():
                    result_dict[key].append(value)

            # Calculate average of the list of values for each key
            return_dict = {key: np.nanmean(values, axis) 
                            for key, values in result_dict.items()}
            return return_dict
                
        else:
            return np.nanmean(child_vals, axis)
        
    @property
    def mean(self):
        return np.mean(self.calc)
    
    @property
    def sem(self):
        return self.get_sem(collapse_sem_data=True)
    
    @property
    def sem_envelope(self):
        return self.get_sem(collapse_sem_data=False)
    
    def get_mean(self, axis=0):
        return np.mean(self.calc, axis=axis)
        
    def get_sem(self, collapse_sem_data=False):
        """
        Calculates the standard error of an object's data. If object's data is a vector, it will always return a float.
        If object's data is a matrix, the `collapse_sem_data` argument will determine whether it returns the standard
        error of its children's average data points or whether it computes the standard error over children maintaining
        the original shape of children's data, as you would want, for instance, if graphing a standard error envelope
        around firing rate over time.
        """

        if self.calc_opts.get('sem_level'):
            sem_children = self.get_descendants(stop_at=self.calc_opts.get('sem_level'))
        else:
            sem_children = self.children

        if isinstance(sem_children[0].calc, dict):

            return_dict = {}

            for key in sem_children[0]:
                vals = [child.calc[key] for child in sem_children if not self.is_nan(child.calc)]
                if collapse_sem_data:
                    vals = [np.mean(val) for val in vals]
                return_dict[key] = sem(vals) 

            return return_dict

        else:
            vals = [child.calc for child in sem_children if not self.is_nan(child.calc)]

            if collapse_sem_data:
                vals = [np.mean(val) for val in vals]

            return sem(vals)
                
    @staticmethod
    def extend_into_bins(sources, extend_by):
        if 'frequency' in extend_by:
            sources = [freq_bin for src in sources for freq_bin in src.frequency_bins]
        if 'time' in extend_by:
            sources = [time_bin for src in sources for time_bin in src.time_bins]
        return sources
            
    def get_median(self, stop_at=None, extend_by=None):
        if not stop_at:
            stop_at = self.calc_opts.get('stop_at')
        vals_to_summarize = self.get_descendants(stop_at=stop_at)
        vals_to_summarize = self.extend_into_bins(vals_to_summarize, extend_by)
        return np.median([obj.calc for obj in vals_to_summarize])

    def is_nan(self, value):
        if isinstance(value, float) and np.isnan(value):
            return True
        elif isinstance(value, np.ndarray) and np.all(np.isnan(value)):
            return True
        elif isinstance(value, dict) and all(self.is_nan(val) for val in value.values()):
            return True                                
        else:
            return False
    
    @property
    def concatenation(self):
        return self.concatenate()
    
    def concatenate(self, depth=1, attr='calc', method=None):
        return self.apply_fun_to_accumulated_data(
            np.concatenate, depth=depth, attr=attr, method=method)
    
    @property
    def stack(self):
        return self.get_stack()
    
    def get_stack(self, depth=1, attr='calc', method=None):
        return self.apply_fun_to_accumulated_data(
            np.vstack, depth=depth, attr=attr, method=method)

    def apply_fun_to_accumulated_data(self, fun, depth=1, attr='calc', method=None):
        f = lambda x: (
            getattr(x, method)() if method else getattr(x, attr)
            ) if hasattr(x, method if method else attr) else None
        
        sources = self.sort_accumulated(self.accumulate(max_depth=depth))[depth]
        results = [f(source) for source in sources]
        return fun(results)
    
    def sort_accumulated(self, accumulated, depth='all'):
        sort_by = self.calc_opts.get('sort_by')
        if sort_by:
            attr = sort_by[0]
            for level, obj_list in accumulated.items():
                if depth in [level, 'all']:
                    accumulated[level] = sorted(
                        obj_list, key=lambda obj: getattr(obj, attr, always_last))
                    if sort_by[1] == 'descending':
                        accumulated[level].reverse()
        return accumulated

    @property
    def grandchildren_stack(self):
        return self.get_stack(depth=2)
   
    @property
    def scatter(self):
        return [child.mean for child in self.included_children]
    
    @property
    def grandchildren_scatter(self):
        return [gchild.mean for gchild in self.accumulate(max_depth=2)[2]]
    
    @property
    def greatgrandchildren_scatter(self):
        return [ggchild.mean for ggchild in self.accumulate(max_depth=3)[3]]
    
    def accumulate(self, max_depth=1, depth=0, accumulator=None):
        
        if accumulator is None:
            accumulator = defaultdict(list)
        
        accumulator[depth].append(self)

        if depth != max_depth and self.included_children:
            for child in self.included_children:
                child.accumulate(max_depth, depth + 1, accumulator)
        
        return accumulator

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
        return [self] + self.parent.ancestors
        
    def get_descendants(self, stop_at=None, descendants=None, all=False):
   
        if descendants is None:
            descendants = []

        if self.name == stop_at or not hasattr(self, 'children'):
            descendants.append(self)
        else:
            if all:
                descendants.append(self)

            for child in self.children:
                child.get_descendants(descendants=descendants)

        return descendants

    @property
    def has_reference(self):
        return hasattr(self, 'reference') and self.reference is not None
    
    @property
    def percent_change(self):
        return self.get_percent_change()
    
    @property
    def evoked(self):
        return self.get_evoked()
    
    def get_evoked(self):
        fun = lambda orig, ref: orig - ref
        return self.get_comparison_calc('evoked', fun)

    def get_percent_change(self):
        fun = lambda orig, ref: orig/np.mean(ref) * 100 - 100
        return self.get_comparison_calc('percent_change', fun)

    def get_comparison_calc(self, comparison, fun):
        # This complexity is solving problems like "we'd like to get see a value
        # for a period with a reference of its own parent unit without triggering
        # infinite recursion"
        self.calc_mode = comparison
        comparison_dict = self.calc_opts.get(comparison, {'level': 'period'})
        level = comparison_dict['level']
        if level not in self.hierarchy or (
            self.hierarchy.index(self.name) < self.hierarchy.index(level)):
            return self.get_average(f'get_{comparison}', stop_at=level)
        # we are currently at a lower tree level than the % change ref level
        elif self.hierarchy.index(self.name) > self.hierarchy.index(level):
            ref_obj = [anc for anc in self.ancestors if anc.name == level][0]
            ref = self.get_ref(ref_obj, comparison_dict['reference'])
        # we are currently at the % change ref level
        else: 
            ref = self.get_ref(self, comparison_dict['reference'])
        orig = getattr(self, f"get_{self.calc_type}")()
        return fun(orig, ref)

    def get_ref(self, obj, reference_period_type):
        if obj.has_reference:
            return getattr(obj.reference, f"get_{self.calc_type}")
        else:
            return obj.get_reference_calc(reference_period_type)
        
    def get_reference_calc(self, reference_period_type):
        orig_period_type = self.selected_period_type
        self.selected_period_type = reference_period_type
        reference_calc = getattr(self, f"get_{self.calc_type}")()
        self.selected_period_type = orig_period_type
        return reference_calc
        
        
  
