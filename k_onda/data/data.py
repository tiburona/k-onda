from collections import defaultdict
import numpy as np
from typing import Sequence, Union, Optional
import xarray as xr

from k_onda.base import Base
from k_onda.utils import cache_method, always_last, operations, sem, is_truthy, drop_inconsistent_coords

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
        if self.calc_opts.get('percent_change'):
            return self.percent_change
        elif self.calc_opts.get('evoked'):
            return self.evoked
        else:
            self.calc_mode = 'normal'  # the other cases set calc_mode later
            return getattr(self, f"get_{calc_type}")()
       
    def resolve_calc_fun(self, calc_type):
        stop_at=self.calc_opts.get('base', 'event')
        if self.name == stop_at:
            return getattr(self, f"get_{calc_type}_")()
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
    
    @property
    def experiment_wise_index(self):
        if self.name in ['period', 'event']:
            name = f'{self.calc_type}_{self.name}'
        else:
            name = self.name
        return getattr(self.experiment, f'all_{name}s').index(self)
        

    def sort(self, sort, items):
        if not sort:
            return items
        sort_key, order = sort
        sorted_lst = sorted(
            items, 
            key=lambda x: getattr(x, sort_key), 
            reverse=(order == 'descending'))
        return sorted_lst

    
    def sort_children(self, children):
        sort = self.calc_opts.get('sort', {}).get(self.name)
        if not sort:
            return children
        else:
            sort_key, order = sort
            sorted_children = sorted(
                children, 
                key=lambda x: getattr(x, sort_key), 
                reverse=(order == 'descending'))
            return sorted_children
            
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

    @staticmethod
    def xmean(child_vals, axis=None):

        if self.name == 'period':
            a = 'foo'
       
        if not is_truthy(child_vals):
            return xr.DataArray(np.nan)
        
        if axis is None:
            return child_vals.mean(skipna=True)
        
        if isinstance(child_vals, xr.DataArray): 

            # resolve axis name if an int was passed
            axis_name = (
                child_vals.dims[axis] if isinstance(axis, int) else axis
            )
            return child_vals.mean(dim=axis_name, skipna=True)
        
        cleaned = drop_inconsistent_coords(child_vals)

        agg = xr.concat(cleaned, dim="child", coords="minimal", compat="no_conflicts")
        axis_name = agg.dims[axis] if isinstance(axis, int) else axis or "child"
        return agg.mean(dim=axis_name, skipna=True, keep_attrs=True)
            
    @cache_method
    def get_average(self, base_method, stop_at='event', level=0, axis=0, exclude=True, *args, **kwargs):
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

        # TODO: methods that return dicts are currently broken but they will be fixed when
        # they return xr data sets
        
        if not self.include() and exclude:
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
            if not child.include() and exclude:
                continue
            child_val = child.get_average(
                base_method, level=level+1, stop_at=stop_at, axis=axis, **kwargs)
            if not self.is_nan(child_val):
                child_vals.append(child_val)
        if self.name == 'unit':
            a = 'foo'
        return self.xmean(child_vals, axis)
    
    @property
    def mean(self):
        return self.xmean(self.calc)
    
    @property
    def sem(self):
        return self.get_sem(collapse_sem_data=True)
    
    @property
    def mean_and_sem(self):
        return xr.Dataset({'mean': self.mean, 'sem': self.sem})
    
    @property
    def sem_envelope(self):
        return self.get_sem(collapse_sem_data=False)
    
    def get_mean(self, axis=0):
        return self.xmean(self.calc, axis=axis)
        
    def get_sem(self, collapse_sem_data=False):
        """
        Calculates the standard error of an object's data.

        - If `collapse_sem_data=True`, first computes the mean for each child before computing SEM.
        - If the data is a dictionary (e.g., multiple conditions), computes SEM for each key separately.
        """

        def agg_and_calc(vals):
            return sem([val.mean(skipna=True) for val in vals] if collapse_sem_data else vals)

        sem_children = (self.get_descendants(stop_at=self.calc_opts.get('sem_level'))
                        if self.calc_opts.get('sem_level') else self.children)

        first_calc = sem_children[0].calc  

        if isinstance(first_calc, dict):
            return {key: agg_and_calc([child.calc[key] for child in sem_children 
                                       if not self.is_nan(child.calc)]) 
                                       for key in first_calc}

        return agg_and_calc([child.calc for child in sem_children if not self.is_nan(child.calc)])
                
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
        arrays = [obj.calc for obj in vals_to_summarize]
        if arrays:
            np_arrays = []
            for arr in arrays:
                if hasattr(arr, "values"):
                    arr = arr.values
                if np.isscalar(arr):
                    arr = np.array([arr])
                np_arrays.append(np.ravel(arr))
            flattened = np.concatenate(np_arrays)
            if self.selected_brain_region == 'bla' and self.identifier == 'IG160':
                a = 'foo'
            return np.nanmedian(flattened)
        else:
            return float("nan")

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
        kwargs = self.calc_opts.get('concatenation', {})
        if not kwargs.get('concatenator'):
            kwargs['concatenator'] = self.name
        if not kwargs.get('concatenated'):
            kwargs['concatenated'] = self.children[0].name
        return self.concatenate(**kwargs)
        
    def concatenate(self, concatenator=None, concatenated=None, attrs=None, dim_xform=None):
        """Concatenates data from descendant nodes using xarray."""

        if attrs is None:
            attrs = ['calc']
        
        def get_func(attr):
            def func(obj):
                # Try to get the attribute from the object
                val = getattr(obj, attr, None)
                # If the attribute exists and is callable (i.e. a bound method), call it
                return val() if callable(val) else val
            return func
            
        if self.name == concatenator:
           
            # Fetch descendants from the correct level of the accumulator (base case)
            depth_index = self.hierarchy.index(concatenated) - self.hierarchy.index(self.name)

            # Apply the function to each descendant
            if len(attrs) == 1:
                children_data = [
                    get_func(attrs[0])(child) for child in self.accumulate(max_depth=depth_index)[concatenated]
                ]
                a = 'foo'
            else:
                children_data = [
                    xr.Dataset({attr: get_func(attr)(child) for attr in attrs}) 
                    for child in self.accumulate(max_depth=depth_index)[concatenated]
                ]
            if not children_data:
                return xr.DataArray([])
            
            if isinstance(children_data[0], (xr.DataArray, xr.Dataset)):
                if ((concatenator == 'animal' and self.calc_type != 'spike') or 
                    concatenator in ['unit', 'period']):

                    result = xr.concat(children_data, dim="time")
                    new_time = np.arange(result.sizes["time"])
                    if dim_xform:
                        new_time = eval(dim_xform)(new_time)
                    result = result.assign_coords(time=("time", new_time))
                    return result

                else:
                    raise NotImplementedError(f"Concatenation not implemented for {concatenator}")              

        else:
            # Successively average levels of the hierarchy until we reach the concatenator (recursive case)
            children_data = [
                child.concatenate(concatenator=concatenator, concatenated=concatenated, 
                                attrs=attrs, dim_xform=dim_xform)
                for child in self.children if child.include()
            ]
            
            concatenated_data = xr.concat(children_data, dim="child") if children_data else xr.DataArray([])
            return concatenated_data.mean(dim="child", skipna=True)
         
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
        key = self.hierarchy[self.hierarchy.index(self.name) + depth]
        sources = self.sort_accumulated(self.accumulate(max_depth=depth))[key]
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
        key = self.hierarchy[self.hierarchy.index(self.name) + 2]
        return [gchild.mean for gchild in self.accumulate(max_depth=2)[key]]
    
    @property
    def greatgrandchildren_scatter(self):
        return [ggchild.mean for ggchild in self.accumulate(max_depth=3)[3]]
    
    def accumulate(self, max_depth=1, depth=0, accumulator=None):
        
        if accumulator is None:
            accumulator = defaultdict(list)
        
        accumulator[self.name].append(self)
        
        if depth != max_depth and self.included_children:
            for child in self.included_children:
                child.accumulate(max_depth, depth + 1, accumulator)
        
        return accumulator   

    @property
    def ancestors(self):
        if self.name == 'experiment':
            return [self]
        if hasattr(self, 'parent'):
            return self.parent.ancestors + [self]
        
    @property
    def hierarchy(self):
        ancestor_names = [obj.name for obj in self.ancestors]
        obj = self
        while len(getattr(obj, 'children', [])):
            obj = obj.children[0]
            ancestor_names.append(obj.name)
        return ancestor_names
 
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
        # we are currently at a higher tree level than the comparison ref level
        if level not in self.hierarchy or (
            self.hierarchy.index(self.name) < self.hierarchy.index(level)):
            return self.get_average(f'get_{comparison}', stop_at=self.calc_opts.get('base', 'event'))
        # we are currently at a lower tree level than the comparison ref level
        elif self.hierarchy.index(self.name) > self.hierarchy.index(level):
            ref_obj = [anc for anc in self.ancestors if anc.name == level][0]
            ref = self.get_ref(ref_obj, comparison_dict['reference'])
        # we are currently at the comparison ref level
        else: 
            ref = self.get_ref(self, comparison_dict['reference'])
        orig = getattr(self, f"get_{self.calc_type}")()
        return fun(orig, ref)

    def get_ref(self, obj, reference_period_type):
        if obj.has_reference:
            return getattr(obj.reference, f"get_{self.calc_type}")
        else:
            return obj.get_reference_calc(reference_period_type)
        
    def get_reference_calc(self, reference):
        # record the original values of the period attributes
        period_attrs = ['selected_period_type', 'selected_period_types']
        orig_vals = [getattr(self, attr) for attr in period_attrs]

        # infer whether we are setting period_types or period_type from value of reference
        if isinstance(reference, str):
            attr_to_set = 'selected_period_type'
        else:
            attr_to_set = 'selected_period_types'
        
        # set the relevent attribute to its new value and get the calculation
        setattr(self, attr_to_set, reference)
        reference_calc = getattr(self, f"get_{self.calc_type}")()

        # reset the original value before returning
        if attr_to_set == 'selected_period_types':
            self.selected_period_type = orig_vals[0]
        else:
            self.selected_period_types = orig_vals[1]
                
        return reference_calc


        
        
  
