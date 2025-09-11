from collections import defaultdict
import numpy as np
import xarray as xr

from k_onda.utils import cache_method, always_last, sem, is_truthy, \
    drop_inconsistent_coords, round_coords, fill_missing_arrays


# TODO a lot of methods in here need to deal with dictionaries for the granger case,
# like sem, mean, etc.


class Aggregates:



    def get_calc(self, calc_type=None):
        if calc_type is None:
            calc_type = self.calc_type
        if self.calc_opts.get('percent_change'):
            return self.percent_change
        elif self.calc_opts.get('evoked'):
            return self.evoked
        elif self.calc_opts.get('histogram'):
            return self.histogram
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
    def calc_result(self):
        if 'concatenation' in self.calc_opts:
            return self.concatenation
        else:
            return self.calc

    @property
    def calc(self):
        return self.get_calc()

    @staticmethod
    def xmean(child_vals, axis=None):
        """
        Always returns an xr.DataArray.

        - If child_vals is empty/falsey (but zeros are allowed), return a NaN scalar DataArray.
        - If child_vals is a scalar or size-1 array/DataArray, return a *scalar* (0-D) DataArray.
        - Otherwise, compute the mean (optionally along `axis`), preserving attrs when possible.
        """
        if not is_truthy(child_vals, zero_ok=True):
            # NaN scalar DA
            return xr.DataArray(np.nan)

        def _as_da_scalar(x):
            """Coerce x to a scalar (0-D) DataArray, preserving attrs if x is a DA."""
            if isinstance(x, xr.DataArray):
                # squeeze to drop any size-1 dims, keep attrs
                return x.squeeze(drop=True)
            # np.ndarray (size==1) or python/np scalar -> scalar DA
            return xr.DataArray(np.asarray(x).squeeze())

        # Normalize scalars / size-1 inputs to a scalar DA
        if isinstance(child_vals, (float, np.floating)) or \
        (isinstance(child_vals, (np.ndarray, xr.DataArray)) and np.size(child_vals) == 1):
            return _as_da_scalar(child_vals)

        # From here on, we have non-trivial arrays / lists of DAs
        if axis is None:
            if isinstance(child_vals, xr.DataArray):
                return child_vals.mean(skipna=True, keep_attrs=True)

            cleaned = drop_inconsistent_coords(child_vals)
            cleaned = round_coords(cleaned, decimals=8)
            agg = xr.concat(cleaned, dim="child", coords="minimal", compat="no_conflicts")
            return agg.mean(dim="child", skipna=True, keep_attrs=True)

        # axis provided
        if isinstance(child_vals, xr.DataArray):
            axis_name = child_vals.dims[axis] if isinstance(axis, int) else axis
            return child_vals.mean(dim=axis_name, skipna=True, keep_attrs=True)

        cleaned = drop_inconsistent_coords(child_vals)
        cleaned = round_coords(cleaned, decimals=8)
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
        
        children = self.children

        if not len(children):
            return float('nan')
        
        child_vals = []

        for child in children:
            if not child.include() and exclude:
                continue
            child_val = child.get_average(
                base_method, level=level+1, stop_at=stop_at, axis=axis, **kwargs)
            if not self.is_nan(child_val):
                child_vals.append(child_val)
        return self.xmean(child_vals, axis)
    
    @property
    def data_set(self):
        return xr.Dataset({k: getattr(self, k) for k in self.calc_opts.get('data_set', {})})
    
    @property
    def mean(self):
        return self.get_mean()
    
    @property
    def sem(self):
        return self.get_sem(collapse_sem_data=True)
    
    @property
    def sem_envelope(self):
        return self.get_sem(collapse_sem_data=False)
    
    def get_mean(self, axis=0):
        if self.calc_opts.get('concatenation'):
            concatenation = self.concatenation
            if concatenation.ndim == 1:
                return concatenation
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
        
           
        first_calc = sem_children[0].calc_result 

        if isinstance(first_calc, dict):
            return {key: agg_and_calc([child.calc[key] for child in sem_children 
                                       if not self.is_nan(child.calc)]) 
                                       for key in first_calc}

        return agg_and_calc([child.calc_result for child in sem_children if not self.is_nan(child.calc_result)])
                
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
        
    def concatenate(self, concatenator=None, concatenated=None, attrs=None, 
                    dim_xform=None, child_xform=None):
        """Concatenates data from descendant nodes using xarray."""

        if (concatenated == 'period' 
            and 'period' not in self.hierarchy 
            and any('calculator' in level for level in self.hierarchy)):
            concatenated = [level for level in self.hierarchy if 'calculator' in level][0]
        
        if attrs is None:
            attrs = ['calc']
        
        def get_func(attr):
            def func(obj):
                if not obj.include():
                    return float('nan')
                # Try to get the attribute from the object
                val = getattr(obj, attr, None)
                # If the attribute exists and is callable (i.e. a bound method), call it
                return val() if callable(val) else val
            return func
            
        if self.name == concatenator:
           
            # Fetch descendants from the correct level of the accumulator (base case)
            depth_index = self.hierarchy.index(concatenated) - self.hierarchy.index(self.name)

            children = self.accumulate(max_depth=depth_index)[concatenated]

            # Apply the function to each descendant
            if len(attrs) == 1:
              
                children_data = [
                    get_func(attrs[0])(child) for child in children
                ]
               
            else:
                children_data = [
                    xr.Dataset({attr: get_func(attr)(child) for attr in attrs}) 
                    for child in children
                ]
            if not children_data:
                return xr.DataArray([])
            
            children_data = fill_missing_arrays(children_data)
            
            if isinstance(children_data[0], (xr.DataArray, xr.Dataset)):
                if ((concatenator == 'animal' and self.calc_type != 'spike') or 
                    concatenator in ['unit', 'period']):
                   
                    result = xr.concat(children_data, dim="time")
                    if child_xform:
                        new_time = np.array([eval(child_xform)(child) for child in children])
                    else:
                        new_time = np.arange(result.sizes["time"])
                    if dim_xform:
                        new_time = eval(dim_xform)(new_time)
                    
                    result = result.assign_coords(time=("time", new_time))
                    # Reorder coords to match dims:
                    result = result.assign_coords(**{dim: result.coords[dim] for dim in result.dims})
                    return result

                else:
                    raise NotImplementedError(f"Concatenation not implemented for {concatenator}")              

        else:
            # Successively average levels of the hierarchy until we reach the concatenator (recursive case)
            children_data = [
                child.concatenate(concatenator=concatenator, concatenated=concatenated, 
                                attrs=attrs, dim_xform=dim_xform, child_xform=child_xform)
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
    
    @property
    def histogram(self):
        return self.get_histogram()
    
    def get_histogram(self):
        histogram = self.calc_opts.get('histogram', {})

        # Resolve base level and depth
        if histogram.get('base') is not None:
            base = histogram['base']
            max_depth = self.hierarchy.index(base) - self.index
        else:
            base = self.hierarchy[self.index + 1]
            max_depth = 1

        # Pull and flatten the data
        data_objects = self.accumulate(max_depth=max_depth)[base]
        data_values = np.array([getattr(do, f"get_{self.calc_type}")() for do in data_objects])      

        data_values = data_values[np.isfinite(data_values)]  # drop NaN/inf

        # Early return for no data
        if data_values.size == 0:
            empty = np.array([], dtype=float)
            return xr.DataArray(
                np.array([], dtype=int),
                dims=["bin"],
                coords={"bin": empty, "left": ("bin", empty), "right": ("bin", empty), "center": ("bin", empty)},
                name=f"{base}_hist",
                attrs={"range": None, "bins": 0, "density": False, "base": base},
            )

        # Histogram parameters
        num_bins = int(histogram.get('bins', 20))
        if histogram.get('range') is not None:
            lo, hi = histogram['range']
            lo, hi = float(lo), float(hi)
        else:
            lo = float(np.nanmin(data_values))
            hi = float(np.nanmax(data_values))
            if hi == lo:  # widen a touch to avoid a single degenerate bin
                eps = 1e-9 if hi == 0 else abs(hi) * 1e-6
                lo, hi = lo - eps, hi + eps

        density = bool(histogram.get('density', False))

        # Compute histogram
        counts, edges = np.histogram(data_values, bins=num_bins, range=(lo, hi), density=density)
        left = edges[:-1]
        right = edges[1:]
        center = (left + right) / 2.0

        # Build DataArray with rich coords
        da = xr.DataArray(
            counts,
            dims=["bin"],
            coords={
                "left": ("bin", left),              # left edges
                "right": ("bin", right),            # right edges
                "center": ("bin", center),          # bin centers
                "bin": np.arange(len(left)),        # simple integer bin index
            },
            name=f"{base}_hist",
            attrs={"range": (lo, hi), "bins": num_bins, "density": density, "base": base},
        )
        return da

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
        try:
            key = self.hierarchy[self.hierarchy.index(self.name) + 2]
        except IndexError:
            return []
        return [gchild.mean for gchild in self.accumulate(max_depth=2)[key]]
    
    @property
    def greatgrandchildren_scatter(self):
        return [ggchild.mean for ggchild in self.accumulate(max_depth=3)[3]]
    
    def accumulate(self, max_depth=1, depth=0, accumulator=None, filter=False): 
        
        if accumulator is None:
            accumulator = defaultdict(list)
        
        accumulator[self.name].append(self)

        if filter:
            included_children = self.included_children
        
            if depth != max_depth and included_children:
                for child in included_children:
                    child.accumulate(max_depth, depth + 1, accumulator)
        else:
            if depth != max_depth:
                for child in self.children:
                    child.accumulate(max_depth, depth + 1, accumulator)

        return accumulator   
    
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


        
        
  
