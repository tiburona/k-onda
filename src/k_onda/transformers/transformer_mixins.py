from collections.abc import Iterable
import pint
from .selector import DimBounds

from k_onda.utils import w_units, is_unitful


class CalculateMixin:

    def add(self, other, key=None, key_output_mode=None):
        return self.shift(other, key=key, key_output_mode=key_output_mode)
    
    def substract(self, other, key=None, key_output_mode=None):
        return self.shift(-other, key=key, key_output_mode=key_output_mode)
    
    def multiply_by(self, other, key=None, key_output_mode=None):
        return self.scale(other, key=key, key_output_mode=key_output_mode)
    
    def divide_by(self, other, key=None, key_output_mode=None):
        return self.scale(1/other, key=key, key_output_mode=key_output_mode)
    
    def scale(self, factor, key=None, key_output_mode=None):
        from . import Scale
        return Scale(factor)(self, key=key, key_output_mode=key_output_mode)

    def shift(self, offset, key=None, key_output_mode=None):
        from . import Shift
        return Shift(offset)(self, key=key, key_output_mode=key_output_mode)

    def reduce(self, dim, method="mean", key=None, key_output_mode=None):
        from . import ReduceDim
        return ReduceDim(dim, method)(self, key=key, key_output_mode=key_output_mode)

    def normalize(self, method="rms", dim=None, key=None, key_output_mode=None):
        from . import Normalize
        return Normalize(method, dim)(self, key=key, key_output_mode=key_output_mode)

    def median_filter(self, kernel_sizes, key=None, key_output_mode=None, ):
        from . import MedianFilter
        return MedianFilter(kernel_sizes)(self, key=key, key_output_mode=key_output_mode)

    def filter(self, config, key=None, key_output_mode=None):
        from . import Filter
        return Filter(config)(self, key=key, key_output_mode=key_output_mode)

    def spectrogram(self, config, key=None, key_output_mode=None):
        from . import Spectrogram
        return Spectrogram(config)(self, key=key, key_output_mode=key_output_mode)
    
    def threshold(self, comparison, threshold, key=None, key_output_mode=None):
        from . import Threshold
        return Threshold(comparison, threshold)(self, key=key, key_output_mode=key_output_mode)

    def apply_mask(self, mask, key=None, key_output_mode=None):
        from . import ApplyMask
        return ApplyMask(mask)(self, key=key, key_output_mode=key_output_mode)
    
    def fwhm(self, config=None, key=None, key_output_mode=None):
        from . import FWHM
        if config is None:
            config = {}
        return FWHM(**config)(self, key=key, key_output_mode=key_output_mode)


class IntersectionMixin:

    def intersection(self, other, tolerance_decimals=9):
        from . import Intersection
        return Intersection(tolerance_decimals)(self, other)
    


class PointProcessMixin:
    
    def rate(self, intervals=None, exclude_initial=None, key=None, key_output_mode=None):
        from . import Rate
        return Rate(intervals=intervals, exclude_initial=exclude_initial)(self, key=key, key_output_mode=key_output_mode)
    


class SelectMixin:
    def select(
            self, 
            selection=None, 
            new_dim=None, 
            mode=None, 
            conditions=None, 
            units=None, 
            window=None, 
            metadim=None,
            **kwargs
            ):
        
        from . import Selector
        from k_onda.loci import Interval, IntervalSet, MarkerSet

        if isinstance(selection, MarkerSet):
            raise NotImplementedError("Need to deal with MarkerSets later")
        
        if new_dim is not None and not isinstance(selection, IntervalSet):
            raise ValueError("You can't create a new_dim unless you're selecting an " \
            "IntervalSet (or MarkerSet when it gets implemented)")
        
        if kwargs:
            dim_bounds, conditions = self.parse_kwargs(kwargs, selection, conditions)

        if dim_bounds:
            selection = self.locus_from_dim_bounds(dim_bounds, units, conditions, metadim)

        if isinstance(selection, str):
            if selection == 'epochs':
                selection = self.origin.session.epochs
            elif selection == 'events':
                selection = self.origin.session.events
            else:
                raise ValueError(f"Unknown value {selection} passed to `select`.")
            
        if new_dim:
            if not metadim and not selection.metadim:
                raise ValueError("If you want to create a `new_dim` you must supply a `metadim`.")
            elif metadim and not selection.metadim:
                selection.metadim = metadim
            
        if window:
            if isinstance(selection, (Interval, IntervalSet)):
                raise ValueError("It doesn't make sense to define a Window on something that's" \
                "already an Interval or IntervalSet")
            else:
              
                window_span = w_units(
                    window, 
                    selection.metadim, 
                    units, 
                    self.session.experiment.ureg
                    )
                
                selection = selection.to_intervals(window_span)
                window = DimBounds({selection.metadim: window})
            
        return Selector(mode, selection, new_dim, window)(self)
    
    def parse_kwargs(self, kwargs, selection, conditions):

        dim_bounds = {}

        # if either conditions or selection have been explicitly passed to `select`,
        # assume all kwargs belong to the other.
        if conditions:
            dim_bounds = kwargs
        elif selection:
            conditions = kwargs
        else:
            dim_bounds = {k: v for k, v in kwargs.items if self.is_inferrably_a_dim(k)}
            conditions = {k: v for k, v in kwargs.items if not self.is_inferrably_a_dim(k)}
        
        return dim_bounds, conditions

    def is_inferrably_a_dim(self, key):
        coord_contains_patterns = ['time', 'freq', 'spike', 'sample', 'loc', 'pos']
        coord_is_patterns = ['x', 'y']
        if any([string in key for string in coord_contains_patterns]):
            return True
        if any([string == key for string in coord_is_patterns]):
            return True
        else:
            return False
        
    def filter_selection_by_conditions(self, selection, conditions):
        return selection.where(**conditions)

    def locus_from_dim_bounds(self, dim_bounds, units, conditions, metadim):
        
        from k_onda.loci import Interval, IntervalSet

        dim = list(dim_bounds.keys())[0]
        span = list(dim_bounds.values())[0]

        ureg=self.origin.session.ureg

        cls = Interval if isinstance(dim_bounds, dict) else IntervalSet
            
        return cls(dim, span, ureg=ureg, units=units, metadim=metadim, conditions=conditions)     


class StackMixin:
    def stack_signals(self, dim=None):
        from . import StackSignals
        return StackSignals(dim=dim)(self)


class UnstackMixin:
    def unstack_signals(self, dim=None):
        from . import UnstackSignals
        return UnstackSignals(dim=dim)(self)
    

class AggregateMixin:
    def aggregate(self, method='mean'):
        from . import Aggregator
        return Aggregator(method=method)(self)
    



        

    


    


