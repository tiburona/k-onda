from functools import wraps


def with_key(cls):
    for name, method in list(vars(cls).items()):
        if callable(method) and not name.startswith("_"):

            @wraps(method)
            def wrapper(self, *args, _orig=method, key=None, 
                        key_output_mode=None, **kwargs):
                calculator = _orig(self, *args, **kwargs)
                return calculator(self, key=key, 
                                  key_output_mode=key_output_mode)

            setattr(cls, name, wrapper)
    return cls


@with_key
class CalculateMixin:

    def add(self, other):
        return self.shift(other)
    
    def substract(self, other):
        return self.shift(-other)
    
    def multiply_by(self, other):
        return self.scale(other)
    
    def divide_by(self, other):
        return self.scale(1/other)
    
    def scale(self, factor):
        from .transformers import Scale
        return Scale(factor)

    def shift(self, offset):
        from .transformers import Shift
        return Shift(offset)

    def reduce(self, dim, method="mean"):
        from .transformers import ReduceDim
        return ReduceDim(dim, method)

    def normalize(self, method="rms", dim=None):
        from .transformers import Normalize
        return Normalize(method, dim)

    def median_filter(self, kernel_sizes):
        from .transformers import MedianFilter
        return MedianFilter(kernel_sizes)

    def filter(self, config):
        from .transformers import Filter
        return Filter(config)

    def spectrogram(self, config):
        from .transformers import Spectrogram
        return Spectrogram(config)
    
    def threshold(self, comparison, threshold):
        from .transformers import Threshold
        return Threshold(comparison, threshold)

    def apply_mask(self, mask):
        from .transformers import ApplyMask
        return ApplyMask(mask)


class IntersectionMixin:

    def intersection(self, other, tolerance_decimals=9):
        from .transformers import Intersection
        return Intersection(tolerance_decimals)(self, other)
    

@with_key
class EventMixin:
    
    def event_rate(self, intervals=None, exclude_initial=None):
        from .transformers import Rate
        return Rate(intervals=intervals, exclude_initial=exclude_initial)(self)
    

@with_key
class GenericSelectMixin:
    def select(self, selection=None, **dim_endpoints):
        from .transformers import Selector
        return Selector(selection=selection, **dim_endpoints)
    

class DimSelectMixin:
    def band(self, freq_band):
        return self.select(frequency=(freq_band.f_lo, freq_band.f_hi))

    def window(self, epoch):
        return self.select(time=(epoch.t0, epoch.t1))


class SelectMixin(GenericSelectMixin, DimSelectMixin):
    pass


class StackMixin:
    def stack_signals(self, dim=None):
        from .transformers import StackSignals
        return StackSignals(dim=dim)(self)


class UnstackMixin:
    def unstack_signals(self, dim=None):
        from .transformers import UnstackSignals
        return UnstackSignals(dim=dim)(self)
    


    


