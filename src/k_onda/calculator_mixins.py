from functools import wraps

from .arithmetic_calculators import Scale, Shift
from .calculator import Filter, Normalize, Threshold, ApplyMask, MedianFilter, ReduceDim, GroupSignals, SplitSignals


def with_key(cls):
    for name, method in list(vars(cls).items()):
        if callable(method) and not name.startswith('_'):
            @wraps(method)
            def wrapper(self, *args, _orig=method, key=None, **kwargs):
                calculator = _orig(self, *args, **kwargs)
                return calculator(self, key=key)
            setattr(cls, name, wrapper)
    return cls
            

@with_key
class SignalCalculatorMixin:
    def scale(self, factor):
        return Scale(factor)

    def shift(self, offset):
        return Shift(offset)
    
    def reduce(self, dim, method='mean'):
        return ReduceDim(dim, method)

    def normalize(self, method='rms'):
        return Normalize(method)

    def threshold(self, comparison, threshold):
        return Threshold(comparison, threshold)

    def apply_mask(self, mask):
        return ApplyMask(mask)

    def median_filter(self, kernel_sizes):
        return MedianFilter(kernel_sizes)


@with_key
class TimeSeriesCalculatorMixin(SignalCalculatorMixin):
    
    def filter(self, config):
        return Filter(config)
    
    
@with_key
class TimeFrequencyCalculatorMixin(SignalCalculatorMixin):
    
    def spectrogram(self, config):
        from .time_frequency_calculators import Spectrogram
        return Spectrogram(config)
    

class CollectionMixin:
    def group_signals(self, stack_dim=None):
        return GroupSignals(stack_dim=stack_dim)(self)
    

class SignalBundleMixin:
    def split_signals(self, stack_dim=None):
        return SplitSignals(stack_dim=stack_dim)(self)
                      