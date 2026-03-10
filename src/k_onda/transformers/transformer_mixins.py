

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
    


class GenericSelectMixin:
    def select(self, selection=None, mode='pushdown', units=None, **dim_endpoints):
        from . import Selector
        return Selector(mode, selection, units, **dim_endpoints)(self)
    

class DimSelectMixin:
    def band(self, freq_band):
        return self.select(frequency=(freq_band.f_lo, freq_band.f_hi), mode=freq_band.mode)

    def span(self, epoch):
        return self.select(time=(epoch.t0, epoch.t1), mode=epoch.mode)


class SelectMixin(GenericSelectMixin, DimSelectMixin):
    pass
   


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
    



        

    


    


