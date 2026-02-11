from .arithmetic_calculators import Scale, Shift
from .calculator import Filter, Normalize, Threshold, ApplyMask


class SignalCalculatorMixin:
    def scale(self, factor):
        return Scale(factor)(self)
    
    def shift(self, offset):
        return Shift(offset)(self)
    
    def normalize(self, method='rms'):
        return Normalize(method)(self)
    
    def threshold(self, comparison, threshold):
        return Threshold(comparison, threshold)(self)
    
    def apply_mask(self, mask):
        return ApplyMask(mask)(self)


class TimeSeriesCalculatorMixin(SignalCalculatorMixin):
    def filter(self, config):
        return Filter(config)(self)
    
    
    
    

class TimeFrequencyCalculatorMixin(SignalCalculatorMixin):
    
    def spectrogram(self, config):
        from .time_frequency_calculators import Spectrogram
        return Spectrogram(config)(self)