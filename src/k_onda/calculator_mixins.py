from .arithmetic_calculators import Scale, Shift
from .calculator import Filter, Normalize


class SignalCalculatorMixin:
    def scale(self, factor):
        return Scale(factor)(self)
    
    def shift(self, offset):
        return Shift(offset)(self)


class TimeSeriesCalculatorMixin(SignalCalculatorMixin):
    def filter(self, config):
        return Filter(config)(self)
    
    def normalize(self, method='rms'):
        return Normalize(method)(self)
    

class TimeFrequencyCalculatorMixin(SignalCalculatorMixin):
    
    def spectrogram(self, config):
        from .time_frequency_calculators import Spectrogram
        return Spectrogram(config)(self)