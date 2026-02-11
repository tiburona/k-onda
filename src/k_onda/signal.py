import pint
import numpy as np

from .arithmetic_calculators import Scale, Shift

from .calculator_mixins import (
    SignalCalculatorMixin, 
    TimeSeriesCalculatorMixin, 
    TimeFrequencyCalculatorMixin
    )

from .select_mixin import SelectMixin
from .calculator import Intersection, ApplyMask



class Signal(SignalCalculatorMixin, SelectMixin):

    @classmethod
    def from_data(cls, data, **kwargs):
        """Create a signal from pre-computed data."""
        signal = object.__new__(cls)
        signal.parent = None
        signal.transform = None
        signal.calculator = None
        signal._cached_data = data
        signal.storage_strategy = "memory"
        for key, value in data.attrs.items():
            setattr(signal, key, value)
        for key, value in kwargs.items():
            setattr(signal, key, value)
        return signal

    # strategy is one of: "lazy", "memory", "disk"
    def __init__(
            self, 
            parent, 
            transform,
            calculator, 
            storage_strategy="lazy"
            ):
        self.parent = parent
        self.transform = transform
        self.calculator = calculator
        self.storage_strategy = storage_strategy,
        self._cached_data = None

    def __truediv__(self, other):
        return Scale(1 / other)(self)
    
    def __mul__(self, other):
        return Scale(other)(self)
    
    def __sub__(self, other):
        return Shift(-other)(self)  # if you have a Shift calculator
    
    def _materialize(self):
        if self._cached_data is None:
            self._cached_data = self.transform(self.parent.data)
        return self._cached_data
    
    @property
    def data(self):
        return self._materialize()

    

class TimeSeriesSignal(Signal, TimeSeriesCalculatorMixin, TimeFrequencyCalculatorMixin):

    dim_defaults = {'time': 's'}

    def __init__(self, parent, transform, calculator, storage_strategy="lazy"):
        super().__init__(parent, transform, calculator, storage_strategy)
        self.sampling_rate = self.parent.sampling_rate # there might be reason to overwrite this in a transform
        


class TimeFrequencySignal(Signal):
    dim_defaults = {'time': 's', 'frequency': 'Hz'}
  


class EventSignal(Signal):
    pass


class EpochScalarSignal(Signal):
    pass
      

class EventSignal(Signal):
    # stored as timestamps
    # .window(epoch) returns timestamps in range
    
    def to_binary(self, sampling_rate):
        return BinarySignal(parent=self, sampling_rate=sampling_rate)


class BinarySignal(Signal):  
    # stored as boolean array at some sampling rate
    # supports & | ~

    def __init__(
            self, 
            parent, 
            transform, 
            calculator, 
            storage_strategy="lazy", 
            sampling_rate=None, 
            length=None,
            endpoints=None,
            intervals=None):
        super().__init__(parent, transform, calculator, storage_strategy=storage_strategy)
        self.intervals = intervals
        self.length = length
        self.endpoints = endpoints
        self.sampling_rate = sampling_rate or getattr(self.parent, 'sampling_rate', None)

    def __rand__(self, other):
        # called when: non_mask & mask
        if isinstance(other, BinarySignal):
            return Intersection()(other, self)
        elif isinstance(other, Signal):
            return ApplyMask()(other, self)
    
    @property
    def data(self):
        if self.intervals:
            return self._bool_train_from_intervals()
        else:
            return self._materialize()

        
    def _bool_train_from_intervals(self):
    
        length = self.length or (self.endpoints[1] - self.endpoints[0])/self.sampling_rate
        arr = np.full(length, False)

        for start, end in self.intervals:
            arr[slice(start, end)] = True

        return arr
            


   

class ValidityMask(BinarySignal):


    pass