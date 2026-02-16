import pint
import numpy as np
import xarray as xr
from collections import defaultdict

from .arithmetic_calculators import Scale, Shift

from .calculator_mixins import (
    SignalCalculatorMixin, 
    TimeSeriesCalculatorMixin, 
    TimeFrequencyCalculatorMixin,
    SignalBundleMixin
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
        signal.origin = kwargs.pop('origin', None)
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
            origin=None,
            storage_strategy="lazy"
            ):
        self.parent = parent
        self.transform = transform
        self.calculator = calculator
        self.origin = origin
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

    def __init__(self, parent, transform, calculator, origin=None, storage_strategy="lazy"):
        super().__init__(parent, transform, calculator, origin, storage_strategy)
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

    def __init__(self, parent, transform, calculator, origin=None, storage_strategy="lazy"):
        super().__init__(parent, transform, calculator, origin, storage_strategy)
        self.sampling_rate = getattr(self.parent, 'sampling_rate', None) # there might be reason to overwrite this in a transform
       
    def to_binary(self):
        if self.sampling_rate is None:
            raise ValueError("can't expand signal to binary without sampling_rate")
        return BinarySignal(parent=self, sampling_rate=self.sampling_rate)
    
    def payload(self, key):
        def transform(data):
            if not isinstance(data, (xr.Dataset, dict)):
                raise TypeError("`data` must be a dictionary or xarray Dataset")
            return data[key]
        
        return EventSignal(
            parent=self,
            transform=transform,  
            origin=self.origin,
            calculator=None
        )
    

class BinarySignal(Signal):  
    # stored as boolean array at some sampling rate
    # supports & | ~

    def __init__(
            self, 
            parent, 
            transform, 
            calculator, 
            origin=None,
            storage_strategy="lazy", 
            sampling_rate=None, 
            length=None,
            endpoints=None,
            intervals=None):
        super().__init__(parent, transform, calculator, origin, storage_strategy=storage_strategy)
        self.intervals = intervals
        self.length = length
        self.endpoints = endpoints
        self.sampling_rate = sampling_rate or getattr(self.parent, 'sampling_rate', None)

    def __and__(self, other):
        return Intersection()(other, self)

    def __rand__(self, other):
        # called when: non_mask & mask
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


# experiment.all_neurons.signals().median_filter(key='waveform').group_by_identity().extract_features('waveform', 'firing_rate')
# all_neurons()



class SignalBundle(SignalCalculatorMixin, SignalBundleMixin):

    def __init__(self, parent, transform=None, calculator=None, signal_class=None):
        self.parent = parent
        self.signals = self.parent.signals
        self.transform = transform
        self.calculator = calculator
        self._cached_data = None
        self.signal_class = signal_class or type(self.signals[0])
        self.is_bundle = True

    @property
    def data(self):
        return self._materialize()

    def _materialize(self):
        if self._cached_data is None:
            if type(self.parent).__name__ == 'Collection':
                self._cached_data = self.transform(self.parent.signals)
            else:
                self._cached_data = self.transform(self.parent.data)
        return self._cached_data
    
    @property
    def data(self):
        return self._materialize()

    def group_by(self):
        # what does this have to do?
        # the last calculator to operate on this has already altered the chain on self.components
        # but data needs to be assigned back to them.  this should maybe call a method like
        # assign_data_to_components
        # then we need to know the *function* to use for grouping.
        # maybe this could *either* live on the identity or be passed in.
        # so a neuron could know that its firing rate is the total session time of its components 
        # divided by the num spikes of its components
        # wait is this doing the job of extract features?
        # what else would this method do?  just pass forward a Collection?
        # a Collection of what, though?  Maybe smaller SignalBundles, one for each neuron
        # (which in the common case are actally just going to be a bundle of one signal)
        # then extract_features 
        # but how is this going to work?
        # a signal's data is self.transform(parent.data)
        # I did a thing in call where I added the transform to the individual signal but I'm wodering if that's right 
        # because really splitting the signal bundle back into signals, that can then be grouped into a collection
        # is a transform.  
        #  group_signals (and then split_signals) should maybe be thought of as more like calculators
        # they can still be convenience methods on collections/signal bundles respectively via mixins
        # I don't know that individual signals that are part of singal bundles need transforms applied during __call__ 
        # on a calculator.  I think their parent, signal bundle, will have that information. 
        # as long as group_signals and split_signals are transforms that record provenance.  
        # so maybe I should make specialized calculators that handle grouping and splitting 
        pass

  


  