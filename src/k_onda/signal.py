from .arithmetic_calculators import Scale, Shift

from .calculator_mixins import (
    SignalCalculatorMixin, 
    TimeSeriesCalculatorMixin, 
    TimeFrequencyCalculatorMixin
    )



class Signal(SignalCalculatorMixin):
    # strategy is one of: "lazy", "memory", "disk"
    def __init__(self, parent, transform, config, storage_strategy="lazy"):
        self.parent = parent
        self.transform = transform
        self.config = config
        self.storage_strategy = storage_strategy
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

    def window(self, epoch):

        if not hasattr(self, 'parent'):
            return self.data.sel(time=slice(epoch.t0, epoch.t1))

        parent_data = self.parent.window(epoch) # TODO: add padding
        if self.transform is None:
            data = parent_data
        else:
            data = self.transform(parent_data)
        
        return data
    

    

class TimeSeriesSignal(Signal, TimeSeriesCalculatorMixin, TimeFrequencyCalculatorMixin):

    def __init__(self, parent, transform, config, storage_strategy="lazy"):
        super().__init__(parent, transform, config, storage_strategy)
        self.sampling_rate = self.parent.sampling_rate # there might be reason to overwrite this in a transform
        


class TimeFrequencySignal(Signal):
    pass


class EventSignal(Signal):
    pass


class EpochScalarSignal(Signal):
    pass
      