import numpy as np
import xarray as xr

from ..calculator_mixins import CalculateMixin, UnstackMixin, SelectMixin, IntersectionMixin


class Signal(CalculateMixin, SelectMixin, IntersectionMixin):


    # strategy is one of: "lazy", "memory", "disk"
    def __init__(
        self,
        parent,
        transform,
        transformer=None,
        origin=None,
        start=None,
        duration=None,
        storage_strategy="lazy",
    ):
        self.parent = parent
        self.transform = transform
        self.transformer = transformer
        self.origin = origin
        self.start = start or self.parent.start
        self.duration = duration or self.parent.duration
        self._storage_strategy = storage_strategy
        self._cached_data = None
        self._lineage = []
        self._history = None
        
    @property
    def origin(self):
        if hasattr(self, "_origin") and self._origin is not None:
            return self._origin
        return getattr(self.parent, "origin", None)

    @origin.setter
    def origin(self, value):
        self._origin = value
     
    @property
    def lineage(self):
        if not self._lineage and self.parent:
            return self._collect_lineage(self, self._lineage)
        else:
            return self._lineage
        
    def _collect_lineage(self, signal, accum):
        accum.append(signal)
        if getattr(signal, 'parent', None):
            return self._collect_lineage(signal.parent, accum)
        else:
            return accum
        
    @property
    def history(self):
        if self._history is None:
            self._history = [signal.transform for signal in self.lineage]
        return self._history
    
    def __add__(self, other):
        return self.add(other)
    
    def __sub__(self, other):
        return self.subtract(other)
    
    def __mul__(self, other):
        return self.multiply_by(other)
    
    def __truediv__(self, other):
        return self.divide_by(other)

    def _materialize(self):
        if self._cached_data is None:
            self._cached_data = self.transform(self.parent.data)
        return self._cached_data

    @property
    def data(self):
        return self._materialize()
    
    @property
    def endpoints(self):
        return {
            dim: (self.data.coords[dim][0], self.data.coords[dim][-1]) 
            for dim in self.data
            }

    @property
    def data_identity(self):
        if type(self.origin) in [tuple, list]:
            origin = None
            for entity in self.origin:
                if entity is not None:
                    origin = entity
                    break
        else:
            origin = self.origin

        return getattr(origin, "data_identity", None)


class TimeSeriesSignal(Signal):
    dim_defaults = {"time": "s"}

    def __init__(
            self, 
            parent, 
            transform, 
            transformer=None, 
            origin=None, 
            start=None, 
            duration=None,
            storage_strategy="lazy"
    ):
        super().__init__(
            parent, 
            transform, 
            transformer, 
            origin, 
            start, 
            duration, 
            storage_strategy
            )
        # there might be reason to overwrite this in a transform
        self.sampling_rate = self.parent.sampling_rate


class TimeFrequencySignal(TimeSeriesSignal, CalculateMixin, SelectMixin):
    dim_defaults = {"time": "s", "frequency": "Hz"}


class ScalarSignal(Signal):
    pass


class EpochScalarSignal(Signal):
    pass


class EventSignal(Signal):
    # stored as timestamps
    # .window(epoch) returns timestamps in range

    def __init__(
        self,
        parent,
        transform,
        transformer=None,
        origin=None,
        time_key=None,
        start=None,
        duration=None,
        storage_strategy="lazy",
    ):
        super().__init__(
            parent, 
            transform, 
            transformer, 
            origin, 
            start, 
            duration, 
            storage_strategy)
        
        # there might be reason to overwrite this in a transform
        self.sampling_rate = getattr(self.parent, "sampling_rate", None)
        self.time_key = time_key or getattr(self.parent, "time_key", None)

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
            calculator=None,
        )

    @property
    def neuron(self):
        # Convenience accessor for neuron-backed EventSignals.
        from ..sources.spike_sources import Neuron

        if isinstance(self.data_identity, Neuron):
            return self.data_identity
        return None


class DistributionSignal(Signal):
    pass


class BinarySignal(Signal):
    # stored as boolean array at some sampling rate
    # supports & | ~

    def __init__(
        self,
        parent,
        transform,
        transformer=None,
        origin=None,
        start=None,
        duration=None,
        storage_strategy="lazy",
        sampling_rate=None,
        length=None,
        endpoints=None,
        intervals=None,
    ):
        super().__init__(
            parent, 
            transform, 
            transformer, 
            origin, 
            start, 
            duration, 
            storage_strategy)
        
        self.intervals = intervals
        self.length = length
        self.endpoints = endpoints
        self.sampling_rate = sampling_rate or getattr(self.parent, "sampling_rate", None)

    def __and__(self, other):
        return self.intersection(other)

    def __rand__(self, other):
        # called when: non_mask & mask
        return other.apply_mask(self)

    @property
    def data(self):
        if self.intervals:
            return self._bool_train_from_intervals()
        return self._materialize()

    def _bool_train_from_intervals(self):
        length = self.length or (self.endpoints[1] - self.endpoints[0]) / self.sampling_rate
        arr = np.full(length, False)

        for start, end in self.intervals:
            arr[slice(start, end)] = True

        return arr


class ValidityMask(BinarySignal):
    pass


class SignalStack(CalculateMixin, UnstackMixin):
    def __init__(self, parent, transform=None, calculator=None, signal_class=None):
        self.parent = parent
        self.signals = self.parent.signals
        self.transform = transform
        self.calculator = calculator
        self._cached_data = None
        self.signal_class = signal_class or type(self.signals[0])
        self.is_stack = True

    def _materialize(self):
        if self._cached_data is None:
            if type(self.parent).__name__ == "Collection":
                self._cached_data = self.transform(self.parent.signals)
            else:
                self._cached_data = self.transform(self.parent.data)
        return self._cached_data

    @property
    def data(self):
        return self._materialize()

    def group_by(self):
        # Placeholder for future API.
        pass
