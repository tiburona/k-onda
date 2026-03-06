import numpy as np
import xarray as xr
from collections.abc import Iterable

from ..calculator_mixins import CalculateMixin, UnstackMixin, SelectMixin, IntersectionMixin
from k_onda.graph.traversal import build_generations




class Signal(CalculateMixin, SelectMixin, IntersectionMixin):


    # strategy is one of: "lazy", "memory", "disk"
    def __init__(
        self,
        inputs,
        transform,
        data_schema,
        *,
        transformer=None,
        optimizer=None,
        origin=None,
        start=None,
        duration=None,
        coord_map=None,
        source_signal=None,
        storage_strategy="lazy",
    ):
        self.inputs = inputs if isinstance(inputs, Iterable) else (inputs,)
        self.transform = transform
        self.transformer = transformer
        self.data_schema = data_schema
        self.optimizers = []
        if optimizer:
            self.optimizers.append = optimizer
        self.origin = origin
        self.start = start or getattr(self.parent, 'start', None)
        self.duration = duration or getattr(self.parent, 'duration', None)
        self.coord_map = coord_map
        self.source_signal = source_signal
        self._storage_strategy = storage_strategy
        self._cached_data = None
        self._validate_inputs()
      
    
    @property
    def parent(self):
        return self.inputs[0] if len(self.inputs) == 1 else None
        
    @property
    def origin(self):
        if hasattr(self, "_origin") and self._origin is not None:
            return self._origin
        self._origin = getattr(self.parent, "origin", None)
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = value
    
    @property
    def data_dims(self):
        return self.data_schema.dims
    
    @property
    def generations(self):
        return build_generations(self)
    
    @property
    def transform_history(self):
        func = lambda node, _: getattr(node, 'transformer', None)
        return build_generations(self, func)

    def _validate_inputs(self):
        return True

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
            input_data = [input.data for input in self.inputs]
            self._cached_data = self.transform(*input_data)
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

    def __init__(self, inputs, transform, *, sampling_rate=None, **kwargs):
        super().__init__(inputs, transform, **kwargs)
        self.sampling_rate = sampling_rate or self.parent.sampling_rate


class TimeFrequencySignal(TimeSeriesSignal, CalculateMixin, SelectMixin):
    dim_defaults = {"time": "s", "frequency": "Hz"}


class ScalarSignal(Signal):
    pass


class EpochScalarSignal(Signal):
    pass


class PointProcessSignal(Signal):

    def __init__(
            self, 
            inputs, 
            transform, 
            *, 
            sampling_rate = None, 
            coord_map=None, 
            **kwargs):
        super().__init__(inputs, transform, **kwargs)
    
        self.sampling_rate = sampling_rate or getattr(self.parent, 'sampling_rate', None)
        self.coord_map = coord_map or getattr(self.parent, "coord_map", None)

    def __getitem__(self, key):
        return self.payload(key)

    def to_binary(self):
        if self.sampling_rate is None:
            raise ValueError("can't expand signal to binary without sampling_rate")
        return BinarySignal(parent=self, sampling_rate=self.sampling_rate)

    def payload(self, key):
        def transform(data):
            if not isinstance(data, (xr.Dataset, dict)):
                raise TypeError("`data` must be a dictionary or xarray Dataset")
            return data[key]

        return PointProcessSignal(
            parent=self,
            transform=transform,
            origin=self.origin,
            transformer=None,
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

    def __init__(self, inputs, transform, data_schema, *, sampling_rate=None, intervals=None, **kwargs):
        super().__init__(inputs, transform, data_schema, **kwargs)
        
        self.intervals = intervals
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
        length = self.duration or (self.endpoints[1] - self.endpoints[0]) / self.sampling_rate
        arr = np.full(length, False)

        for start, end in self.intervals:
            arr[slice(start, end)] = True

        return arr


class ValidityMask(BinarySignal):
    pass


class SignalStack(CalculateMixin, UnstackMixin):
    def __init__(
            self, 
            data_schema,
            collection=None, 
            inputs=None, 
            transform=None, 
            transformer=None, 
            signal_class=None):

        if collection is not None:
            self.collection = collection
            self.signals = collection.signals
            self.inputs = tuple(collection.signals)
        elif inputs is not None:
            self.collection = inputs[0].collection
            self.signals = inputs[0].collection.signals
            self.inputs = inputs
        else:
            raise ValueError(f"collection or inputs must be provided for SignalStack")
        self.data_schema = data_schema
        self.transform = transform
        self.transformer = transformer
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
    def parent(self):
        if len(self.inputs) == 1 and getattr(self.inputs[0], 'is_stack', False):
            return self.inputs[0]
        return self.collection
        
    @property
    def data(self):
        return self._materialize()
    
    @property
    def data_dims(self):
        return self.data_schema.dims

    def group_by(self):
        # Placeholder for future API.
        pass
