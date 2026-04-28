import numpy as np
import xarray as xr
from collections.abc import Iterable

from ..transformers.transformer_mixins import CalculateMixin, UnstackMixin, IntersectionMixin
from k_onda.transformers import SelectMixin
from k_onda.graph.traversal import build_generations, list_nodes, new_tree
from k_onda.central.registry import types


@types.register
class Signal(CalculateMixin, SelectMixin, IntersectionMixin):


    # strategy is one of: "lazy", "memory", "disk"
    def __init__(
        self,
        inputs,
        transform,
        data_schema,
        *,
        context=None,
        transformer=None,
        optimizer=None,
        origin=None,
        start=None,
        duration=None,
        source_signal=None,
        storage_strategy="lazy",
    ):
        self.inputs = inputs if isinstance(inputs, Iterable) else (inputs,)
        self.transform = transform
        self.transformer = transformer
        self._data_schema = data_schema
        self.context = context
        self.optimizers = []
        if optimizer:
            self.optimizers.append = optimizer
        self.origin = origin
        self.start = start or getattr(self.parent, 'start', None)
        self.duration = duration or getattr(self.parent, 'duration', None)
        self.source_signal = source_signal
        self._storage_strategy = storage_strategy
        self._cache = None
        self.conditions = {}
        self._validate_inputs()
        self._is_compiled = False
      
    
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
    def is_source(self):
        return not getattr(self, 'inputs', None)

    @property
    def data_schema(self):
        return self._data_schema
    
    @property
    def generations(self):
        return build_generations(self)
    
    @property
    def transform_history(self):
        func = lambda node, _: getattr(node, 'transformer', None)
        return build_generations(self, func)
    
    def compile(self, memo=None):
        memo = {} if memo is None else memo
        leaf = new_tree(self, memo=memo)
        plan = leaf.plan_on_signal()
        for node in list_nodes(plan):
            node._is_compiled = True
        return plan

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
        if not self._is_compiled:
            raise ValueError("You must call .compile() on a signal before accessing"
            "the .data property.")
        if self._cache is None:
            input_data = [input.data for input in self.inputs]
            self._cache = self.transform(*input_data)
        return self._cache

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
        # TODO: architectural debt. Threading data_identity through the origin
        # chain means deepcopying any signal drags the entire entity graph along,
        # causing hash failures on partially-constructed DataIdentity objects
        # (deepcopy uses __new__, not __init__). Short-term fix: add __deepcopy__
        # to DataIdentity to return self. Long-term fix: store data_identity_id
        # (uid) on the root signal and resolve it via a registry, so signals can
        # still belong to a DataIdentity without holding a live reference.
        if type(self.origin) in [tuple, list]:
            origin = None
            for entity in self.origin:
                if entity is not None:
                    origin = entity
                    break
        else:
            origin = self.origin

        return getattr(origin, "data_identity", None)
    
    def output_class_for_selection(self):
        pass

    
@types.register
class TimeSeriesSignal(Signal):
    dim_defaults = {"time": "s"}

    def __init__(self, inputs, transform, *, sampling_rate=None, **kwargs):
        super().__init__(inputs, transform, **kwargs)
        self.sampling_rate = sampling_rate or self._sampling_rate_from_context()

    def _sampling_rate_from_context(self):
        origin = self.origin
        if isinstance(origin, tuple):
            origin = next((o for o in origin if o is not None), None)
        return getattr(origin, 'sampling_rate', None)


@types.register
class TimeFrequencySignal(TimeSeriesSignal, CalculateMixin, SelectMixin):
    dim_defaults = {"time": "s", "frequency": "Hz"}


@types.register
class ScalarSignal(Signal):
    pass



@types.register
class DatasetSignal(Signal):

    def __init__(self, inputs, transform, **kwargs):
        super().__init__(inputs, transform, **kwargs)
        
    def __getitem__(self, key):
        return self.payload(key)
    
    def payload(self, key):
        def transform(data):
            if not isinstance(data, (xr.Dataset, dict)):
                raise TypeError("`data` must be a dictionary or xarray Dataset")
            return data[key]

        # TODO: I'm not sure that payload should be the same type for the DatasetSingals
        # For PointProcessSignal, yes.  Need to think about this.

        return type(self)(
            inputs=(self,),
            transform=transform,
            origin=self.origin,
            transformer=None,
            data_schema=self.data_schema[key]
        )


@types.register
class PointProcessSignal(Signal):

    def __init__(
            self, 
            inputs, 
            transform, 
            *, 
            sampling_rate = None,
            **kwargs):
        super().__init__(inputs, transform, **kwargs)
    
        self.sampling_rate = sampling_rate or getattr(self.parent, 'sampling_rate', None)

    def to_binary(self):
        if self.sampling_rate is None:
            raise ValueError("can't expand signal to binary without sampling_rate")
        return BinarySignal(parent=self, sampling_rate=self.sampling_rate)

    @property
    def neuron(self):
        # Convenience accessor for neuron-backed PointProcessSignals.
        from ..sources.spike_sources import Neuron

        if isinstance(self.data_identity, Neuron):
            return self.data_identity
        return None


@types.register
class DistributionSignal(Signal):
    pass


@types.register
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


@types.register
class ValidityMask(BinarySignal):
    pass


@types.register
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
        self._data_schema = data_schema
        self.transform = transform
        self.transformer = transformer
        self._cache = None
        self.signal_class = signal_class or type(self.signals[0])
        self.is_stack = True
        self._is_compiled = False

    def compile(self):
        # TODO: What this should eventually do is find the constituent signals, call compile on them
        # and rewrite the graph.
        raise ValueError("SignalStack cannot currently be compiled as a terminal node. " \
            "Compile a downstream signal or unstack before materialization.")
    
    def _materialize(self):
        if not self._is_compiled:
            raise ValueError("SignalStack cannot currently be compiled as a terminal node. " \
            "Compile a downstream signal or unstack before materialization.")
       
        if self._cache is None:
            if type(self.parent).__name__ == "Collection":
                self._cache = self.transform(self.parent.signals)
            else:
                self._cache = self.transform(self.parent.data)
        return self._cache
    
    @property
    def parent(self):
        if len(self.inputs) == 1 and getattr(self.inputs[0], 'is_stack', False):
            return self.inputs[0]
        return self.collection
        
    @property
    def data(self):
        return self._materialize()
    
    @property
    def data_schema(self):
        return self._data_schema

    def group_by(self):
        # Placeholder for future API.
        pass


@types.register
class AggregatedSignal(Signal):

    def _materialize(self):
        if not self._is_compiled:
            raise ValueError("You must call .compile() on an aggregated signal before accessing"
            "the .data property.")
        if self._cache is None:
            if hasattr(self.parent, 'signals'): # parent is a Collection
                self._cache = self.transform(self.parent.signals)
            else:
                self._cache = self.transform(self.inputs)

        return self._cache
    

@types.register
class IndexedSignal(Signal):

    def __init__(
            self, 
            inputs, 
            transform, 
            **kwargs):
        super().__init__(inputs, transform, **kwargs)

    def _materialize(self):
        if not self._is_compiled:
            raise ValueError("You must call .compile() on a signal before accessing"
            "the .data property.")
        if self._cache is None:

            if all(isinstance(inp, Signal) for inp in self.inputs):
                input_data = [input.data for input in self.inputs]
                self._cache = self.transform(*input_data)
            else:
                self._cache = self.transform()
        return self._cache
    
    def kmeans(self, n_clusters=8, **kwargs):
        from k_onda.transformers import KMeans
        return KMeans(n_clusters=n_clusters, **kwargs)(self)
    
    def classify(self, label_name, label_spec=None, label_func=None):
        from k_onda.sinks import Classify
        node = self.compile()
        chain = []
        while isinstance(node, IndexedSignal):
            chain.append(node)
            node = node.inputs[0] if node.inputs else None

        return Classify(label_name, label_spec=label_spec, label_func=label_func)(*chain)
        

@types.register
class SelectorSignal(Signal):


    def __init__(
            self, 
            inputs, 
            transform, 
            **kwargs):
        super().__init__(inputs, transform, **kwargs)
    
    @property
    def output_class(self):
        return getattr(self.inputs[0], "output_class", type(self.inputs[0]))
    
  
    
    
    

    



   
