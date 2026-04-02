from collections import namedtuple
from dataclasses import dataclass
from functools import partial, wraps
import numpy as np
import xarray as xr
from copy import deepcopy

from k_onda.central import SignalLike, DatasetSchema


def resolve_target_data(self, data, key):
    if key is None:
        return data

    if not isinstance(data, xr.Dataset):
        raise ValueError(
            f"{type(self).__name__} received key='{key}' but input is not a dataset."
        )
    if key not in data:
        raise KeyError(f"{key} was not found in dataset variables: {list(data.keys())}")
    return data[key]


def merge_keys(self, data, result, key_spec):
        
        output_mode = key_spec.output_mode
        
        if not isinstance(data, xr.Dataset):
            if output_mode is None:
                return result
            else:
                raise ValueError("`key_mode` provided when data is not xr.Dataset")
        
        if output_mode is None:
            output_mode = self.key_mode
        
        if output_mode == 'standalone':
            return result

        if output_mode == 'replace':
            merged = data.copy()
            merged[key_spec.input_name] = result
            return merged

        if output_mode == 'append':
            if self.name in data:
                raise ValueError(f"output_mode is 'append' and key {self.name} exists in data.")
            merged = data.copy()
            
            merged[self.name] = result
            return merged

        raise ValueError("Unknown key mode")


def with_key_access(func):
    @wraps(func)
    def wrapper(self, data, *args, key_spec=None, **kwargs):

        if key_spec is None:
            return func(self, data, *args, **kwargs)
        
        key = key_spec.input_name

        subset = resolve_target_data(self, data, key)

        result = func(self, subset, *args, **kwargs)

        merged_result = merge_keys(self, data, result, key_spec)
    
        return merged_result

    return wrapper


class Transform:
    """A transform function with optional metadata."""

    def __init__(self, fn, padlen=None, signal_class=None, **kwargs):
        self.fn = fn
        self.signal_class = signal_class
        self.padlen = padlen
        self.kwargs = kwargs
        

    def __call__(self, data):
        return self.fn(data)
    

@dataclass
class Policies:
    pass

KeySpec = namedtuple('KeySpec', 'input_name output_mode', defaults=[None, 'replace'])


class Transformer:

    fixed_output_class = None

    def __call__(self, input, key=None, key_output_mode=None):
        from ..sources import Collection, CollectionMap

        key_spec = KeySpec(input_name=key, output_mode=key_output_mode)

        if isinstance(input, CollectionMap):
            return self._call_on_collection_map(input, key_spec)
        
        if isinstance(input, Collection):
            return self._call_on_collection(input, key_spec)
        
        return self._call_on_signal(input, key_spec)
    
    def _call_on_collection_map(self, collection_map, key_spec):

        from ..sources import CollectionMap
        
        group_on = getattr(collection_map, "group_on", None)

        return CollectionMap(
            groups={
                k: self._call_on_collection(v, key_spec)
                for k, v in collection_map.items()
            },
            group_on=group_on,
        )

    def _call_on_collection(self, collection, key_spec):
        from ..sources import Collection

        return Collection(
            [self._call_on_signal(signal, key_spec) for signal in collection]
             )
    
    def _call_on_signal(self, signal, key_spec):
        self._validate_input(signal, key_spec=key_spec)
        output_class = self.resolve_output_class(signal)
        transform = self._get_transform(signal, key_spec)
        output_schema = self.make_output_schema(signal.data_schema, key_spec=key_spec)

        output_signal = output_class(
            inputs=(signal,), 
            transform=transform, 
            transformer=self,
            data_schema=output_schema
            )
        return output_signal
    
    def resolve_output_class(self, input):
        # If we're operating on a StackedSignal, preserve the stack type.
        # If this calculator has a fixed output class, return that.
        # Otherwise ask the parent signal what it would produce.
        if getattr(input, "is_stack", False):
            return type(input)
        return self.fixed_output_class or self._infer_output_class(input)

    def _infer_output_class(self, entity):
        return getattr(entity, "output_class", type(entity))
    
    def make_output_schema(self, *input_schemas, key_spec):
        """Compute the output schema."""

        input_schema = input_schemas[0]
        key = key_spec.input_name
        output_mode = key_spec.output_mode or getattr(self, 'key_mode', 'replace')

        if isinstance(input_schema, DatasetSchema) and key is not None:
            key_schema = input_schema[key]
            new_key_schema = self.output_schema(key_schema)
            if output_mode == 'standalone':
                return new_key_schema
            elif output_mode == 'replace':
                return input_schema.replace_key(self.name, new_key_schema)
            elif output_mode == 'append':
                return input_schema.add_key(self.name, new_key_schema)

        else:
            # Plain Schema input 
            return self.output_schema(input_schema)
        
    # this gets overridden
    def output_schema(self, input_schema):
        return input_schema


class CalculatorPolicies(Policies):
    pass
    

class Calculator(Transformer):
    name = None
    key_mode = 'replace'  # replace | append | standalone
    require_some_finite = True
    require_all_finite = False
    allow_empty = False

    def _validate_input(self, input, **kwargs):
        if not isinstance(input, SignalLike):
            raise ValueError("Calculators can only operate on Data Components, Signals, " \
            "StackedSignals, Collections, and GroupedCollections.")

    def _get_transform(self, input, key_spec):

        apply_kwargs = self._get_apply_kwargs(input, key_spec)
        transform_kwargs = self._get_transform_kwargs(input, apply_kwargs)
        return Transform(partial(self._apply, **apply_kwargs), **transform_kwargs)

    def _get_apply_kwargs(self, input, key_spec):
        result = {'key_spec': key_spec}
        result.update(self._get_extra_apply_kwargs(input))
        return result

    def _get_transform_kwargs(self, input, apply_kwargs):
        if getattr(input, "is_stack", None):
            signal_class = (self.fixed_output_class or
                            self._infer_output_class(input.signals[0]))
        else:
            signal_class = None

        kwargs = {"signal_class": signal_class}
        kwargs.update(self._get_extra_transform_kwargs(input, apply_kwargs))

        return kwargs

    # These two methods get overridden
    def _get_extra_apply_kwargs(self, input):
        return {}

    def _get_extra_transform_kwargs(self, input, apply_kwargs):
        return deepcopy(apply_kwargs)
    
    @with_key_access
    def _apply(self, data, *args, **kwargs):
        self._validate_data(data, **kwargs)
        result = self._apply_inner(data, *args, **kwargs)
        if isinstance(result, tuple):
            result, wrap_kwargs = result
        else:
            wrap_kwargs = {}
        result = self._wrap_result(result, data, **wrap_kwargs)
        return result

    def _validate_data(self, data, **kwargs):
        if not isinstance(data, (xr.DataArray, xr.Dataset)):
            raise ValueError("data must be an xarray DataArray or Dataset.")
        
        if hasattr(data, "data_vars") and len(data.data_vars) == 0:
            raise ValueError(f"{type(self)}: Event dataset has no variables.")

        def get_magnitude(arr):
            try:
                mag = arr.pint.magnitude  # preferred for pint-aware arrays
            except Exception:
                mag = arr
            return np.asarray(mag)

        if isinstance(data, xr.Dataset):
            arrays = [get_magnitude(arr) for arr in data.data_vars.values()]
        else:
            arrays = [get_magnitude(data)]
             
        for arr in arrays:
            if not self.allow_empty and arr.size == 0:
                raise ValueError(f"{type(self)}: An empty data array is not allowed.")
            if self.require_all_finite and not np.isfinite(arr).all():
                raise ValueError(f"{type(self)}: All values must be finite.")
            if self.require_some_finite and not np.isfinite(arr).any():
                raise ValueError(f"{type(self)}: The data contains no finite values.")

        dim = kwargs.get("dim") or getattr(self, "dim", None)
        if dim and dim not in data.dims:
            raise ValueError(f"dim {dim} was not found in the data.")
        
    # This method gets overridden by every Calculator.
    def _apply_inner(self, data, *args, **kwargs):
        return data
    
    def _wrap_result(self, result, *args):
        return result.assign_attrs({'transformer': self._calculator_name()})

    def _calculator_name(self):
        return self.name or type(self).__name__.lower()  


class PaddingCalculator(Calculator):
    def _get_extra_transform_kwargs(self, parent, apply_kwargs):
        extra_args = super()._get_extra_transform_kwargs(parent, apply_kwargs)
        extra_args.update({"padlen": self._compute_padlen(parent, apply_kwargs)})
        return extra_args

    # This gets overridden.
    def _compute_padlen(self, parent, apply_kwargs):
        return {}
