from collections import namedtuple
from dataclasses import dataclass
from functools import partial, wraps
import numpy as np
import xarray as xr
from typing import Protocol, runtime_checkable
from copy import deepcopy


@runtime_checkable
class SignalLike(Protocol):
    data: ...


def resolve_target_data(self, data, key_spec=None):
    if key_spec is None:
        return data
    
    key = key_spec.input_name

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
            merged[self.name] = result
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
        self.padlen = padlen or {}
        self.signal_class = signal_class
        self.kwargs = kwargs

    def __call__(self, data):
        return self.fn(data)
    

@dataclass
class Policies:
    pass

KeySpec = namedtuple('KeySpec', 'input_name output_mode')


class Transformer:

    def __call__(self, parent, key=None, key_output_mode=None):
        from ..sources import Collection, GroupedCollection

        key_spec = KeySpec(input_name=key, output_mode=key_output_mode)

        if isinstance(parent, GroupedCollection):
            return self._call_on_grouped_collection(parent, key_spec)
        
        if isinstance(parent, Collection):
            return self._call_on_collection(parent, key_spec)
        
        return self._call_on_signal(parent, key_spec)
    
    def _call_on_grouped_collection(self, grouped_collection, key_spec):

        from ..sources import GroupedCollection
        
        group_on = getattr(grouped_collection, "group_on", None)

        return GroupedCollection(
            groups={
                k: self._call_on_collection(v, key_spec)
                for k, v in grouped_collection.items()
            },
            group_on=group_on,
        )

    def _call_on_collection(self, collection, key_spec):
        from ..sources import Collection

        return Collection(
            [self._call_on_signal(signal, key_spec) for signal in collection]
             )
    
    def _call_on_signal(self, signal, key_spec):
        self._input_validation(signal)
        child_class = self.get_child_class(signal)
        transform = self._get_transform(signal, key_spec)
        child_signal = child_class(parent=signal, transform=transform, transformer=self)
        return child_signal
    
    @staticmethod
    def get_output_class(entity):
        return getattr(entity, "output_class", type(entity))


class CalculatorPolicies(Policies):
    pass
    

class Calculator(Transformer):
    name = None
    obligate_output_class = None
    key_mode = 'replace'  # replace | append | standalone
    require_some_finite = True
    require_all_finite = False
    allow_empty = False

    def _input_validation(self, parent):
        if not isinstance(parent, SignalLike):
            raise ValueError("Calculators can only operate on Data Components, Signals, " \
            "StackedSignals, Collections, and GroupedCollections.")
        
    def get_child_class(self, parent):
        # If we're operating on a StackedSignal, we'll return a StackedSignal.
        # If this calculator has an obligate Signal type to return, we'll return one
        # of those. Otherwise we'll query the parent signal to get the output class.
        if getattr(parent, "is_stack", False):
            return type(parent)
        return self.obligate_output_class or self.get_output_class(parent)

    def _get_transform(self, parent, key_spec):

        apply_kwargs = self._get_apply_kwargs(parent, key_spec)
        transform_kwargs = self._get_transform_kwargs(parent, apply_kwargs)
        return Transform(partial(self._apply, **apply_kwargs), **transform_kwargs)

    def _get_apply_kwargs(self, parent_signal, key_spec):
        result = {'key_spec': key_spec}
        result.update(self._get_extra_apply_kwargs(parent_signal))
        return result

    def _get_transform_kwargs(self, parent, apply_kwargs):
        if getattr(parent, "is_stack", None):
            signal_class = (self.obligate_output_class or 
                            self.get_output_class(parent.signals[0]))
        else:
            signal_class = None

        kwargs = {"signal_class": signal_class}
        kwargs.update(self._get_extra_transform_kwargs(parent, apply_kwargs))

        return kwargs

    # These two methods get overridden
    def _get_extra_apply_kwargs(self, parent):
        return {}

    def _get_extra_transform_kwargs(self, parent, apply_kwargs):
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
            
        try:
            mag = data.pint.magnitude  # preferred for pint-aware arrays
        except Exception:
            mag = data
        arr = np.asarray(mag)

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
        return result

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
