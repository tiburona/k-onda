from collections import namedtuple
from functools import partial
import numpy as np
import xarray as xr
from copy import deepcopy

from k_onda.central import type_registry as tr
from k_onda.utils import ValidationMixin


class Transform:
    """A transform function with optional metadata."""

    def __init__(self, fn, padlen=None, signal_class=None, key_spec=None, **kwargs):
        self.fn = fn
        self.signal_class = signal_class
        self.padlen = padlen
        self.key_spec = key_spec
        self.kwargs = kwargs

    def __call__(self, *data):
        if not data:
            return self.fn()
        return self.fn(*data)


KeySpec = namedtuple(
    "KeySpec", 
    "input_name output_mode", 
    defaults=[None, "replace"]
    )

MultiSpec = namedtuple(
    "MultiSpec", 
    "match_on collection_policy map_policy", 
    defaults=[None, "exactly_one", "exact_keys"]
    )


class Transformer(ValidationMixin):
    """A Transformer is a callable object that consumes a signal and returns a new signal.
    Transformers are configured at initialization and then immutable. When a Transformer
    is called on a group of Signals (e.g., a Collection or CollectionMap) it dispatches
    to signals.  The base Transformer class also handles key access when the signal
    has a DatasetSchema."""

    fixed_output_class = None
    arity = "one"
    accepted_data_types = (xr.DataArray, xr.Dataset)

    def __call__(
        self, 
        *inputs, 
        key=None, 
        key_output_mode=None, 
        match_on = None,
        multi_input_collection_policy="exactly_one", 
        multi_input_map_policy="exact_keys",
        ):

        self._validate_arity(len(inputs))

        key_spec = KeySpec(input_name=key, output_mode=key_output_mode)

        multi_spec = MultiSpec(
            match_on=match_on, 
            collection_policy=multi_input_collection_policy, 
            map_policy=multi_input_map_policy
            )

        input = inputs[0]

        if len(inputs) == 1:

            if isinstance(input, tr.CollectionMap):
                return self._call_on_collection_map(input, key_spec=key_spec)

            if isinstance(input, tr.Collection):
                return self._call_on_collection(input, key_spec=key_spec)

            if isinstance(input, tr.DataIdentity):
                return self._call_on_data_identity(input, key_spec=key_spec)

            return self._call_on_signal(input, key_spec=key_spec)

        else:
            if isinstance(input, tr.CollectionMap):
                return self._multi_input_call_on_collection_map(
                    *inputs, key_spec=key_spec, multi_spec=multi_spec
                    )
            
            if isinstance(input, tr.Collection):
                return self._multi_input_call_on_collection(
                    *inputs, key_spec=key_spec, multi_spec=multi_spec
                    )

            if isinstance(input, tr.DataIdentity):
                return self._multi_input_call_on_data_identity(*inputs, key_spec=key_spec)

            return self._multi_input_call_on_signal(*inputs, key_spec=key_spec)


    def _call_on_collection_map(self, collection_map, key_spec):

        group_on = getattr(collection_map, "group_on", None)

        return tr.CollectionMap(
            groups={
                k: self._call_on_collection(v, key_spec)
                for k, v in collection_map.items()
            },
            group_on=group_on,
        )

    def _multi_input_call_on_collection_map(self, *inputs, key_spec=None, multi_spec=None):
        if not all(isinstance(input, tr.CollectionMap) for input in inputs):
            raise TypeError(
                f"{self.format_call()}: received a CollectionMap and another input "
                "type as arguments."
                )
        if multi_spec.map_policy == "exact_keys":
            if not all(input.keys() == inputs[0].keys() for input in inputs[1:]):
                raise ValueError(
                    f"{self.format_call()} was called with multi_input_map_policy 'exact_keys' but "
                    "inputs' keys do not match."
                )

            return tr.CollectionMap(
                groups={
                    k: self._multi_input_call_on_collection(
                        *(input[k] for input in inputs), key_spec=key_spec, multi_spec=multi_spec
                        )
                    for k in inputs[0]
                }
            )
            

        else:
            raise NotImplementedError(
                f"{self.format_call()} was called with multi_input_map_policy " 
                f"{multi_spec.map_policy!r}, but this policy has not been implemented. "
                "Use 'exact_keys'."
            )  

    def _call_on_collection(self, collection, key_spec=None):
        if isinstance(collection.members[0], tr.Signal):
            return tr.Collection(
                [self._call_on_signal(signal, key_spec=key_spec) for signal in collection]
            )
        elif isinstance(collection.members[0], tr.DataIdentity):
            return tr.Collection(
                [self._call_on_data_identity(di, key_spec) for di in collection.members]
            )
        elif isinstance(collection.members[0], tr.Collection):
            return tr.Collection(
                [
                    self._call_on_collection(member, key_spec)
                    for member in collection.members
                ]
            )
        else:
            raise ValueError("What did you put in this collection, bro?")

    def _multi_input_call_on_collection(self, *inputs, key_spec=None, multi_spec=None):

        if not all(isinstance(input, tr.Collection) for input in inputs):
            raise TypeError(
                f"{self.format_call()}: received a Collection and another input "
                "type as arguments."
            )

        if multi_spec.collection_policy != "exactly_one":
            raise NotImplementedError(
                f"{self.format_call()} was called with multi_input_collection_policy " 
                f"{multi_spec.collection_policy!r}, but this policy has not been "
                "implemented. Use 'exactly_one'."
                )  
        
        return tr.Collection(
            [
                self._call_on_collection_members(member, inputs, key_spec, multi_spec) 
                for member in inputs[0].members
                ]
        )

    def _construct_conditions(self, member, multi_spec):
        if isinstance(member, tr.Collection):

            member_conditions = [
                self._construct_conditions(child, multi_spec) for child in member
            ]
            reference = member_conditions[0]

            if any(conditions != reference for conditions in member_conditions[1:]):
                raise ValueError(
                    "Nested Collection members do not share one matching identity."
                )

            return reference

        names = multi_spec.match_on or ("subject", "data_identity")
        return {
            name: self._resolve_match_value(member, name)
            for name in names
        }

    def _resolve_match_value(self, member, name):
        if name == "data_identity" and isinstance(member, tr.DataIdentity):
            return member
        return getattr(member, name, None)

    def _call_on_collection_members(self, member, collections, key_spec, multi_spec):
       
        inputs = [member]
        for collection in collections[1:]:
            reference = self._construct_conditions(member, multi_spec)
            matching_members = [
                m for m in collection if self._construct_conditions(m, multi_spec) == reference
                ]

            if len(matching_members) != 1:
                raise ValueError(
                    f"{self.format_call()}: found {len(matching_members)} for collection "
                    f"but the match policy is exactly_one."
                )
                
            else:
                inputs.extend(matching_members)

        if isinstance(member, tr.Collection):
            return self._multi_input_call_on_collection(
                *inputs, key_spec=key_spec, multi_spec=multi_spec
                )
            
        elif isinstance(member, tr.DataIdentity):
            return self._multi_input_call_on_data_identity(*inputs, key_spec=key_spec)
        
        else:
            return self._multi_input_call_on_signal(*inputs, key_spec=key_spec)

    def _call_on_data_identity(self, data_identity, key_spec=None):
        return tr.Collection(
            [
                self._call_on_signal(component.to_signal(), key_spec=key_spec)
                for component in data_identity.data_components
            ]
        )

    def _multi_input_call_on_data_identity(self, *data_identities, key_spec=None):
        if any(len(di.data_components) > 1 for di in data_identities):
            raise ValueError(
                f"{self.format_call()} mult input operations on data identities are ambiguous " \
                "when they have multiple components. Aggregate the data components within identities" \
                " first."
                )
        return tr.Collection([
            self._multi_input_call_on_signal(
                *[di.data_components[0].to_signal() for di in data_identities], key_spec=key_spec
                )
            ])

    def _call_on_signal(self, signal, key_spec=None):
        self._validate_input(signal, key_spec=key_spec)
        if isinstance(signal, tr.DataComponent):
            signal = signal.to_signal()
        output_class = self.resolve_output_class(signal)
        if isinstance(signal.data_schema, tr.DatasetSchema):
            key_spec = self.resolve_dataset_defaults(key_spec, signal, output_class)
    
        return output_class(
            inputs=(signal,),
            transformer=self,
            key_spec=key_spec,
            data_schema=None,
            transform=None
        )

    def _multi_input_call_on_signal(self, *signals, key_spec=None):
        self._validate_input(*signals, key_spec=key_spec)
        signals = [
            signal.to_signal() 
            if isinstance(signal, tr.DataComponent) 
            else signal 
            for signal in signals
            ]

        output_class = self.resolve_output_class(signals[0])
        if isinstance(signals[0].data_schema, tr.DatasetSchema):
            key_spec = self.resolve_dataset_defaults(key_spec, signals[0], output_class)

        return output_class(
            inputs=signals,
            transformer=self,
            key_spec=key_spec,
            data_schema=None,
            transform=None
        )

    def _get_apply_kwargs(self, *inputs, key_spec=None):
        return {}
    
    def _get_transform_kwargs(self, *inputs, apply_kwargs=None):
            return {}
    
    def build_transform_for(self, output_signal):
        return self._get_transform(
            *output_signal.inputs,
            key_spec=output_signal.key_spec,
            apply_kwargs=output_signal.apply_kwargs
        )

    def _get_transform(self, *inputs, key_spec=None, apply_kwargs=None):
        if apply_kwargs is None:
            apply_kwargs = self._get_apply_kwargs(*inputs, key_spec=key_spec)

        transform_kwargs = self._get_transform_kwargs(
            *inputs,
            apply_kwargs=apply_kwargs
        )

        return Transform(partial(self._apply, **apply_kwargs), **transform_kwargs)

    def resolve_dataset_defaults(self, key_spec, signal, output_class):
        input_name = key_spec.input_name
        output_mode = key_spec.output_mode or getattr(self, "key_mode", "replace")

        if input_name is None:
            dim = getattr(self, "dim", None)
            if dim:
                input_name = signal.data_schema.default_variable_for(self.dim)
                if input_name is None:
                    raise ValueError(
                        "You didn't provide a Dataset key and it can't beinferred."
                    )
        key_spec = KeySpec(input_name=input_name, output_mode=output_mode)
        return key_spec

    def resolve_output_class(self, input):
        # If we're operating on a StackedSignal, preserve the stack type.
        # If this calculator has a fixed output class, return that.
        # Otherwise ask the parent signal what it would produce.
        if getattr(input, "is_stack", False):
            return type(input)
        return self.fixed_output_class or self._infer_output_class(input)

    def _infer_output_class(self, entity):
        return getattr(entity, "output_class", type(entity))

    def _validate_arity(self, input_count):
        if (self.arity in [1, "1", "one"] and input_count != 1 or
            self.arity in [2, "2", "two"] and input_count != 2 or
            self.arity in ["one_or_more", ">=1"] and input_count < 1 or
            self.arity in ["two_or_more", ">=2"] and input_count < 2
            ):
            arity_description = {
                1: "one input",
                "1": "one input",
                "one": "one input",
                2: "two inputs",
                "2": "two inputs",
                "two": "two inputs",
                "one_or_more": "one or more inputs",
                ">=1": "one or more inputs",
                "two_or_more": "two or more inputs",
                ">=2": "two or more inputs",
            }[self.arity]
            raise ValueError(
                f"{self.format_call()}: takes {arity_description}."
            )

    def _validate_input(self, *inputs, **kwargs):

        if "key_spec" in kwargs and kwargs["key_spec"].input_name is not None:
            if not any(isinstance(input.data_schema, tr.DatasetSchema) for input in inputs):
                raise ValueError(
                    f"{self.format_call()} received key but none of the inputs are Datasets."
                )
    
    def _validate_data_schema(self, *input_schemas):
        for input_schema in input_schemas:
            if isinstance(input_schema, tr.DatasetSchema):
                data_type = xr.Dataset
            elif isinstance(input_schema, tr.Schema):
                data_type = xr.DataArray
            else:
                raise TypeError(
                    f"{self.format_call()}: expected a Schema or DatasetSchema, "
                    f"not {type(input_schema).__name__}."
                )

            if data_type not in self.accepted_data_types:
                accepted = ", ".join(
                    accepted_type.__name__
                    for accepted_type in self.accepted_data_types
                )
                raise TypeError(
                    f"{self.format_call()}: input schema describes "
                    f"{data_type.__name__} data, but this transformer accepts "
                    f"{accepted} data."
                )

    def make_output_schema(self, *input_schemas, key_spec=None, **schema_kwargs):
        """Compute the output schema."""

        key = key_spec.input_name if key_spec is not None else None

        if (
            key is not None 
            and not any(isinstance(schema, tr.DatasetSchema) for schema in input_schemas)
            ):
            raise ValueError(
                f"{self.format_call()}: key was provided but none of the inputs are Datasets."
                )

        output_mode = (
            key_spec.output_mode
            if key_spec is not None and key_spec.output_mode is not None
            else getattr(self, "key_mode", "replace")
        )

        schemas = []

        for input_schema in input_schemas:
            if isinstance(input_schema, tr.DatasetSchema) and key is not None:
                key_schema = input_schema[key]
                self._validate_data_schema(key_schema)
                schemas.append(key_schema)
            else:
                self._validate_data_schema(input_schema)
                schemas.append(input_schema)

        
        new_key_schema = self.output_schema(*schemas, **schema_kwargs)
        primary_input_schema = input_schemas[0]

        if isinstance(primary_input_schema, tr.DatasetSchema) and key is not None:
            if output_mode == "standalone":
                return new_key_schema
            elif output_mode == "rename":  
                return primary_input_schema.replace_key(self.name, new_key_schema)
            elif output_mode == "replace":
                return primary_input_schema.replace_key(key_spec.input_name, new_key_schema)
            elif output_mode == "append":
                return primary_input_schema.add_key(self.name, new_key_schema)
        else:
            return new_key_schema

    # this gets overridden
    def output_schema(self, *input_schemas, **schema_kwargs):
        return input_schemas[0].copy()
    
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

        if output_mode == "standalone":
            return result

        if output_mode == "replace":
            merged = data.copy()
            merged[key_spec.input_name] = result
            return merged
        
        if output_mode == "rename":
            merged = data.copy()
            merged[self.name] = result
            return merged

        if output_mode == "append":
            if self.name in data:
                raise ValueError(
                    f"output_mode is 'append' and key {self.name} exists in data."
                )
            merged = data.copy()

            merged[self.name] = result
            return merged

        raise ValueError("Unknown key mode")


class Calculator(Transformer):
    """A base class for most Transformers, Calculator performs data validation and 
    defines the template methods that its descendant calculators will use to generate 
    the transform, apply it to data it receives from its inputs, and wrap the result 
    return the result as an xarray DataArray or Dataset."""
    
    name = None
    key_mode = "replace"  # replace | append | standalone
    require_some_finite = True
    require_all_finite = False
    allow_empty = False

    def _validate_input(self, *inputs, **kwargs):
        super()._validate_input(*inputs, **kwargs)
        for input in inputs:
            if not isinstance(input, (tr.Signal, tr.SignalStack)):
                raise TypeError(
                    f"{self.format_call()}: calculator inputs must resolve to "
                    f"Signals, not {type(input).__name__}."
                )
    
    def _get_apply_kwargs(self, *inputs, key_spec):
        schema = inputs[0].data_schema
         
        if (
            key_spec and 
            key_spec.input_name and 
            isinstance(schema, tr.DatasetSchema)
            ):
            schema = schema[key_spec.input_name]

        result = {
            "key_spec": key_spec,
            "data_schema": schema,
            "diagnostic_context": self._build_diagnostic_context(inputs[0], key_spec),
        }
        result.update(self._get_extra_apply_kwargs(*inputs))
        return result

    @staticmethod
    def _diagnostic_identifier(entity):
        if entity is None:
            return None

        for attribute in ("display_id", "label", "id", "uid"):
            value = getattr(entity, attribute, None)
            if value is not None:
                return str(value)
        return type(entity).__name__

    def _build_diagnostic_context(self, input, key_spec):
        context = {"signal": type(input).__name__}
        entities = {
            "data identity": getattr(input, "data_identity", None),
            "origin": getattr(input, "origin", None),
            "subject": getattr(input, "subject", None),
            "session": getattr(input, "session", None),
        }
        for name, entity in entities.items():
            identifier = self._diagnostic_identifier(entity)
            if identifier is not None:
                context[name] = identifier

        conditions = getattr(input, "conditions", None)
        if conditions:
            context["conditions"] = repr(conditions)
        if key_spec is not None and key_spec.input_name is not None:
            context["key"] = key_spec.input_name

        return context

    @staticmethod
    def _format_diagnostic_context(context, data):
        details = [f"{name}={value}" for name, value in context.items()]
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            details.append(f"dimensions={dict(data.sizes)}")
        return ", ".join(details)

    def _get_transform_kwargs(self, *inputs, apply_kwargs):
        input = inputs[0]
        if getattr(input, "is_stack", None):
            signal_class = self.fixed_output_class or self._infer_output_class(
                input.signals[0]
            )
        else:
            signal_class = None

        kwargs = {"signal_class": signal_class}
        kwargs.update(self._get_extra_transform_kwargs(*inputs, apply_kwargs=apply_kwargs))

        return kwargs

    # These two methods get overridden
    def _get_extra_apply_kwargs(self, *inputs):
        return {}

    def _get_extra_transform_kwargs(self, *inputs, apply_kwargs=None):
        apply_kwargs = apply_kwargs or {}
        return deepcopy(apply_kwargs)

    def _apply(self, *input_data, key_spec=None, diagnostic_context=None, **kwargs):
       
        if key_spec is not None and key_spec.input_name is not None:
            data_for_apply = tuple(
                data if not isinstance(data, xr.Dataset) 
                else self.resolve_target_data(data, key=key_spec.input_name) 
                for data in input_data 
                )
        else:
            data_for_apply = input_data

        self._validate_data(*data_for_apply, **kwargs)

        try:
            result = self._apply_inner(*data_for_apply, **kwargs)
        except ZeroDivisionError as error:
            if diagnostic_context:
                context = self._format_diagnostic_context(
                    diagnostic_context, *data_for_apply
                )
                raise ZeroDivisionError(
                    f"{error}\nInput context: {context}"
                ) from None
            raise

        if isinstance(result, tuple):
            result, wrap_kwargs = result
        else:
            wrap_kwargs = {}

        result = self._wrap_result(result, *data_for_apply, **wrap_kwargs)
        
        if key_spec is not None and key_spec.input_name is not None:
            result = self.merge_keys(input_data[0], result, key_spec)

        return result

    def _validate_data(self, *input_data, **kwargs):

        def get_magnitude(arr):
                try:
                    mag = arr.pint.magnitude  # preferred for pint-aware arrays
                except Exception:
                    mag = arr
                return np.asarray(mag)

        for data in input_data:
            if hasattr(data, "data_vars") and len(data.data_vars) == 0:
                raise ValueError(f"{type(self)}: Event dataset has no variables.")

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

    # This method gets overridden by every Calculator.
    def _apply_inner(self, data, *args, **kwargs):
        return data

    def _wrap_result(self, result, *args):
        return result.assign_attrs({"transformer": self._calculator_name()})

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
