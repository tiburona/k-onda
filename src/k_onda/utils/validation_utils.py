"""Utilities shared by objects that validate their constructor arguments."""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps
import inspect
import math
from numbers import Real
import reprlib
from typing import Any, get_type_hints, Self

import numpy as np
import pint

from typeguard import (
    check_type,
    CollectionCheckStrategy,
    config as typeguard_config,
    typechecked,
    TypeCheckError,
)


_VALUE_REPR = reprlib.Repr()
_VALUE_REPR.maxstring = 120
_VALUE_REPR.maxother = 120
typeguard_config.collection_check_strategy = CollectionCheckStrategy.ALL_ITEMS


def validate_types(func):
    """Enforce the type hints on one public API method."""
    checked = typechecked(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return checked(*args, **kwargs)
        except TypeCheckError as error:
            raise TypeError(f"{func.__name__}(): {error}.") from None

    return wrapper


def validate_nonempty(call_name: str, **parameters: Any) -> None:
    """Reject supplied API parameters that are empty."""
    for name, value in parameters.items():
        if value is not None and len(value) == 0:
            raise ValueError(f"{call_name}(): {name} cannot be empty.")


def validate_string_values(
    call: str,
    parameter: str,
    values: Iterable[Any],
    *,
    item_name: str | None = None,
    nonempty: bool = True,
    unique: bool = False,
) -> None:
    """Validate the string values contained by one parameter."""
    try:
        items = tuple(values)
    except TypeError:
        raise TypeError(f"{call}: {parameter} must be iterable.") from None

    item_name = item_name or f"value in {parameter}"
    if any(not isinstance(value, str) for value in items):
        raise TypeError(f"{call}: every {item_name} must be a string.")
    if nonempty and any(not value.strip() for value in items):
        raise ValueError(f"{call}: every {item_name} must be non-empty.")
    if unique and len(set(items)) != len(items):
        raise ValueError(f"{call}: {parameter} cannot contain repeated values.")


class ValidationMixin:
    """Remember and format the arguments received by a constructor."""

    _validation_args: tuple[Any, ...]
    _validation_kwargs: dict[str, Any]

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        instance = super().__new__(cls)
        object.__setattr__(instance, "_validation_args", args)
        object.__setattr__(instance, "_validation_kwargs", kwargs)
        return instance

    def format_call(self) -> str:
        """Return this object's constructor call with named arguments."""
        signature = inspect.signature(type(self))
        bound = signature.bind(
            *self._validation_args,
            **self._validation_kwargs,
        )

        arguments: list[str] = []
        for name, value in bound.arguments.items():
            parameter = signature.parameters[name]
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                arguments.extend(
                    f"{key}={_VALUE_REPR.repr(item)}" for key, item in value.items()
                )
            else:
                arguments.append(f"{name}={_VALUE_REPR.repr(value)}")

        return f"{type(self).__name__}({', '.join(arguments)})"

    def validate_type_hints(self) -> None:
        """Validate constructor arguments against their runtime type hints."""
        signature = inspect.signature(type(self))
        bound = signature.bind(
            *self._validation_args,
            **self._validation_kwargs,
        )
        bound.apply_defaults()
        type_hints = get_type_hints(type(self).__init__)

        for name, expected_type in type_hints.items():
            if name == "return":
                continue
            value = getattr(self, name, bound.arguments.get(name))
            try:
                check_type(
                    value,
                    expected_type,
                    collection_check_strategy=CollectionCheckStrategy.ALL_ITEMS,
                )
            except TypeCheckError as error:
                raise TypeError(
                    f"{self.format_call()}: {name} {error}."
                ) from None

    def validate_parameter(
        self,
        name: str,
        value: Any,
        *,
        types: type | tuple[type, ...] | None = None,
        choices: tuple[Any, ...] = (),
        allow_none: bool = False,
        ignored_values: tuple[Any, ...] = (),
        nonempty: bool = False,
        nonempty_string: bool = False,
    ) -> None:
        """Validate one parameter against explicit runtime constraints."""
        if any(value is ignored for ignored in ignored_values):
            return
        if allow_none and value is None:
            return

        if types is None:
            expected_types = ()
        else:
            expected_types = types if isinstance(types, tuple) else (types,)
        matches_type = bool(expected_types) and isinstance(value, expected_types)
        matches_choice = any(
            type(value) is type(choice) and value == choice for choice in choices
        )

        if (expected_types or choices) and not matches_type and not matches_choice:
            requirements = []
            if expected_types:
                type_names = " or ".join(
                    expected_type.__name__ for expected_type in expected_types
                )
                requirements.append(f"type {type_names}")
            if choices:
                requirements.append(f"one of {choices!r}")
            if allow_none:
                requirements.append("None")

            error = ValueError if any(
                type(value) is type(choice) for choice in choices
            ) else TypeError
            raise error(
                f"{self.format_call()}: {name} must be {' or '.join(requirements)}."
            )

        if nonempty_string and isinstance(value, str) and not value.strip():
            raise ValueError(f"{self.format_call()}: {name} cannot be empty.")

        if nonempty and len(value) == 0:
            raise ValueError(f"{self.format_call()}: {name} cannot be empty.")

    def validate_string_iterable(
        self,
        item_name: str,
        values: Iterable[Any],
        *,
        nonempty: bool = True,
        unique: bool = False,
    ) -> None:
        """Validate that each item is a string, optionally excluding empty strings."""
        validate_string_values(
            self.format_call(),
            f"{item_name} values",
            values,
            item_name=item_name,
            nonempty=nonempty,
            unique=unique,
        )

    def validate_number(
        self,
        name: str,
        value: Any,
        *,
        number_type: type = Real,
        allow_none: bool = False,
        minimum: Real | None = None,
        maximum: Real | None = None,
        nonzero: bool = False,
        finite: bool = False,
        allow_quantity: bool = False,
    ) -> None:
        """Validate a number and optional inclusive bounds."""
        if allow_none and value is None:
            return
        is_quantity = allow_quantity and isinstance(value, pint.Quantity)
        if is_quantity:
            magnitude = np.asarray(value.magnitude)
            if magnitude.ndim != 0:
                raise ValueError(f"{self.format_call()}: {name} must be scalar.")
            value = magnitude.item()
        if isinstance(value, bool) or not isinstance(value, number_type):
            requirement = f"a non-boolean {number_type.__name__}"
            if is_quantity:
                requirement = (
                    f"a scalar Quantity with a {number_type.__name__} magnitude"
                )
            raise TypeError(
                f"{self.format_call()}: {name} must be {requirement}."
            )
        if nonzero and value == 0:
            raise ValueError(f"{self.format_call()}: {name} cannot be zero.")
        if finite and not math.isfinite(value):
            raise ValueError(f"{self.format_call()}: {name} must be finite.")
        if minimum is not None and value < minimum:
            raise ValueError(
                f"{self.format_call()}: {name} must be at least {minimum!r}."
            )
        if maximum is not None and value > maximum:
            raise ValueError(
                f"{self.format_call()}: {name} must be at most {maximum!r}."
            )
