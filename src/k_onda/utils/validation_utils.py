"""Utilities shared by objects that validate their constructor arguments."""

from __future__ import annotations

import inspect
import reprlib
from typing import Any, Self


_VALUE_REPR = reprlib.Repr()
_VALUE_REPR.maxstring = 120
_VALUE_REPR.maxother = 120


class ValidationMixin:
    """Remember and format the arguments received by a constructor."""

    _validation_args: tuple[Any, ...]
    _validation_kwargs: dict[str, Any]

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        instance = super().__new__(cls)
        instance._validation_args = args
        instance._validation_kwargs = kwargs
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
