from typing import Protocol, runtime_checkable


class TypeRegistry:
    def __init__(self):
        self._classes = {}

    def register(self, cls):
        self._classes[cls.__name__] = cls
        return cls

    def __getattr__(self, name):
        try:
            return self._classes[name]
        except KeyError:
            raise AttributeError(f"No signal class '{name}' registered")


type_registry = TypeRegistry()


@runtime_checkable
class SignalLike(Protocol):
    data: ...
