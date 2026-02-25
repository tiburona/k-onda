from copy import deepcopy
from functools import partial
import xarray as xr

from .core import Transformer


class StackSignals(Transformer):
    """Concatenate component signals so downstream calculations can be vectorized."""

    def __init__(self, dim=None):
        self.dim = dim

    def __call__(self, parent):
        from ..signals import SignalStack

        return SignalStack(parent=parent, transform=self._apply, calculator=self)

    def _gather_datasets(self, signals):
        keys = signals[0].data.keys()
        data = {}
        boundaries = [0]

        for i, key in enumerate(keys):
            arrays = []
            for signal in signals:
                arr = signal.data[key]
                arrays.append(arr)
                increment = arr.sizes[self.dim] if self.dim else 1
                if i == 0:
                    boundaries.append(boundaries[-1] + increment)

            data[key] = xr.concat(
                arrays, dim=self.dim or "members", combine_attrs="no_conflicts"
            )

        dataset = xr.Dataset(data)
        dataset.attrs["boundaries"] = boundaries
        dataset.attrs["stack_dim"] = self.dim

        return dataset

    def _gather_arrays(self, signals):
        arrays = []
        boundaries = [0]

        for signal in signals:
            arr = signal.data
            arrays.append(arr)
            increment = arr.sizes[self.dim] if self.dim else 1
            boundaries.append(boundaries[-1] + increment)

        data = xr.concat(arrays, dim=self.dim or "members", combine_attrs="no_conflicts")

        data.attrs["boundaries"] = boundaries
        data.attrs["stack_dim"] = self.dim

        return data

    def _apply(self, signals):
        if isinstance(signals[0].data, xr.Dataset):
            return self._gather_datasets(signals)
        return self._gather_arrays(signals)


class UnstackSignals(Transformer):
    def __init__(self, dim=None):
        self.dim = dim

    def get_child_class(self):
        from ..sources import Collection
        return Collection

    def __call__(self, signal_stack):
        signals = []

        for i in range(len(signal_stack.signals)):
            signal_class = signal_stack.transform.signal_class or signal_stack.signal_class
            transform = partial(self._apply, idx=i)
            origin = signal_stack.signals[i].origin
            signal = signal_class(
                parent=signal_stack,
                transform=transform,
                origin=origin,
                calculator=self,
            )
            signals.append(signal)

        return self.get_child_class()(signals)

    def get_child_signal_class(self, signal):
        if not hasattr(signal, "calculator"):
            return signal.output_class
        return signal.calculator.get_child_class()

    def _apply(self, data, idx):
        attrs = deepcopy(data.attrs)
        boundaries = attrs.pop("boundaries")
        stack_dim = attrs.pop("stack_dim")
      
        dim = self.dim or stack_dim
        start, end = boundaries[idx], boundaries[idx + 1]
        selected_data = data.isel({dim: slice(start, end)})
        selected_data.attrs = attrs
        return selected_data
    