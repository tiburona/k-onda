from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from numbers import Real
from typing import Literal

import numpy as np
import pint

from k_onda.central import type_registry as tr
from k_onda.utils import validate_string_values, validate_types


MAX_CONCRETE_OPERAND_VALUES = 1_000



def _validate_concrete_operand_size(input: np.ndarray | pint.Quantity) -> None:
    size = (
        input.size
        if isinstance(input, np.ndarray)
        else np.array(input.magnitude).size
    )
    if size > MAX_CONCRETE_OPERAND_VALUES:
        raise ValueError(
            "Concrete arithmetic operands may contain at most "
            f"{MAX_CONCRETE_OPERAND_VALUES} values; received "
            f"{type(input).__name__} with {size} values. Convert the data to a "
            "Signal and pass it as a signal operand instead."
        )


def _operand_kind(
    input: object,
) -> Literal["signal_operand", "concrete_operand"]:
    if not isinstance(input, bool) and isinstance(input, (int, float, np.number)):
        return "concrete_operand"

    if isinstance(input, (np.ndarray, pint.Quantity)):
        _validate_concrete_operand_size(input)
        return "concrete_operand"

    signal_operand_types = []
    for name in (
        "Signal",
        "DataIdentity",
        "Collection",
        "CollectionMap",
        "SignalStack",
    ):
        operand_type = getattr(tr, name, None)
        if operand_type is not None:
            signal_operand_types.append(operand_type)

    if isinstance(input, tuple(signal_operand_types)):
        return "signal_operand"

    raise TypeError(
        "Arithmetic operands must be a Signal, SignalStack, Collection, "
        "CollectionMap, DataIdentity, or supported concrete numerical value; "
        f"received {type(input).__name__}."
    )


class CalculateMixin:

    def __add__(self, other: object):
        return self.add(other)

    def __sub__(self, other: object):
        return self.subtract(other)

    def __mul__(self, other: object):
        return self.multiply_by(other)

    def __truediv__(self, other: object):
        return self.divide_by(other)

    def _arithmetic(
        self,
        calculator_class,
        other,
        *others,
        alignment="exact",
        key=None,
        key_output_mode=None,
        match_on=None,
        multi_input_collection_policy="exactly_one",
        multi_input_map_policy="exact_keys",
    ):
        if isinstance(match_on, str):
            match_on = (match_on,)

        operands = (other, *others)

        if all(_operand_kind(operand) == "signal_operand" for operand in operands):
            calculator = calculator_class(alignment=alignment)
            inputs = (self, *operands)

        elif all(_operand_kind(operand) == "concrete_operand" for operand in operands):
            if len(operands) > 1:
                raise ValueError("Received multiple concrete operands; you can only pass one.")

            calculator = calculator_class(operand=other, alignment=alignment)
            inputs = (self,)

        else:
            raise ValueError("Signal operands and concrete operands cannot be mixed.")

        return calculator(
            *inputs,
            key=key,
            key_output_mode=key_output_mode,
            match_on=match_on,
            multi_input_collection_policy=multi_input_collection_policy,
            multi_input_map_policy=multi_input_map_policy
        )

    @validate_types
    def add(
        self,
        other: object,
        *others: object,
        alignment: str = "exact",
        key: str | None = None,
        key_output_mode: str | None = None,
        match_on: str | Iterable[str] | None = None,
        multi_input_collection_policy: str = "exactly_one",
        multi_input_map_policy: str = "exact_keys",
    ):

        from . import Add

        return self._arithmetic(
            Add,
            other,
            *others,
            alignment=alignment,
            key=key,
            key_output_mode=key_output_mode,
            match_on=match_on,
            multi_input_collection_policy=multi_input_collection_policy,
            multi_input_map_policy=multi_input_map_policy,
        )

    @validate_types
    def subtract(
        self,
        other: object,
        *others: object,
        alignment: str = "exact",
        key: str | None = None,
        key_output_mode: str | None = None,
        match_on: str | Iterable[str] | None = None,
        multi_input_collection_policy: str = "exactly_one",
        multi_input_map_policy: str = "exact_keys",
        ):
    
        from . import Subtract

        return self._arithmetic(
            Subtract,
            other,
            *others,
            alignment=alignment,
            key=key,
            key_output_mode=key_output_mode,
            match_on=match_on,
            multi_input_collection_policy=multi_input_collection_policy,
            multi_input_map_policy=multi_input_map_policy,
        )

    @validate_types
    def multiply_by(
        self,
        other: object,
        *others: object,
        alignment: str = "exact",
        key: str | None = None,
        key_output_mode: str | None = None,
        match_on: str | Iterable[str] | None = None,
        multi_input_collection_policy: str = "exactly_one",
        multi_input_map_policy: str = "exact_keys",
        ):
    
        from . import Multiply

        return self._arithmetic(
            Multiply,
            other,
            *others,
            alignment=alignment,
            key=key,
            key_output_mode=key_output_mode,
            match_on=match_on,
            multi_input_collection_policy=multi_input_collection_policy,
            multi_input_map_policy=multi_input_map_policy,
        )

    @validate_types
    def divide_by(
        self,
        other: object,
        *others: object,
        alignment: str = "exact",
        key: str | None = None,
        key_output_mode: str | None = None,
        match_on: str | Iterable[str] | None = None,
        multi_input_collection_policy: str = "exactly_one",
        multi_input_map_policy: str = "exact_keys",
        ):
    
        from . import Divide

        return self._arithmetic(
            Divide,
            other,
            *others,
            alignment=alignment,
            key=key,
            key_output_mode=key_output_mode,
            match_on=match_on,
            multi_input_collection_policy=multi_input_collection_policy,
            multi_input_map_policy=multi_input_map_policy,
        )

    @validate_types
    def reduce(
        self,
        dim: str | Iterable[str] | None = None,
        *,
        method: str = "mean",
        key: str | None = None,
        key_output_mode: str | None = None,
    ):
        from . import ReduceDim

        return ReduceDim(dim, method=method)(
            self,
            key=key,
            key_output_mode=key_output_mode,
        )

    @validate_types
    def mean(
        self,
        dim: str | Iterable[str] | None = None,
        *,
        key: str | None = None,
        key_output_mode: str | None = None,
    ):
        from . import ReduceDim

        return ReduceDim(dim, method="mean")(
            self, key=key, key_output_mode=key_output_mode
        )

    @validate_types
    def std(
        self,
        dim: str | Iterable[str] | None = None,
        *,
        key: str | None = None,
        key_output_mode: str | None = None,
    ):
        from . import ReduceDim

        return ReduceDim(dim, method="std")(
            self, key=key, key_output_mode=key_output_mode
        )

    @validate_types
    def normalize(
        self,
        method: str = "rms",
        *,
        dim: str | Iterable[str] | None = None,
        key: str | None = None,
        key_output_mode: str | None = None,
    ):
        from . import Normalize

        return Normalize(method, dim=dim)(
            self, key=key, key_output_mode=key_output_mode
        )

    @validate_types
    def median_filter(
        self,
        kernel_sizes: Mapping[str, int],
        *,
        key: str | None = None,
        key_output_mode: str | None = None,
    ):
        from . import MedianFilter

        return MedianFilter(kernel_sizes)(
            self, key=key, key_output_mode=key_output_mode
        )

    @validate_types
    def filter(self, method: str, **kwargs: object):
        methods = {
            "iir_notch": self.iir_notch,
            "median": self.median_filter,
        }
        try:
            filter_method = methods[method]
        except KeyError:
            known_methods = ", ".join(sorted(methods))
            raise ValueError(
                f"Unknown filter method {method!r}. Available methods: {known_methods}."
            ) from None

        return filter_method(**kwargs)

    @validate_types
    def iir_notch(
        self,
        *,
        f_lo: Real,
        f_hi: Real,
        notch_Q: Real | None = None,
        dim: str = "time",
        key: str | None = None,
        key_output_mode: str | None = None,
    ):
        from . import Filter

        return Filter(
            "iir_notch",
            dim=dim,
            f_lo=f_lo,
            f_hi=f_hi,
            notch_Q=notch_Q,
        )(
            self, key=key, key_output_mode=key_output_mode
        )

    @validate_types
    def spectrogram(self, method: str, **kwargs: object):
        methods = {"multitaper": self.multitaper_spectrogram}
        try:
            spectrogram_method = methods[method]
        except KeyError:
            known_methods = ", ".join(sorted(methods))
            raise ValueError(
                f"Unknown spectrogram method {method!r}. "
                f"Available methods: {known_methods}."
            ) from None

        return spectrogram_method(**kwargs)

    @validate_types
    def multitaper_spectrogram(
        self,
        *,
        freqs: Iterable[Real] | np.ndarray,
        decim: int,
        n_cycles: Real | Iterable[Real] | np.ndarray,
        time_bandwidth: Real,
        output: str = "power",
        key: str | None = None,
        key_output_mode: str | None = None,
    ):
        from . import Spectrogram

        return Spectrogram(
            "multitaper",
            freqs=freqs,
            decim=decim,
            n_cycles=n_cycles,
            time_bandwidth=time_bandwidth,
            output=output,
        )(self, key=key, key_output_mode=key_output_mode)

    @validate_types
    def threshold(
        self,
        comparison: str,
        threshold: Real | pint.Quantity,
        *,
        key: str | None = None,
        key_output_mode: str | None = None,
    ):
        from . import Threshold

        return Threshold(comparison, threshold)(
            self, key=key, key_output_mode=key_output_mode
        )

    @validate_types
    def apply_mask(
        self,
        mask: object,
        *,
        tolerance_decimals: int = 9,
        key: str | None = None,
        key_output_mode: str | None = None,
    ):
        from . import ApplyMask

        return ApplyMask(tolerance_decimals=tolerance_decimals)(
            self, 
            mask, 
            key=key, 
            key_output_mode=key_output_mode
        )

    @validate_types
    def fwhm(
        self,
        *,
        dim: str = "sample",
        include_valleys: bool = True,
        peak_selection: str = "prominence",
        key: str | None = None,
        key_output_mode: str | None = None,
    ):
        from . import FWHM

        return FWHM(
            dim=dim,
            include_valleys=include_valleys,
            peak_selection=peak_selection,
        )(self, key=key, key_output_mode=key_output_mode)

    @validate_types
    def count(
        self,
        *,
        bins: (
            int
            | Sequence[Real | pint.Quantity]
            | np.ndarray
            | pint.Quantity
            | Callable[..., object]
            | None
        ) = None,
        bin_size: Real | pint.Quantity | Callable[..., object] | None = None,
        hist_range: (
            Sequence[Real | pint.Quantity]
            | np.ndarray
            | pint.Quantity
            | Callable[..., object]
            | None
        ) = None,
        stat: str = "count",
        density: bool = False,
        dim: str = "time",
        range_source: str = "data",
        bin_coord: str = "left",
        key: str | None = None,
        key_output_mode: str | None = None,
    ):
        from . import Histogram

        return Histogram(
            bins=bins,
            bin_size=bin_size,
            hist_range=hist_range,
            stat=stat,
            density=density,
            dim=dim,
            range_source=range_source,
            bin_coord=bin_coord,
        )(self, key=key, key_output_mode=key_output_mode)



class IntersectionMixin:
    @validate_types
    def intersection(
        self,
        *others: object,
        tolerance_decimals: int = 9,
    ):
        from . import Intersection

        return Intersection(tolerance_decimals=tolerance_decimals)(self, *others)


class PointProcessMixin:
    @validate_types
    def rate(
        self,
        *,
        key: str | None = None,
        key_output_mode: str | None = None,
    ):
        from . import Rate

        return Rate()(self, key=key, key_output_mode=key_output_mode)


class StackMixin:
    @validate_types
    def stack_signals(self, dim: str | None = None):
        from . import StackSignals

        return StackSignals(dim=dim)(self)


class UnstackMixin:
    def unstack_signals(self):
        from . import UnstackSignals

        return UnstackSignals()(self)


class AggregateMixin:    

    @validate_types
    def aggregate(
        self,
        across: str | Iterable[str] = "signal",
        *,
        method: str = "mean",
        group_by: str | Iterable[str] | None = None,
        preserve_groups: bool = False,
        order: str = "sequential",
    ):
        if order not in {"sequential", "simultaneous"}:
            raise ValueError(
                "mean() order must be 'sequential' or 'simultaneous'."
            )
        if order == "simultaneous":
            raise NotImplementedError(
                "Simultaneous averaging has not been implemented."
            )

        across = self._normalize_aggregate_names("across", across)
        group_by = self._normalize_aggregate_names("group_by", group_by)

        planned_data_schema = self.get_planned_data_schema()

        # Any dims that are not in the xarray data at the time of execution must 
        # exist as metadata on the signal object. 
        # E.g. "subject", "session", "neuron_type"
        collection_coords = [dim for dim in across + group_by 
                             if not planned_data_schema.is_selectable(dim)]

        # Collect any of these dims and turn them into coords on a single long 
        # dim in the xarray data.
        if not isinstance(self, tr.Signal):
            signal = tr.AssembleArray(
                collection_coords=collection_coords,
                preserve_groups = preserve_groups, 
                )(self, planned_input_schema=planned_data_schema)
            
            if not across:
                across = ["signal"]
        else:
            signal = self

        self._validate_params_on_aggregate(across, group_by, signal.data_schema)

        stages = self._create_stages(group_by, across, signal.data_schema)
    
        signal = self._group_and_reduce_in_stages(signal, stages, method=method)

        return signal

    def _normalize_aggregate_names(self, parameter, value):
        if value is None:
            return []
        names = [value] if isinstance(value, str) else list(value)
        validate_string_values(
            "mean()",
            parameter,
            names,
            unique=True,
        )
        return names
    
    def _validate_params_on_aggregate(self, across, group_by, data_schema):
        for coord in group_by:
            if coord not in data_schema.collectable_coords:
                raise ValueError(f"Coord {coord} not found in data schema")
        for dim in across:
            if not data_schema.is_selectable(dim):
                raise ValueError(f"{dim} not found in data schema.")

    def _find_long_across_dims(self, across, group_by, data_schema):

        if data_schema.observation_axis:
            long_dim = data_schema.observation_axis.name
            long_across = [
                dim for dim in across 
                if dim not in data_schema.dim_names
                if dim in data_schema.coord_names_by_dim(long_dim)
                ]
            long_groupby = [
                coord for coord in group_by
                if data_schema.axis_by_coord_name(coord).name == long_dim
            ]
        else:
            long_dim = None
            long_across = []
            long_groupby = []

        return long_dim, long_across, long_groupby

    def _create_stages(self, group_by, across, data_schema):
        long_dim, long_across, long_groupby = self._find_long_across_dims(across, group_by, data_schema)
        ax_coord_dict = data_schema.ax_coord_map(group_by)

        dims_to_reduce = list(across)
        if long_dim:
            # Insert long dim where it goes in the order of concrete dims to reduce
            for i, dim in enumerate(across):
                if dim in long_across:
                    dims_to_reduce.insert(i, long_dim)
                    break

        stages = []
        for dim in dims_to_reduce:
            stage = {"reduce_dim": dim}
            if dim == long_dim:
                stage["group_by"] = list(dict.fromkeys(long_groupby + long_across))
            elif dim in long_across:
                stage["group_by"] = []
            elif ax_coord_dict.get(dim):
                stage["group_by"] = [coord for coord in group_by if coord in ax_coord_dict.get(dim)]
            else:
                stage["group_by"] = []
            stages.append(stage)

        return stages
        
    def _group_and_reduce_in_stages(self, signal, stages, method="mean"):

        for stage in stages:
            if stage.get("group_by"):
                signal = tr.GroupBy(coords=stage["group_by"])(signal)
            signal = tr.ReduceDim(stage["reduce_dim"], method=method)(signal)
        
        return signal
    
    def get_planned_data_schema(self):

        from .selector.planner import PlanSelection
        
        planned_obj = PlanSelection()(self)

        if isinstance(planned_obj, tr.Collection):
            return planned_obj.signals[0].data_schema
        elif isinstance(planned_obj, tr.CollectionMap):
            return next(iter(planned_obj.values()))[0].data_schema 
        else:
            return planned_obj.data_schema

        
   
        

    

        
