
from k_onda.central import type_registry


class CalculateMixin:
    def add(self, other, key=None, key_output_mode=None):
        return self.shift(other, key=key, key_output_mode=key_output_mode)

    def subtract(self, other, key=None, key_output_mode=None):
        return self.shift(-other, key=key, key_output_mode=key_output_mode)

    def multiply_by(self, other, key=None, key_output_mode=None):
        return self.scale(other, key=key, key_output_mode=key_output_mode)

    def divide_by(self, other, key=None, key_output_mode=None):
        return self.scale(1 / other, key=key, key_output_mode=key_output_mode)

    def scale(self, factor, key=None, key_output_mode=None):
        from . import Scale

        return Scale(factor)(self, key=key, key_output_mode=key_output_mode)

    def shift(self, offset, key=None, key_output_mode=None):
        from . import Shift

        return Shift(offset)(self, key=key, key_output_mode=key_output_mode)

    def reduce(self, dim, method="mean", key=None, key_output_mode=None):
        from . import ReduceDim

        return ReduceDim(dim, method)(self, key=key, key_output_mode=key_output_mode)

    def normalize(self, method="rms", dim=None, key=None, key_output_mode=None):
        from . import Normalize

        return Normalize(method, dim)(self, key=key, key_output_mode=key_output_mode)

    def median_filter(
        self,
        kernel_sizes,
        key=None,
        key_output_mode=None,
    ):
        from . import MedianFilter

        return MedianFilter(kernel_sizes)(
            self, key=key, key_output_mode=key_output_mode
        )

    def filter(self, config, key=None, key_output_mode=None):
        from . import Filter

        return Filter(config)(self, key=key, key_output_mode=key_output_mode)

    def spectrogram(self, config, key=None, key_output_mode=None):
        from . import Spectrogram

        return Spectrogram(config)(self, key=key, key_output_mode=key_output_mode)

    def threshold(self, comparison, threshold, key=None, key_output_mode=None):
        from . import Threshold

        return Threshold(comparison, threshold)(
            self, key=key, key_output_mode=key_output_mode
        )

    def apply_mask(self, mask, key=None, key_output_mode=None):
        from . import ApplyMask

        return ApplyMask(mask)(self, key=key, key_output_mode=key_output_mode)

    def fwhm(self, config=None, key=None, key_output_mode=None):
        from . import FWHM

        if config is None:
            config = {}
        return FWHM(**config)(self, key=key, key_output_mode=key_output_mode)

    def count(self, config=None, key=None, key_output_mode=None, **kwargs):
        from . import Histogram

        config = config or {} | kwargs
        if config.get("bins") is None and config.get("bin_size") is None:
            config["bins"] = 10
        return Histogram(**config)(self, key=key, key_output_mode=key_output_mode)



class IntersectionMixin:
    def intersection(self, other, tolerance_decimals=9):
        from . import Intersection

        return Intersection(tolerance_decimals)(self, other)


class PointProcessMixin:
    def rate(
        self, intervals=None, exclude_initial=None, key=None, key_output_mode=None
    ):
        from . import Rate

        return Rate(intervals=intervals, exclude_initial=exclude_initial)(
            self, key=key, key_output_mode=key_output_mode
        )


class StackMixin:
    def stack_signals(self, dim=None):
        from . import StackSignals

        return StackSignals(dim=dim)(self)


class UnstackMixin:
    def unstack_signals(self, dim=None):
        from . import UnstackSignals

        return UnstackSignals(dim=dim)(self)


class AggregateMixin:    

    def mean(self, across=None, group_by=None, preserve_groups=False, order='sequential'):
        
        planned_data_schema = self.get_planned_data_schema()

        if group_by is None:
            group_by = []
        elif isinstance(group_by, str):
            group_by = [group_by]

        if across is None:
            across = []
        elif isinstance(across, str):
            across = [across]

        # Any dims that are not in the xarray data at the time of execution must 
        # exist as metadata on the signal object. 
        # E.g. "subject", "session", "neuron_type"
        collection_coords = [dim for dim in across + group_by 
                             if not planned_data_schema.is_selectable(dim)]

        # Collect any of these dims and turn them into coords on a single long 
        # dim in the xarray data.
        if not isinstance(self, type_registry.Signal):
            signal = type_registry.AssembleArray(
                collection_coords=collection_coords,
                preserve_groups = preserve_groups, 
                planned_input_schema = planned_data_schema
                )(self)
            
            if not len(across):
                across = ["signal"]
        else:
            signal = self

        self._validate_params_on_aggregate(across, group_by, signal.data_schema)

        if order == "simultaneous": 
            raise NotImplementedError(
                    "Simultaneous averaging has not been implemented."
                )
           
        stages = self._create_stages(group_by, across, signal.data_schema)
    
        signal = self._group_and_reduce_in_stages(signal, stages)

        return signal
    
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
        
    def _group_and_reduce_in_stages(self, signal, stages):

        for stage in stages:
            if stage.get("group_by"):
                signal = type_registry.GroupBy(coords=stage["group_by"])(signal)
            signal = type_registry.ReduceDim(stage["reduce_dim"])(signal)
        
        return signal
    
    def get_planned_data_schema(self):
        planned_obj = self.planned_for_schema(self)

        if isinstance(planned_obj, type_registry.Collection):
            return planned_obj.signals[0].data_schema
        elif isinstance(planned_obj, type_registry.CollectionMap):
            return next(iter(planned_obj.values()))[0].data_schema 
        else:
            return planned_obj.data_schema

    def planned_for_schema(self, obj):
        if isinstance(obj, type_registry.Signal):
            return obj.plan_on_signal()
        if isinstance(obj, type_registry.Collection):
            return type_registry.Collection([self.planned_for_schema(member) for member in obj])
        if isinstance(obj, type_registry.CollectionMap):
            return type_registry.CollectionMap(
                groups={k: self.planned_for_schema(v) for k, v in obj.items()}
                )
        if isinstance(obj, type_registry.DataIdentity):
            return type_registry.Collection([
                self.planned_for_schema(component) for component in obj.data_components
            ])
        
   
        

    

        
