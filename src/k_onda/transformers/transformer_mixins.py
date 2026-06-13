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
        
        if isinstance(self, type_registry.Collection):
            data_schema = self.signals[0].data_schema
        elif isinstance(self, type_registry.CollectionMap):
            data_schema = next(iter(self.values()))[0].data_schema
        else:
            data_schema = self.data_schema

        if group_by is None:
            group_by = []
        elif isinstance(group_by, str):
            group_by = [group_by]

        if across is None:
            across = []
        elif isinstance(across, str):
            across = [across]

        collection_dims = [dim for dim in across + group_by 
                           if dim not in data_schema.dim_names]

        if not isinstance(self, type_registry.Signal):
            signal = type_registry.Aggregator(
                group_by=collection_dims,
                preserve_groups = preserve_groups
                )(self)
            
            if not len(across):
                across = ["signals"]
        else:
            signal = self

        return self._reduce_dims_in_data(signal, across, group_by, signal.data_schema, order)


    def _reduce_dims_in_data(self, signal, across, group_by, data_schema, order):
        grouping_coords_in_data = [coord for coord in group_by if data_schema.coord_names]

        reduced_dims_in_data = [dim for dim in across if dim in data_schema.dim_names]

        if group_by:
            signal = type_registry.GroupBy(coords=grouping_coords_in_data)(signal)

        if order == 'simultaneous':
            signal = type_registry.ReduceDim(dim=reduced_dims_in_data)(signal)
        else:
            for dim in reduced_dims_in_data:
                signal = type_registry.ReduceDim(dim=dim)(signal)

        return signal
        
       

        


       


        
        
        
    

        
