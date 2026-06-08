import xarray as xr
from operator import attrgetter
from pint_xarray import PintIndex
import numpy as np
import math

from .core import Transformer, KeySpec
from k_onda.central import type_registry, AxisInfo, AxisKind


@type_registry.register
class Aggregator(Transformer):
    def __init__(self, method="mean", group_by=None, new_dim=None):
        self.method = method
        self.group_by = group_by
        self.new_dim = new_dim
        

    def __call__(self, input, key=None, key_output_mode=None):

        if key is not None or key_output_mode is not None:
            raise NotImplementedError("Key access is not yet implemented for Aggregator")

        key_spec = KeySpec(input_name=key, output_mode=key_output_mode)

        if isinstance(input, type_registry.CollectionMap):
            if self.group_by is not None:
                raise ValueError("input of type CollectionMap is already grouped.")
            return self._call_on_collection_map(input, key_spec)

        elif isinstance(input, type_registry.Collection):
            return self._call_on_collection(input, key_spec, grouping=self.group_by)

        else:
            raise ValueError(
                "Aggregator must be called on a Collection or a Grouped_Collection "
            )

    def _call_on_collection_map(self, collection_map, key_spec):

    

        inputs = tuple(sig for key in collection_map for sig in collection_map[key])
        keys = [key for key in collection_map 
                for _ in range(len(collection_map[key]))]

        data_schema = self.output_schema(inputs[0].data_schema)
        


        return type_registry.AggregatedSignal(
            inputs = inputs,
            transformer=self,
            transform = self._get_transform(inputs),
            data_schema=data_schema,
            key_spec=key_spec,
            apply_kwargs=self._make_apply_kwargs(inputs, keys)
            
        )
        
    def _make_apply_kwargs(self, collection, data_schema, grouping):
        
        def build_grouping_func(grouping, strict=True):
            def grouping_func(entity):
                if getattr(entity, grouping, None) is not None:
                    return getattr(entity, grouping)
                elif (
                    hasattr(entity, "data_identity")
                    and getattr(entity.data_identity, "name", None) == grouping
                ):
                    return entity.data_identity

                try:
                    return attrgetter(grouping)(entity)
                except AttributeError as e:
                    if strict:
                        raise e
                    return None

            return grouping_func
        
        apply_kwargs = {
            'data_schema': data_schema
        }
        
        if grouping is None:
            return apply_kwargs
        
        # need to figure out how I reconstute neurons in the last pathway.
        # or do I?  maybe the fact that I have a collection here 
        # is evidence that I never do.
       
        grouping_func = build_grouping_func(grouping)

        group_keys = [grouping_func(sig) for sig in collection.signals]

        return apply_kwargs | {
            "group_keys": group_keys
        }
        
    def _call_on_collection(self, collection, key_spec, grouping=None):

        inputs = tuple(collection.signals)
        transform = self._get_transform(inputs)
        data_schema = self.output_schema(inputs[0].data_schema)

        return type_registry.AggregatedSignal(
            inputs=inputs,
            transformer=self,
            transform=transform,
            data_schema=data_schema,
            key_spec=key_spec,
            apply_kwargs=self._make_apply_kwargs(collection, data_schema, grouping)
        )

    def output_schema(self, input_schema):
        if self.method == "concat":
            return input_schema.with_axis(AxisInfo("members", kind=AxisKind.AXIS))
        return input_schema

    def _gather_datasets(self, data, grouping_vars=None):
        keys = data[0].keys()
        gathered_data = {}

        for key in keys:
            arrays = []
            for dataset in data:
                arr = dataset[key]
                arrays.append(arr)
        
            gathered_data[key] = self._gather_arrays(arrays, grouping_vars)

        return xr.Dataset(gathered_data)
    
    def canonicalize_arrays(self, arrs):
       
        indexes_to_drop = {coord_name: xindex for coord_name, xindex in arrs[0].xindexes.items() 
                           if isinstance(xindex, PintIndex)}        
        if indexes_to_drop:
            arrs_with_dropped_indexes = tuple(
                arr.drop_indexes(coord_name) 
                for arr in arrs
                for coord_name in indexes_to_drop
            )
        else:
            arrs_with_dropped_indexes = arrs

        arrs = [self.canonicalize_array(arr) for arr in arrs_with_dropped_indexes]
        
        return arrs, indexes_to_drop

    def canonicalize_array(self, arr):
        for coord in arr.coords:
            coord_da = arr.coords[coord]
            try:
                dtype = coord_da.pint.magnitude.dtype
            except Exception:
                dtype = coord_da.dtype

            if not np.issubdtype(dtype, np.floating):
                continue
            
            tolerance = self._get_tolerance(coord_da)
            num_decimals = -1 * math.floor(math.log10(abs(tolerance)))

            arr = arr.assign_coords({
                coord: arr.coords[coord].round(num_decimals)
            })
        return arr
    
    def _get_tolerance(self, coord_da):
        values = np.asarray(coord_da)
        values = values[np.isfinite(values)]
        values = np.sort(np.unique(values))
        min_spacing = np.min(np.diff(values))
        tolerance = min_spacing/100
        return tolerance

    def _form_groups(self, group_keys):
        unique_group_keys = list(set(group_keys))
        group_key_index = [unique_group_keys.index(key) for key in group_keys]
        labels = [getattr(key, 'label', None) for key in unique_group_keys]
        conditions = [getattr(key, 'conditions', {}) for key in unique_group_keys]

        grouping_vars = {
            'unique_group_keys': unique_group_keys,
            'group_key_index': group_key_index
        }

        return grouping_vars, labels, conditions

            
    def _gather_arrays(self, data, grouping_vars=None):

        grouping_vars = grouping_vars or {}
        unique_group_keys = grouping_vars.get('unique_group_keys')
        group_key_index = grouping_vars.get('group_key_index')

        if unique_group_keys is not None:
            gathered_arrs = [[] for _ in range(len(unique_group_keys))]
            for i, arr in enumerate(data):
                gathered_arrs[group_key_index[i]].append(arr)
        else:
            gathered_arrs = [data]

        grouped_arrs = [self.concat_arrs(arrs, dim="signals") for arrs in gathered_arrs]

        if self.method == "concat":
            result = grouped_arrs
        else:
            result = [getattr(arr, self.method)(dim="signals") 
                      for arr in grouped_arrs]
        
        return result
    
    @staticmethod
    def coord_is_float(coord):
        try:
            dtype = coord.pint.magnitude.dtype
        except Exception:
            dtype = coord.dtype

        return np.issubdtype(dtype, np.floating)

    def drop_inconsistent_coords(self, arrays, *, exclude_dims=True):
        arrays = list(arrays)
        first = arrays[0]

        coords_to_drop = []

        for name, coord in first.coords.items():

            if exclude_dims and name in first.dims:
                continue
            
            if any(name not in arr.coords for arr in arrays[1:]):
                coords_to_drop.append(name)
                continue
            
            if self.coord_is_float(coord):
                tolerance = self._get_tolerance(coord)
                if any(
                    not np.allclose(
                        np.asarray(first.coords[name]), 
                        np.asarray(arr.coords[name]), 
                        atol=tolerance
                        ) 
                    for arr in arrays[1:]
                ):
                    coords_to_drop.append(name)
                    continue
            else:
                if any(
                    not np.array_equal(
                        np.asarray(first.coords[name]),
                        np.asarray(arr.coords[name])
                        )
                    for arr in arrays[1:]
                ):
                    coords_to_drop.append(name)
                    continue

        return [arr.drop_vars(coords_to_drop, errors="ignore") for arr in arrays]


    def concat_arrs(self, arrs, dim, canonicalize=True):

        if canonicalize:
            arrs, dropped_indexes = self.canonicalize_arrays(arrs)
        else:
            dropped_indexes = {}

        arrs = self.drop_inconsistent_coords(arrs, exclude_dims=True)
      
        arr = xr.concat(arrs, dim=dim, combine_attrs="no_conflicts")
        
        for coord_name, xindex in dropped_indexes.items():
            arr = arr.set_xindex(coord_name)
            
            if hasattr(xindex, "units"):
                units = xindex.units[coord_name]
                arr = arr.pint.quantify({coord_name: units})
        
        return arr

    def _gather(self, data, grouping_vars=None):
        if isinstance(data[0], xr.Dataset):
            return self._gather_datasets(data, grouping_vars)
        return self._gather_arrays(data, grouping_vars)
    
    def _apply_metadata_coords(self, result, labels, conditions):
        grouping_dim = self.group_by or "members"
        if conditions:
            result = result.assign_coords(
                {
                    condition: (
                        grouping_dim, [condition_dict[condition] for condition_dict in conditions]
                    ) for condition in conditions[0]
                }
                )
        if labels:
            result = result.assign_coords({'label': (grouping_dim, labels)})
        
        return result


    def _apply(self, *data, group_keys=None, **kwargs):


        if group_keys:
            grouping_vars, labels, conditions = self._form_groups(group_keys)
            gathered_data = self._gather(data, grouping_vars)
        else:
            gathered_data = self._gather(data)

        if len(gathered_data) > 1:
            result = self.concat_arrs(gathered_data, dim=self.group_by or "members", canonicalize=False)
            result = self._apply_metadata_coords(result, labels, conditions)
        else:
            result = gathered_data[0]

       
        return result

