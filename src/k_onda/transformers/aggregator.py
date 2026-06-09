import xarray as xr
from operator import attrgetter
from pint_xarray import PintIndex
import numpy as np
import math

from .core import Transformer, KeySpec
from k_onda.central import type_registry, AxisInfo, AxisKind


@type_registry.register
class Aggregator(Transformer):
    def __init__(
            self, 
            method="mean", 
            group_by=None, 
            new_dim=None, 
            preserve_groups=False, 
            group_dim_name=None
            ):
        self.method = method
        self.group_by = group_by
        self.new_dim = new_dim
        self.preserve_groups = preserve_groups
        self.group_dim_name = group_dim_name
        
    def __call__(self, input, key=None, key_output_mode=None):

        if key is not None or key_output_mode is not None:
            raise NotImplementedError("Key access is not yet implemented for Aggregator")

        key_spec = KeySpec(input_name=key, output_mode=key_output_mode)

        if isinstance(input, type_registry.CollectionMap):
            return self._call_on_collection_map(input, key_spec)

        elif isinstance(input, type_registry.Collection):
            return self._call_on_collection(input, key_spec)

        else:
            raise ValueError(
                "Aggregator must be called on a Collection or a Grouped_Collection "
            )
        
    def _call_on_collection_map(self, collection_map, key_spec):

        if not self.preserve_groups and self.group_by:
            raise ValueError("A CollectionMap is already grouped, and you have passed `group_by`." \
            " If you want to group within the CollectionMap's collections, use preserve_groups=True.")

        inputs = tuple(sig for key in collection_map for sig in collection_map[key])

        if self.preserve_groups:
            group_on = getattr(collection_map, "group_on", None)
            return type_registry.CollectionMap(
                groups={
                    k: self._call_on_collection(v, key_spec)
                    for k, v in collection_map.items()
                },
                group_on=group_on
            )
        
        # you need a key for every signal in the collection map, not just every collection
        group_keys = [key for key in collection_map for _ in range(len(collection_map[key]))]
        
        if self.group_dim_name:
            group_dim = self.group_dim_name
        elif isinstance(collection_map.group_on, str):
            group_dim = collection_map.group_on
        else:
            group_dim = "members"

        data_schema = self._make_aggregate_schema(inputs[0].data_schema, group_dim=group_dim)

        apply_kwargs = {
            "data_schema": data_schema,
            "group_dim": group_dim,
            "group_keys": group_keys
        }

        return type_registry.AggregatedSignal(
            inputs = inputs,
            transformer=self,
            transform = self._get_transform(inputs),
            data_schema=data_schema,
            key_spec=key_spec,
            apply_kwargs=apply_kwargs
        )
        
    def _call_on_collection(self, collection, key_spec):

        inputs = tuple(collection.signals)
        transform = self._get_transform(inputs)

        if self.group_by:
            if callable(self.group_by):
                grouping_func = self.group_by
            else:
                grouping_func = self.build_grouping_func(self.group_by)
            group_keys = [grouping_func(sig) for sig in collection.signals]
        else:
            group_keys = None

        if (group_keys is None and self.method != "concat"):
            group_dim = None
        elif self.group_dim_name:
            group_dim = self.group_dim_name
        elif isinstance(self.group_by, str):
            group_dim = self.group_by
        else:
            group_dim = "members"

        data_schema = self._make_aggregate_schema(inputs[0].data_schema, group_dim=group_dim)

        apply_kwargs = {
            "data_schema": data_schema,
            "group_dim": group_dim,
            "group_keys": group_keys
        }

        return type_registry.AggregatedSignal(
            inputs=inputs,
            transformer=self,
            transform=transform,
            data_schema=data_schema,
            key_spec=key_spec,
            apply_kwargs=apply_kwargs
        )
    
    @staticmethod
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

    def _make_aggregate_schema(self, input_schema, *, group_dim=None):
        schema = input_schema
        if group_dim is not None:
            schema = schema.with_axis(AxisInfo(group_dim, kind=AxisKind.AXIS))
        if self.method == "concat":
            schema = schema.with_axis(AxisInfo("signals", kind=AxisKind.AXIS))
        return schema

    def canonicalize_arrays(self, arrs):
       
        indexes_to_drop = {coord_name: xindex for coord_name, xindex in arrs[0].xindexes.items() 
                           if isinstance(xindex, PintIndex)}  
        if indexes_to_drop:
            for coord_name in indexes_to_drop:
                arrs = tuple(arr.drop_indexes(coord_name) for arr in arrs)
        arrs = [self.canonicalize_array(arr) for arr in arrs]
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
            if tolerance is None or not np.isfinite(tolerance) or tolerance <= 0:
                continue
            num_decimals = -1 * math.floor(math.log10(abs(tolerance)))

            arr = arr.assign_coords({
                coord: arr.coords[coord].round(num_decimals)
            })
        return arr
    
    def _get_tolerance(self, coord_da):
        values = np.asarray(coord_da)
        values = values[np.isfinite(values)]
        values = np.sort(np.unique(values))

        if len(values) < 2:
            return None
        
        return np.min(np.diff(values)) / 100

    def _form_groups(self, group_keys):
        unique_group_keys = list(dict.fromkeys(group_keys))
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

        all_coord_names = list(dict.fromkeys(
            coord
            for arr in arrays
            for coord in arr.coords
        ))

        for name in all_coord_names:
            first = [arr for arr in arrays if name in arr.coords][0]
            rest = [arr for arr in arrays if arr is not first]

            coord = first.coords[name]
            
            if exclude_dims and name in first.dims:
                continue
            
            if any(name not in arr.coords for arr in rest):
                coords_to_drop.append(name)
                continue

            if any(arr.coords[name].dims != first.coords[name].dims 
                   for arr in rest
                   ):
                coords_to_drop.append(name)
                continue
            
            if self.coord_is_float(coord):
                tolerance = self._get_tolerance(coord) 
                if tolerance is None:
                    equal = all(np.array_equal(
                        np.asarray(first.coords[name]), 
                        np.asarray(arr.coords[name])
                    )
                    for arr in rest)
                else:
                    equal = all(np.allclose(
                        np.asarray(first.coords[name]), 
                        np.asarray(arr.coords[name]),
                        atol=tolerance,
                        rtol=0.0
                    )
                    for arr in rest)
                if not equal:
                    coords_to_drop.append(name)
                    continue

            else:
                if any(
                    not np.array_equal(
                        np.asarray(first.coords[name]),
                        np.asarray(arr.coords[name])
                        )
                    for arr in rest
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
      
        arr = xr.concat(arrs, dim=dim, combine_attrs="no_conflicts", coords="minimal")
        
        for coord_name, xindex in dropped_indexes.items():
            arr = arr.set_xindex(coord_name)
            
            if hasattr(xindex, "units"):
                units = xindex.units[coord_name]
                arr = arr.pint.quantify({coord_name: units})
        
        return arr
    
    def _apply_metadata_coords(self, result, labels, conditions, group_dim):
     
        condition_names = list(dict.fromkeys(
            condition
            for condition_dict in conditions
            for condition in condition_dict
        ))
        if conditions:
            result = result.assign_coords(
                {
                    condition: (
                        group_dim, [condition_dict.get(condition) for condition_dict in conditions]
                    ) for condition in condition_names
                }
                )
        if labels:
            result = result.assign_coords({'label': (group_dim, labels)})
        
        return result
    
    def _apply_to_arrays(self, data, group_keys, group_dim):
        if group_keys:
            grouping_vars, labels, conditions = self._form_groups(group_keys)
            gathered_data = self._gather_arrays(data, grouping_vars)
        else:
            gathered_data = self._gather_arrays(data)

        if len(gathered_data) > 1:
            result = self.concat_arrs(gathered_data, dim=group_dim, canonicalize=False)
            result = self._apply_metadata_coords(result, labels, conditions, group_dim)
        else:
            result = gathered_data[0]

        return result

    def _apply(self, *data, group_keys=None, group_dim=None, **kwargs):

        if isinstance(data[0], xr.DataArray):
            return self._apply_to_arrays(data, group_keys, group_dim)
    
        keys = data[0].keys()
        gathered_data = {}

        for key in keys:
            arrays = []
            for dataset in data:
                arr = dataset[key]
                arrays.append(arr)
        
            gathered_data[key] = self._apply_to_arrays(arrays, group_keys, group_dim)
        
        return xr.Dataset(gathered_data)
       

