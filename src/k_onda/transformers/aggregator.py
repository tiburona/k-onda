import xarray as xr
from pint_xarray import PintIndex
import numpy as np
import math

from .core import Transformer, KeySpec
from k_onda.central import type_registry, AxisInfo, AxisKind, CoordInfo


@type_registry.register
class Aggregator(Transformer):
    def __init__(self, collection_coords=None, preserve_groups=False, planned_input_schema=None):
        self.collection_coords = collection_coords or []
        if isinstance(self.collection_coords, str):
            self.collection_coords = [self.collection_coords]
        self.preserve_groups = preserve_groups
        self.planned_input_schema = planned_input_schema
        
    def __call__(self, input, key=None, key_output_mode=None):

        if key is not None or key_output_mode is not None:
            raise NotImplementedError("Key access is not yet implemented for Aggregator")

        key_spec = KeySpec(input_name=key, output_mode=key_output_mode)

        if isinstance(input, type_registry.CollectionMap):
            return self._call_on_collection_map(input, key_spec=key_spec)

        elif isinstance(input, type_registry.Collection):
            return self._call_on_collection(input, key_spec=key_spec)

        else:
            raise ValueError(
                "Aggregator must be called on a Collection or a Grouped_Collection "
            )
        
    def _call_on_collection_map(self, collection_map, key_spec):

        if self.preserve_groups:
            group_on = getattr(collection_map, "group_on", None)
            return type_registry.CollectionMap(
                groups={
                    k: self._call_on_collection(v, key_spec=key_spec)
                    for k, v in collection_map.items()
                },
                group_on=group_on
            )
        
        return self._call_on_collection(collection_map.as_collection(), key_spec=key_spec)
        
    def _call_on_collection(self, collection, key_spec=None):

        inputs = tuple(collection.signals)
        
        if self.collection_coords: 
            if callable(self.collection_coords):
                grouping_func = self.collection_coords
            else:
                grouping_func = self.build_grouping_func(self.collection_coords)
            labels_and_factors = [grouping_func(sig) for sig in collection.signals]
        else:
            labels_and_factors = []
             
      
        input_schema = self.planned_input_schema or inputs[0].data_schema
        data_schema = self.make_output_schema(input_schema, key_spec=key_spec)

        apply_kwargs = {
            "data_schema": data_schema,
            "labels_and_factors": labels_and_factors
        }

        return type_registry.AggregatedSignal(
            inputs=inputs,
            transformer=self,
            transform=None,
            data_schema=data_schema,
            key_spec=key_spec,
            apply_kwargs=apply_kwargs
        )
    
    @staticmethod
    def build_grouping_func(groupings, strict=False):

        def grouping_func(signal):
            result = {}
            for name in groupings:
                value =  Aggregator.resolve_grouping_value(signal, name, strict=strict)
                if value is not None:
                    result[name] = value

            return result
            
        return grouping_func
    
    @staticmethod
    def resolve_grouping_value(signal, name, strict=True):
        context = (
            getattr(signal, "data_identity", None),
            getattr(signal, "subject", None),
            getattr(signal, "session", None),
        )
        
        for obj in context:
            if obj is None:
                continue
            
            #  entity name: "neuron", "subject", "session"
            if getattr(obj, "name", None) == name:
                return getattr(obj, "label", obj)
            
            # factor name: "neuron_type", "treatment", etc.
            factors = getattr(obj, "factors", {}) or {}
            if name in factors:
                return factors[name]
            
        if strict:
            raise AttributeError(f"Could not resolve group_by={name!r} for {signal!r}")

    def output_schema(self, input_schema): 
        input_schema = self.planned_input_schema or input_schema
        schema = input_schema.with_axis(AxisInfo(
            "signal", 
            kind=AxisKind.OBSERVATION_INDEX,
            coords=tuple(CoordInfo(group) for group in self.collection_coords)))
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

    def _concat_arrs(self, arrs, dim, canonicalize=True):

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
    
    def _apply_metadata_coords(self, result, factors, group_dim="signal"):
     
        factor_names = list(dict.fromkeys(
            factor
            for factor_dict in factors
            for factor in factor_dict
        ))
        if factors:
            result = result.assign_coords(
                {
                    factor: (
                        group_dim, [factor_dict.get(factor) for factor_dict in factors]
                    ) for factor in factor_names
                }
                )
        
        return result
        
    def _apply_to_arrays(self, arrs, labels_and_factors, group_dim):
        group_dim = group_dim or "signal"
        result = self._concat_arrs(arrs, dim="signal")

        if not labels_and_factors:
            return result
        
        labels_and_factors = list(labels_and_factors)
        result = self._apply_metadata_coords(result, labels_and_factors, group_dim)
        return result

    def _apply(self, *data, labels_and_factors=None, group_dim=None, **kwargs):

        if isinstance(data[0], xr.DataArray):
            return self._apply_to_arrays(data, labels_and_factors, group_dim)
    
        keys = data[0].keys()
        gathered_data = {}

        for key in keys:
            arrays = []
            for dataset in data:
                arr = dataset[key]
                arrays.append(arr)
        
            gathered_data[key] = self._apply_to_arrays(arrays, labels_and_factors, group_dim)
        
        return xr.Dataset(gathered_data)


@type_registry.register
class GroupBy(Transformer):
  
    def __init__(self, coords):
        self.coords = [coords] if isinstance(coords, str) else coords

    def _validate_input(self, input, key_spec):
        return True
    
    def output_schema(self, input_schema):

        output_schema = input_schema

        for coord in self.coords:
            output_schema = output_schema.with_coord_grouping(coord)
            
        return output_schema
   
    def _apply(self, data, **kwargs):

        return data.groupby(self.coords)
        