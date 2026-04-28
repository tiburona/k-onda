from __future__ import annotations
from collections.abc import MutableMapping
from .registry import types
from dataclasses import dataclass, field
from enum import Enum, auto
from copy import copy


class AxisKind(Enum):
    POINT_PROCESS_INDEX = auto()
    AXIS = auto()


@dataclass(frozen=True)
class CoordInfo:
    name: str
    metadim: str | None = None


@dataclass(frozen=True)
class AxisInfo:
    name: str                                   # concrete dim or variable name
    kind: AxisKind                              # structural: how the machinery treats it
    metadim: str | None = None                  # semantic: what physical quantity it represents
    coords: tuple[CoordInfo, ...] = ()          # all the coords available on the axis 


@types.register
@dataclass
class Schema:
    axes: list[AxisInfo] = field(default_factory=list)
    value_metadim: str | None = None

    def has_axis_kind(self, kind) -> bool:
        return any(ax.kind == kind for ax in self.axes)
    
    def has_name(self, name) -> bool:
        return any(ax.name == name for ax in self.axes)
    
    def has_dim(self, dim) -> bool:
        return self.has_name(dim) or self.has_metadim(dim)
    
    def has_metadim(self, metadim) -> bool:
        return any(ax.metadim == metadim for ax in self.axes)
    
    def has_coord(self, coord, allow_metadim_match=False) -> bool:
        for ax in self.axes:
            for c in ax.coords:
                if c.name == coord or (allow_metadim_match and c.metadim == coord):
                    return True
        return False
    
    def is_selectable(self, name) -> bool:
        return self.has_name(name) or self.has_metadim(name) or self.has_coord(name)
               
    def axes_by_metadim(self, metadim) -> list[AxisInfo]:
        return [ax for ax in self.axes if ax.metadim == metadim]
    
    def axis_by_name(self, name) -> AxisInfo | None:
        return next((ax for ax in self.axes if ax.name == name), None)
    
    def axis_by_coord_name(self, coord_name):
        for ax in self.axes:
            if ax.name == coord_name:
                return ax
            for c in ax.coords:
                if c.name == coord_name or c.metadim == coord_name:
                    return ax
        return None
    
    def axes_of_kind(self, kind) -> list[AxisInfo]:
        return [ax for ax in self.axes if ax.kind == kind]
    
    def without(self, name) -> Schema:
        new_schema = copy(self)
        new_schema.axes = [ax for ax in self.axes if ax.name != name]
        return new_schema
    
    def without_dim(self, dim) -> Schema:
        new_schema = copy(self)
        new_schema.axes = [ax for ax in self.axes 
                           if ax is not self.ax_with_concrete_dim(dim)] 
        return new_schema
        
    def with_added(self, axis) -> Schema:
        new_schema = copy(self)
        new_schema.axes.append(axis)
        return new_schema
    
    @property
    def dim_names(self) -> list[str]:
        return [ax.name for ax in self.axes]
    
    @property
    def metadims(self) -> list[str]:
        return [ax.metadim for ax in self.axes if ax.metadim is not None]
    
    @property
    def coord_names(self) -> list[str]:
        return [c.name for ax in self.axes for c in ax.coords]
    
    @property
    def selectable(self):
        return set(self.dim_names) | set(self.metadims) | set(self.coord_names)
    
    def is_point_process(self) -> bool:
        return any(ax.kind == AxisKind.POINT_PROCESS_INDEX for ax in self.axes)
    
    def is_point_process_essential(self, dim) -> bool:
        for ax in self.axes:
            if ax.kind == AxisKind.POINT_PROCESS_INDEX:
                if ax.name == dim or ax.metadim == dim:
                    return True
        if self.value_metadim == dim:
            return True
        return False
    
    def ax_with_concrete_dim(self, dim) -> AxisInfo:
        # 1. exact dim match
        if self.has_name(dim):
            return self.axis_by_name(dim)
        # 2. axis-level metadim
        elif self.has_metadim(dim):
            return self.axes_by_metadim(dim)[0]
        # 3. coord name or coord metadim, returns None if not found
        elif self.has_coord(dim, allow_metadim_match=True):
            return self.axis_by_coord_name(dim)
        elif self.value_metadim == dim and self.is_point_process():
            return self.axes_of_kind(AxisKind.POINT_PROCESS_INDEX)[0]
        else:
            return None
    
    def concrete_dim_from(self, dim) -> str:
       axis = self.ax_with_concrete_dim(dim)
       return axis if axis is None else axis.name

    def axis_position_from(self, name) -> int | None:
        for i, ax in enumerate(self.axes):
            if ax.name == name:
                return i
        return None
    
    def position_from(self, axis) -> int | None:
        for i, ax in enumerate(self.axes):
            if ax is axis:
                return i
        return None
    
    def metadim_from(self, coord_name) -> str | None:
        axis = self.axis_by_coord_name(coord_name)
        return axis if axis is None else axis.metadim
    
    def is_value_metadim(self, dim) -> bool:
        return self.value_metadim == dim
    

@types.register
class DatasetSchema(MutableMapping):
    def __init__(self, key_schemas):
        self.key_schemas = key_schemas  # dict[str, Schema]

    def __getitem__(self, key):
        return self.key_schemas[key]

    def __setitem__(self, key, value):
        self.key_schemas[key] = value

    def __delitem__(self, key):
        del self.key_schemas[key]

    def __iter__(self):
        return iter(self.key_schemas)

    def __len__(self):
        return len(self.key_schemas)

    @property
    def dim_names(self):
        # union: for Selector, a dim is 'available' here if any key has it
        return set.union(set(), *(s.dim_names for s in self.key_schemas.values()))
    
    @property
    def selectable(self):
        return set.union(*(s.selectable for s in self.key_schemas.values()))
    
    def replace_key(self, key, new_schema):
        return DatasetSchema({**self.key_schemas, key: new_schema})
    
    def add_key(self, key, new_schema):
        if key in self.key_schemas:
            raise ValueError(f"Key '{key}' already exists in DatasetSchema")
        return DatasetSchema({**self.key_schemas, key: new_schema})
    
    def _variables_with_axis_name(self, name):
        return [vname for vname, s in self.key_schemas.items()
                if any(ax.name == name for ax in s.axes)]
    
    def _variables_with_metadim(self, metadim):
        return [vname for vname, s in self.key_schemas.items()
                if s.value_metadim == metadim
                or any(ax.metadim == metadim for ax in s.axes)]
    
    def default_variable_for(self, dim):
        candidates = (self._variables_with_axis_name(dim) or
                      self._variables_with_metadim(dim))
        return candidates[0] if candidates else None
    
    def is_point_process(self, require_all=True):
        if not require_all:
            return any(s.is_point_process() for s in self.key_schemas.values())
        else:
            return all(s.is_point_process() for s in self.key_schemas.values())

    def is_point_process_essential(self, dim):
        # axis-level match in any sub-schema
        if any(s.is_point_process_essential(dim) for s in self.key_schemas.values()):
            return True
        # variable-level match
        if dim in self.key_schemas:
            return True
        return False

    def is_value_metadim(self, dim) -> bool:
        return any(s.is_value_metadim(dim) for s in self.values())

    def variable_for_metadim(self, metadim) -> str | None:
        for vname, s in self.key_schemas.items():
            if s.value_metadim == metadim:
                return vname
        return None

    def is_selectable(self, dim):
        for schema in self.values():
            if schema.is_selectable(dim):
                return True
        return False
    
    def concrete_dim_from(self, dim):
        for s in self.values():
            ax = s.ax_with_concrete_dim(dim)
            if ax is not None:
                return ax.name
        return None
    
    def metadim_from(self, dim):
        for s in self.values():
            metadim = s.metadim_from(dim)
            if metadim is not None:
                return metadim
        return None


        
