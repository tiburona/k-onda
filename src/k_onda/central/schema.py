from __future__ import annotations
from collections.abc import MutableMapping
from functools import reduce
from collections import defaultdict
from .registry import type_registry
from dataclasses import dataclass, field
from enum import Enum, auto
from copy import copy
from operator import and_, or_


class AxisKind(Enum):
    POINT_PROCESS_INDEX = auto()
    AXIS = auto()
    ORDINAL_INDEX = auto()
    OBSERVATION_INDEX = auto()


@dataclass(frozen=True)
class CoordInfo:
    name: str
    metadim: str | None = None
    is_relative: str | bool = False
    is_grouping: bool = False

@dataclass(frozen=True)
class AxisInfo:
    name: str  # concrete dim or variable name
    kind: AxisKind  # structural: how the machinery treats it
    metadim: str | None = None  # semantic: what physical quantity it represents
    coords: tuple[CoordInfo, ...] = ()  # all the coords available on the axis
    created_from_metadim: str | None = None  # these coords are provenance 
    created_from_dim: str | None = None  # metadata specific to ordinal axes 
 

    def __post_init__(self):
        coords = tuple(self.coords)

        if not any(c.name == self.name for c in coords):
            coords = (
                CoordInfo(name=self.name, metadim=self.metadim),
                *coords,
            )

        object.__setattr__(self, "coords", coords)


@type_registry.register
@dataclass
class Schema:
    """Metadata container that tracks the format of data before it is materialized.
    Allows operations like comparing two coordinates (e.g., 'epoch_time', 'event_time'),
    and disovering that they share the same `metadim`, 'time'."""

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
    
    def axis_by_name_has_coord(self, axis_name, coord_name):
        axis = self.axis_by_name(axis_name)
        if axis is None: 
            return False
        for coord in axis.coords:
            if coord_name == coord.name:
                return True
        return False
    
    def names_by_axis_kind(self, axis_kind) -> list[str]:
        return [ax.name for ax in self.axes_of_kind(axis_kind)]

    def axes_of_kind(self, kind) -> list[AxisInfo]:
        return [ax for ax in self.axes if ax.kind == kind]

    def coord_by_name(self, name) -> CoordInfo:
        for ax in self.axes:
            for coord in ax.coords:
                if coord.name == name:
                    return coord
                
    def ax_coord_map(self, coords=None) -> dict:
        if not coords:
            return {ax.name: self.coord_names_by_dim(ax.name) for ax in self.axes}
        else:
            return {
                ax.name: [
                    c for c in self.coord_names_by_dim(ax.name) if c in coords
                    ] 
                for ax in self.axes
                }

    def without(self, name) -> Schema:
        new_schema = copy(self)
        new_schema.axes = [ax for ax in self.axes if ax.name != name]
        return new_schema

    def without_dim(self, dim) -> Schema:
        new_schema = copy(self)
        new_schema.axes = [
            ax for ax in self.axes if ax is not self.ax_with_concrete_dim(dim)
        ]
        return new_schema
    
    def with_axis(self, axis, *, if_exists="keep"):
        if self.has_name(axis.name):
            if if_exists == "keep":
                return self.copy()
            if if_exists == "error":
                raise ValueError(f"Axis {axis.name!r} aslready exists")
        return self.with_added(axis)

    def with_added(self, axis) -> Schema:
        new_schema = copy(self)
        new_schema.axes = [*self.axes, axis]
        return new_schema

    def rename_axis(self, old_name, name):
        new_schema = self.copy()
        old_axis = self.axis_by_name(old_name)
        new_axis = AxisInfo(
            name=name, 
            kind=old_axis.kind, 
            metadim=old_axis.metadim,
            coords=old_axis.coords, 
            created_from_metadim=old_axis.created_from_metadim, 
            created_from_dim=old_axis.created_from_dim
            )
        for i, axis in enumerate(new_schema.axes):
            if axis.name == old_name:
                new_schema.axes[i] = new_axis
        return new_schema

    def copy(self) -> Schema:
        new_schema = copy(self)
        new_schema.axes = list(self.axes)
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
    def selectable(self) -> set[str]:
        return set(self.dim_names) | set(self.metadims) | set(self.coord_names)

    @property
    def observation_axis(self):
        obs_axes = self.axes_by_kind(AxisKind.OBSERVATION_INDEX)
        return obs_axes[0] if obs_axes else None
    
    @property
    def collectable_coords(self):
        return self.coord_names
    
    def axes_by_kind(self, kind):
        return [ax for ax in self.axes if ax.kind == kind]
    
    def coord_names_by_dim(self, dim) -> list[str]:
        axis = self.axis_by_name(dim)
        if not axis:
            return []
        return [coord.name for coord in axis.coords]
    
    def coord_names_by_axis_kind(self, axis_kind) -> list[str]:
        axes = self.axes_of_kind(axis_kind)
        return [coord.name for ax in axes for coord in ax.coords]

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

    def update_axis_coords(self, ax, coords):
        new_schema = copy(self)
        coords = copy(ax.coords) + coords
        new_ax = AxisInfo(
            name=ax.name,
            metadim=ax.metadim, 
            kind=ax.kind, 
            coords=coords,
            created_from_dim=ax.created_from_dim,
            created_from_metadim=ax.created_from_metadim)
        new_schema = new_schema.without(ax.name)
        new_schema = new_schema.with_added(new_ax)
        return new_schema

    def with_coord_grouping(self, coord_name, *, is_grouping=True):
        axis = self.axis_by_coord_name(coord_name)
        if axis is None:
            raise ValueError(f"Coord {coord_name!r} not found")
        new_coords = tuple(
            CoordInfo(
                name=c.name,
                metadim=c.metadim,
                is_relative=c.is_relative,
                is_grouping=is_grouping if c.name == coord_name else c.is_grouping,
            ) 
            for c in axis.coords
        )

        new_axis = AxisInfo(
            name=axis.name,
            kind=axis.kind,
            metadim=axis.metadim,
            coords=new_coords,
            created_from_dim=axis.created_from_dim,
            created_from_metadim=axis.created_from_metadim
        )

        return self.without(axis.name).with_added(new_axis)
    
    def is_grouping_coord(self, coord_name) -> bool:
        coord = self.coord_by_name(coord_name)
        return coord is not None and coord.is_grouping

    
    def reorder_axes(self, axis_names):
        if set(axis_names) != {ax.name for ax in self.axes}:
            raise ValueError("axis_names must contain exactly the schema axes")
        by_name = {ax.name: ax for ax in self.axes}
        return type(self)(
            axes=tuple(by_name[name] for name in axis_names),
              value_metadim=self.value_metadim
              )

    def get_common_metadim(self, our_coord, other_schema, other_coord):
        metadim = self.metadim_from(our_coord)
        return metadim if metadim == other_schema.metadim_from(other_coord) else None
    
    def ordinal_axes_created_from(self, metadim):
        return [ax for ax in self.axes if ax.created_from_metadim == metadim]
            

@type_registry.register
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
    
    def dim_names_intersection(self):
        return set.intersection(set(), *(s.dim_names for s in self.values()))

    @property
    def selectable(self):
        return set.union(*(s.selectable for s in self.key_schemas.values()))
    
    def has_dim(self, dim):
        return any([key_schema.has_dim(dim) for key_schema in self.key_schemas.values()])

    def replace_key(self, key, new_schema):
        return DatasetSchema({**self.key_schemas, key: new_schema})

    def add_key(self, key, new_schema):
        if key in self.key_schemas:
            raise ValueError(f"Key '{key}' already exists in DatasetSchema")
        return DatasetSchema({**self.key_schemas, key: new_schema})

    def _variables_with_axis_name(self, name):
        return [
            vname
            for vname, s in self.key_schemas.items()
            if any(ax.name == name for ax in s.axes)
        ]

    def _variables_with_metadim(self, metadim):
        return [
            vname
            for vname, s in self.key_schemas.items()
            if s.value_metadim == metadim or any(ax.metadim == metadim for ax in s.axes)
        ]

    def default_variable_for(self, dim):
        candidates = self._variables_with_axis_name(
            dim
        ) or self._variables_with_metadim(dim)
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

    def get_common_metadim(self, our_coord, other_schema, other_coord):
        metadim = self.metadim_from(our_coord)
        return metadim if metadim == other_schema.metadim_from(other_coord) else None
    
    def map_schemas(self, func):
        return type(self)({
            key: func(schema)
            for key, schema in self.key_schemas.items()
        })
    
    def collect_unique(self, func):
        return list(dict.fromkeys(
            item 
            for schema in self.values() 
            for item in func(schema)
        ))

    def with_axis(self, axis, *, if_exists="keep"):
        return self.map_schemas(lambda s: s.with_axis(axis, if_exists=if_exists))
    
    @property
    def coord_names(self):
        return self.collect_unique(lambda s: s.coord_names)
    
    @property
    def observation_axis(self):
        for s in self.values():
            if s.observation_axis:
                return s.observation_axis
            
    @property
    def collectable_coords(self):
        return reduce(and_, [set(s.coord_names) for s in self.values()])

    def names_by_axis_kind(self, axis_kind):
        return self.collect_unique(lambda s: s.names_by_axis_kind(axis_kind))

    def coord_names_by_dim(self, dim):
        return self.collect_unique(lambda s: s.coord_names_by_dim(dim))
    
    def coord_names_by_axis_kind(self, axis_kind):
        return self.collect_unique(lambda s: s.coord_names_by_axis_kind(axis_kind))

    def ax_coord_map(self, coords=None, mode="intersection"):

        # for each schema ax_coord_dict is a mapping of axis names to lists of coords
        # in the intersection case you choose only the axes that are on all the schemas
        # and only the coords that are available on that axis for all sub schemas
        # in the union case you choose the union of all the axes, and all the 
        # coords that are available on that axis in any subschema.
        
        if mode == "intersection":
            dim_names = reduce(and_, [set(s.dim_names) for s in self.values()])
            dict_to_return = {
                dim: list(reduce(
                    and_, 
                    list(set(s.ax_coord_map(coords)[dim]) for s in self.values())
                    ))
                    for dim in dim_names 
                    }
            return dict_to_return
        
        elif mode == "union":
            dim_names = self.dim_names
            dict_to_return = {
                dim: list(reduce(
                    or_, 
                    list(set(s.ax_coord_map(coords).get(dim, [])) for s in self.values())
                    ))
                    for dim in dim_names 
                    }
            return dict_to_return
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
    def axis_by_coord_name(self, coord_name, mode="first"):
        if mode == "first":
            for s in self.values():
                if s.axis_by_coord_name(coord_name):
                    return s.axis_by_coord_name(coord_name)
        elif mode == "union":
            raise NotImplementedError("Mode 'union' for DatasetSchema.axis_by_coord_name "
            "is not yet implemented.")
        else:
            raise ValueError(f"Unknown mode {mode}")



    

            
       