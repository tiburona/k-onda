from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import reduce

from k_onda.utils import validate_types

from .core import PlotDirective, replace_plot_node


LayoutBy = str | Iterable[str] | None


class LayoutMixin:

    @validate_types
    def layout(self, panels: str, *, by: LayoutBy = None):
        return SetLayout(panels, by=by)(self)


@dataclass(frozen=True)
class Panel:
    row: int
    col: int
    coords: dict[str, object]


@dataclass
class Layout:
    num_rows: int
    num_cols: int
    panels: list[list[Panel]]
    flat_panels: list[Panel] = field(init=False)

    def __post_init__(self):
        self.flat_panels = [panel for row in self.panels for panel in row]


class SetLayout(PlotDirective):

    def __init__(self, panels: str, *, by: LayoutBy = None):
        self.validate_type_hints()
        by = self._normalize_by(by)
        self._validate_configuration(by, panels)

        self.by = by
        self.panels_string = panels
        self.panel_array = self._parse_panel_array(panels)

    def _normalize_by(self, by):
        if by is None:
            return None
        if isinstance(by, str):
            return (by,)
        return tuple(by)

    def _validate_configuration(self, by, panels):
        if by is not None:
            self.validate_parameter("by", by, nonempty=True)
            self.validate_string_iterable("value in by", by)
            if len(set(by)) != len(by):
                raise ValueError(
                    f"{self.format_call()}: condition names in by cannot be repeated."
                )

        self.validate_parameter("panels", panels, nonempty_string=True)

    def _parse_panel_array(self, panels):
        panel_array = [
            [panel.strip() for panel in row.split(",")]
            for row in panels.split(";")
        ]
        if any(not panel for row in panel_array for panel in row):
            raise ValueError(
                f"{self.format_call()}: panels cannot contain an empty row or panel."
            )
        if self.by is not None:
            for row in panel_array:
                for panel in row:
                    if len(panel.split()) != len(self.by):
                        raise ValueError(
                            f"{self.format_call()}: panel {panel!r} supplies "
                            f"{len(panel.split())} condition levels, but by supplies "
                            f"{len(self.by)} condition names."
                        )
        return panel_array

    def direct(self, input):
        layout_spec = self.parse_layout(input)
        return replace_plot_node(
            input,
            layout_spec=layout_spec,
        )
    
    def _validate_layout_for_schema(self, panel_array, data_schema):
        for row in panel_array:
            for panel in row:
                self._validate_condition_levels(data_schema, panel.split())
              
    def _validate_condition_levels(self, data_schema, panel_levels):
      
        for i, level in enumerate(panel_levels):
            conditions = data_schema.coord_names_by_level(level)
            if self.by is not None:
                if self.by[i] not in conditions:
                    raise ValueError(
                        f"{self.format_call()}: condition level {level!r} does not "
                        f"belong to {self.by[i]!r}."
                    )
            else:
                if len(conditions) > 1:
                    raise ValueError(
                        f"{self.format_call()}: condition level {level!r} is "
                        f"ambiguous; it belongs to {conditions!r}. Use by to "
                        "identify its condition."
                    )
                if len(conditions) < 1:
                    raise ValueError(
                        f"{self.format_call()}: {level!r} is not a level of any "
                        "condition coordinate."
                    )

    def parse_layout(self, input):
        """
        example layout string
        'control IN, control PN; defeat IN, defeat PN'
        """
        data_schema = input.data_source.data_schema
        panel_array = self.panel_array
        self._validate_layout_for_schema(panel_array, data_schema)
        panels = []
        max_cols = 0
       
        for i, r in enumerate(panel_array):
            row = []
            cols_in_row = 0
            
            for j, c in enumerate(r):
                cols_in_row +=1
                levels = c.strip().split()
                if self.by:
                    panel = Panel(row=i, col=j, coords={
                        k: v for k, v in zip(self.by, levels)
                    })
                else:
                    panel = Panel(row=i, col=j, coords={
                        data_schema.coord_names_by_level(level)[0]: level
                        for level in levels
                    })
                row.append(panel)
            if max_cols < cols_in_row:
                max_cols = cols_in_row
            panels.append(row)

        layout = Layout(len(panel_array), max_cols, panels)
        return layout


class LayoutResolver:
    
    def resolve(self, input):
        data_schema = input.data_source.data_schema
        self.validate(data_schema)
        conditions = data_schema.condition_coords
        if len(conditions) == 1:
            condition = conditions[0]
            num_rows = 1
            num_cols = len(condition.levels)
            panels = [[Panel(row=0, col=j, coords={condition.name: level}) 
                      for j, level in enumerate(condition.values())]]
        elif len(conditions) == 2:
            condition_a, condition_b = conditions
            num_rows = len(condition_a.levels)
            num_cols = len(condition_b.levels)
            panels = [[
                Panel(
                    row=i, col=j, coords={condition_a.name: a_level, condition_b.name: b_level}
                    ) 
                for j, b_level in enumerate(condition_b.levels)
                ] for i, a_level in enumerate(condition_a.levels)
                ]
        else:
            condition_a, condition_b, condition_c = conditions
            num_rows = len(condition_a.levels) * len(condition_b.levels)
            num_cols = len(condition_c.levels)
            panels = [[
                Panel(
                    row = i*len(condition_b.levels) + j, 
                    col=k, 
                    coords={
                        condition_a.name: a_level, 
                        condition_b.name: b_level, 
                        condition_c.name: c_level
                        }) 
                        for k, c_level in enumerate(condition_c.levels)]
                        for i, a_level in enumerate(condition_a.levels)
                        for j, b_level in enumerate(condition_b.levels) 
                        
                    ]
        
        return Layout(
            num_rows=num_rows,
            num_cols=num_cols,
            panels = panels
        )

    def validate(self, data_schema):
        conditions = data_schema.condition_coords
        
        if len(conditions) > 3:
            condition_names = "\n".join(condition.name for condition in conditions)
            raise ValueError(
                f"Automatic layout supports at most 3 varying conditions, but found {len(conditions)}:" \
                f"\n{condition_names}\n" \
                "You must define the layout explicitly.")
        num_panels = reduce(
            lambda x, y: x*y, 
            [len(condition.levels) for condition in conditions]
            )
        if num_panels > 20:
            raise ValueError(
                f"Automatic layout supports 20 total levels of conditions, but found {num_panels}." \
                f"You must define the layout explicitly."
            )
