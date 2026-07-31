from dataclasses import dataclass, field
from functools import reduce

from .core import PlotDirective, replace_plot_node


class LayoutMixin:

    def layout(self, by=None, panels=None):
        return SetLayout(by=by, panels=panels)(self)


@dataclass
class Layout:
    num_rows: int
    num_cols: int
    panels: list | None
    flat_panels:  list = field(init=False)

    def __post_init__(self):
        self.flat_panels = [p for row in self.panels for p in row]


@dataclass(frozen=True)
class Panel:
    row: int
    col: int
    coords: dict | None


class SetLayout(PlotDirective):

    def __init__(self, by=None, panels=None):
        self.by = by
        self.panels_string = panels

    def direct(self, input):
        layout_spec = self.parse_layout(input)
        return replace_plot_node(
            input,
            layout_spec=layout_spec,
        )
    
    def validate_layout(self, panel_array, data_schema):
        for row in panel_array:
            for panel in row:
                levels = panel.strip().split()
                if self.by:
                    if len(self.by) != len(levels):
                        raise ValueError(
                            f"The length of {self.by} does not equal the length of {panel}"
                            )
                self.validate_condition_levels(data_schema, levels)
              

    def validate_condition_levels(self, data_schema, panel_levels):
      
        for i, level in enumerate(panel_levels):
            conditions = data_schema.coord_names_by_level(level)
            if self.by:
                if not any(self.by[i] == condition for condition in conditions):
                    raise ValueError(f"{level} does not belong to condition indicated in {self.by[i]}")
            else:
                if len(conditions) > 1:
                    raise ValueError(f"You have provided ambiguous input to layout. {level} does not "
                                     "indicate a unique condition.  Use the `by` keyword.")
                if len(conditions) < 1:
                    raise ValueError(f"{level} does not belong to any conditions")

    def parse_layout(self, input):
        """
        example layout string
        'control IN, control PN; defeat IN, defeat PN'
        """
        data_schema = input.data_source.data_schema
        rows = self.panels_string.split(";")
        panel_array = [row.split(",") for row in rows]
        self.validate_layout(panel_array, data_schema)
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
                        data_schema.coord_names_by_level(level): level for level in levels
                    })
                row.append(panel)
            if max_cols < cols_in_row:
                max_cols = cols_in_row
            panels.append(row)

        layout = Layout(len(rows), max_cols, panels)
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
