import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field, replace
from functools import reduce
from operator import and_
from collections.abc import Iterable
from collections import defaultdict
from matplotlib.patches import Patch
import itertools


from k_onda.utils import is_unitful


BAR_DEFAULT_PROPS = {
    "color": "#4C78A8",
}

PLOT_TYPE_TO_DEFAULTS = {
    "histogram": BAR_DEFAULT_PROPS,
    "time-histogram": BAR_DEFAULT_PROPS,
    "psth": BAR_DEFAULT_PROPS
}


def candidate_matches_selector(selector, candidate):
    for key, selected in selector.items():
        if key not in candidate:
            return False
            
        actual = candidate[key]

        if isinstance(selected, str):
            if actual != selected:
                return False
        elif isinstance(selected, Iterable):
            if actual not in selected:
                return False
        else:
            if actual != selected:
                return False

    return True


class PlotMixin:

    def plot(self, plot_type):
        return SetPlotType(plot_type=plot_type)(self)

    def layout(self, by=None, panels=None):
        return SetLayout(by=by, panels=panels)(self)

    def render(self):
        return Render()(self)
    
    def label(self):
        raise NotImplementedError("Default label method not yet implemented.")
    
    def plot_labels(self, x=None, y=None, units="auto"):
        labels = []

        for ax_param, ax_string in zip([x, y], ["x", "y"]):
            if ax_param:
                if isinstance(ax_param, str):
                    labels.append(Label(scope="figure", axis=ax_string, text=ax_param, units=units))
                elif isinstance(ax_param, dict):
                    params = {"scope":"figure", "axis":ax_string, "units":units} | ax_param
                    labels.append(Label(**params))
                else:
                    raise TypeError(f"Unknown type {type(ax_param)} for `{ax_string}`")
                
        node = self
        for label in labels:
            node = AddLabel(label)(node)

        return node

    def panel_labels(self, top=None, left=None, right=None, bottom=None, where="all", units=None):
        labels = []
        ax_params = [top, left, right, bottom]
        ax_strings = ["top", "left", "right", "bottom"]
        for ax_param, ax_string in zip(ax_params, ax_strings):
            if isinstance(ax_param, str):
                labels.append(
                    Label(scope="panel", side=ax_string, text=ax_param, where=where, units=units)
                    )
            elif isinstance(ax_param, dict):
                params = {
                    "scope": "panel", "side": ax_string, "where": where, "units":units
                    } | ax_param
                labels.append(Label(**params))
            elif ax_param is None:
                continue
            else:
                raise TypeError(f"Unknown type {type(ax_param)} for `{ax_string}`")
               
        node = self
        for label in labels:
            node = AddLabel(label)(node)
            
        return node

    def style(self, rules):
        
        style_rules = []
        for rule in rules:
            selector = {}
            props = {}
            for key in rule:
                if key in self.data_source.data_schema.coord_names:
                    selector[key] = rule[key]
                else:
                    props[key] = rule[key]
            style_rule = StyleRule(selector=selector, props=props)
            style_rules.append(style_rule)
         
        node = self
        for rule in style_rules:
            node = AddStyle(style_rule=rule)(node)

        return node
       
    def colors(self, rules=None, **kwargs):
        data_schema = self.data_source.data_schema
        # possible keys in kwargs
        # "control|defeat"
        # "control&tone"
        # "control"

        rules = rules or []

        style_rules = [
            StyleRule(
                props={"color": rule["color"]}, 
                selector={k: v for k, v in rule.items() if k != "color"}
                )
            for rule in rules
        ]

        if kwargs:
            data_schema = self.data_source.data_schema
            style_rules.extend([self._parse_kwarg(kwargs, kwarg, data_schema) for kwarg in kwargs])
        
        node = self
        for rule in style_rules:
            node = AddStyle(style_rule=rule)(node)

        return node
    
    def _get_coord_level_map(self, levels, data_schema):
        coord_level_map = defaultdict(list)
        for level in levels:
            matches = self._get_coord_matches(level, data_schema)
            coord_level_map[matches[0]].append(level)
        return dict(coord_level_map)
    
    def _get_coord_matches(self, level, data_schema):
        matches = data_schema.coord_names_by_level(level)
        if len(matches) == 0:
            raise ValueError(f"No coord level {level!r} found in data schema.")
        if len(matches) > 1:
            raise ValueError(
                f"Condition value {level!r} is ambiguous; found in coords {matches}."
            )
        return matches
    
    def _parse_kwarg(self, kwargs, kwarg, data_schema):
        if "|" in kwarg:
            if "&" in kwarg:
                raise ValueError("Use structured input format if you have | and & conditions")
            levels = kwarg.split("|")
            coord_level_map = self._get_coord_level_map(levels, data_schema)
            if len(coord_level_map) > 1:
                raise ValueError("You can only use | keywords for combination within a condition." \
                "For more complex combinations, use structured input.")
            rule = StyleRule(
                selector = coord_level_map,
                props = {"color": kwargs[kwarg]}
            )
        elif "&" in kwarg:
            levels = kwarg.split("&")
            coord_level_map = self._get_coord_level_map(levels, data_schema)
            if any(len(coord_level_map[key]) > 1 for key in coord_level_map):
                raise ValueError("You can only use & keywords for combination across conditions." \
                "For more complex combinations, use structured input.")

            rule = StyleRule(
                selector = coord_level_map,
                props = {"color": kwargs[kwarg]}
            )
            
        else:
            matches = self._get_coord_matches(kwarg, data_schema)
            coord = matches[0]
            
            rule = StyleRule(
                selector={coord: kwarg},
                props = {"color": kwargs[kwarg]}
            )

        return rule
    
    def legend(self, *args, position="upper right", title="", **kwargs):
        infer_entries = False
        if not args:
            infer_entries=True
            legend_entries = []
        else:
            if len(args) > 1:
                raise ValueError("`legend` accepts a maximum of one positional arg")
            if not isinstance(args[0], dict):
                raise ValueError("Positional arg passed to `legend` must be of type `dict`.")
            legend_entries = self.construct_legend_entries_from_structured_input(args)

        legend_spec = Legend(
                entries=legend_entries,
                title=title,
                position=position,
                infer_entries=infer_entries,
                kwargs = kwargs
            )

        return AddLegend(legend_spec=legend_spec)(self)

    def construct_legend_entries_from_structured_input(self, args):
        if len(args) != 1:
            raise ValueError(
                "If you are using the structured format for `legend` you can " \
                "only pass one positional arg"
                )
        legend_entries = []

        for label, vals in args[0].items():
            legend_entries.append(
                LegendEntry(
                    label=label,
                    selector={k:v for k, v in vals.items() if k != "traits"},
                    traits=vals.get("traits")
                )
            )
        return legend_entries
            


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


@dataclass(frozen=True)
class PlotSource:
    kind: str                  # "coord" | "values"
    name: str | None = None    # coord name if kind == "coord"


@dataclass
class Label:
    text: str
    scope: str = "figure"
    axis: str = None
    side: str = None
    where: str = "all"
    units: str = "auto"
    kwargs: dict = field(default_factory=dict)


@dataclass
class LabelPlan:
    explicit: list[Label] = field(default_factory=list)
    infer_missing: bool = False


@dataclass
class StyleRule:
    selector: dict = field(default_factory=dict)
    props: dict = field(default_factory=dict)


@dataclass
class LegendEntry:
    label: str
    selector: dict = field(default_factory=dict)
    traits: list[str] | None = None
    props: dict | None = None


@dataclass
class Legend:
    entries: list[LegendEntry] = field(default_factory=list)
    infer_entries: bool = False
    position: str = "upper right"
    title: str | None = None
    kwargs: dict = field(default_factory=dict)


class PlotNode(PlotMixin):
    def __init__(
            self, 

            data_source=None, 
            plot_type=None, 
            layout_spec=None, 
            coords=None, 
            label_plan=None,
            style_rules=None,
            legend_spec=None
            ):
        self.data_source = data_source
        self.plot_type = plot_type
        self.layout_spec = layout_spec
        self.coords = coords
        self.label_plan = label_plan
        self.style_rules = style_rules
        self.legend_spec = legend_spec


class PlotDirective:

    def __call__(self, input):
        return self.direct(input)
    

class SetPlotType(PlotDirective):

    def __init__(self, plot_type=None):
        self.plot_type = plot_type

    def direct(self, input):
        plot_node = PlotNode(
            data_source=input, 
            plot_type=self.plot_type)
        return plot_node


class SetLayout(PlotDirective):

    def __init__(self, by=None, panels=None):
        self.by = by
        self.panels_string = panels

    def direct(self, input):
        layout_spec = self.parse_layout(input)
        return PlotNode(
            data_source = input.data_source,
            plot_type=input.plot_type,
            coords=input.coords,
            label_plan=input.label_plan, 
            layout_spec=layout_spec,
            style_rules=input.style_rules,
            legend_spec=input.legend_spec)
    
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
    

class AddLabel(PlotDirective):

    def __init__(self, label, infer_missing=False):
        self.label = label 
        self.infer_missing = infer_missing
       

    def direct(self, input):
        self.validate_label(input)

        if input.label_plan is None:
            explicit = [self.label] or []
            infer_missing = self.infer_missing
        else:
            if self.label:
                explicit = [*input.label_plan.explicit, self.label]
            else:
                explicit = input.label_plan.explicit
            infer_missing = input.label_plan.infer_missing or self.infer_missing
        
        label_plan = LabelPlan(explicit=explicit, infer_missing=infer_missing)
        
        return PlotNode(
            data_source = input.data_source,
            plot_type=input.plot_type,
            coords=input.coords,
            label_plan=label_plan, 
            layout_spec=input.layout_spec,
            style_rules=input.style_rules,
            legend_spec=input.legend_spec)

    def validate_label(self, input):
        if input.label_plan is None:
            return
        existing_labels = input.label_plan.explicit or []
        if self.label.scope == "figure":
            self.check_figure_labels(existing_labels)
        else:
            self.check_panel_labels(existing_labels)

    def check_figure_labels(self, existing_labels):
        for label in existing_labels:
            if label.axis == "x" and self.label.axis == "x":
                raise ValueError("You are trying to set an x axis figure label" \
                "but the figure already has one.")
            elif label.axis == "y" and self.label.axis == "y":
                raise ValueError("You are trying to set an y axis figure label" \
                "but the figure already has one.")
            
    def check_panel_labels(self, existing_labels):
        panel_labels = [el for el in existing_labels if el.scope == "panel"]
        if any([self._is_conflicting(panel_label, self.label) for panel_label in panel_labels]):
            raise ValueError("You are setting a panel label on a panel and axis that already has" \
            "one.")

    def _is_conflicting(self, panel_label_1, panel_label_2):
        if not panel_label_1.side == panel_label_2.side:
            return False
        if panel_label_1.where == "all" or panel_label_2.where == "all":
            return True
        compatible = {
            "left": "right",
            "right": "left",
            "top": "bottom",
            "bottom": "top"
        }
        if compatible[panel_label_1.where] != panel_label_2.where: 
            return True
        return False


class AddStyle(PlotDirective):
    def __init__(self, style_rule):
        self.style_rule = style_rule

    def direct(self, input):
        style_rules = input.style_rules or []
        return PlotNode(
            data_source = input.data_source,
            plot_type=input.plot_type,
            coords=input.coords,
            label_plan=input.label_plan, 
            layout_spec=input.layout_spec,
            legend_spec=input.legend_spec,
            style_rules=[*style_rules, self.style_rule]
        )
    

class AddLegend(PlotDirective):
    def __init__(self, legend_spec):
        self.legend_spec = legend_spec

    def direct(self, input):
        return PlotNode(
            data_source = input.data_source,
            plot_type=input.plot_type,
            coords=input.coords,
            label_plan=input.label_plan, 
            layout_spec=input.layout_spec,
            style_rules=input.style_rules,
            legend_spec=self.legend_spec
        )


class PlotRoleResolver:

    def resolve(self, plot_node, layout_spec):
        if plot_node.plot_type in ("histogram", "time-histogram", "psth"):
            return self.resolve_histogram(plot_node, layout_spec)
        raise NotImplementedError("Only histogram plots are currently implemented")

    def resolve_histogram(self, plot_node, layout_spec):
        role_source_map = {
            "x": PlotSource(kind="coord", name=self.infer_x_source(plot_node, layout_spec)),
            "y": PlotSource(kind="values")
        }
        return role_source_map

    def infer_x_source(self, input, layout_spec):
        if input.coords:
            # The user has supplied a coord name
            source = next(iter(input.coords.values()))
            
        else:
            # The default coord is the name of the one remaining axis.
            data_schema = input.data_source.data_schema
            panel_coord_names = set().union(
                *(panel.coords.keys() for panel in layout_spec.flat_panels)
            )
            default_axes = data_schema.axis_names_minus_axes_with_coords(panel_coord_names)

            if len(default_axes) != 1:
                raise ValueError("Unable to determine default axis.")
            source = default_axes[0]

        return source
    

class LabelResolver:
    
    def resolve(self, input, role_source_map, layout_spec):
       
        if input.label_plan.infer_missing:
            label_plan = self.infer_missing_labels(input, role_source_map, layout_spec)
        else:
            label_plan = input.label_plan

        return label_plan

    def infer_missing_labels(self, input,  role_source_map, layout_spec):
        label_plan = self.ensure_axis_labels(input.label_plan, role_source_map)
        label_plan = self.ensure_panel_labels(layout_spec, label_plan)
        return label_plan

    def ensure_axis_labels(self, label_plan, role_source_map):
        labels = list(label_plan.explicit)
        for axis in ("x", "y"):
            axis_label = [label for label in label_plan.explicit if label.axis == axis]
            if not axis_label:
                labels.append(Label(role_source_map[axis].name, axis= axis))

        return replace(label_plan, explicit=labels)
    
    def ensure_panel_labels(self, layout, label_plan):

        labels = list(label_plan)

        panel_labels_exist = bool([
            label for label in label_plan.explicit if label.scope == "panel"
            ])
        
        if not panel_labels_exist:
            panels = layout.flat_panels
            coord_sets = {set(panel.coords) for panel in panels}
            
            if len(coord_sets) != 1:
                raise ValueError(
                    "Automatic panel labels require all panels to use the same "
                    "condition coordinates. Use panel_labels() explicitly."
                    )
            
            labels.append(
                Label(
                    text = " ".join(f"{{{coord_name}}}" for coord_name in panels[0].coords),
                    scope="panel",
                    side="bottom",
                    where="all",
                    units=None
                    )
            )

        return replace(label_plan, explicit=labels)


class LegendResolver:

    def resolve(self, input):
        legend_spec = input.legend_spec

        if legend_spec.infer_entries:
            entries = self.infer_legend_entries(input)
        else:
            entries = self.resolve_legend_entries(input)

        return replace(legend_spec, entries=entries)

    def resolve_style(self, selector, style_rules, defaults=None):
        defaults = defaults or {}
        props = {**defaults}
        for rule in style_rules:
            if candidate_matches_selector(rule.selector, selector):
                props = {**props, **rule.props}
        return props
    
    def infer_legend_entries(self, input):
        # Combined legend inference:
        #
        # 1. Find the condition coordinates referenced by StyleRule selectors.
        #    Coordinates not referenced by any StyleRule do not affect the legend.
        #
        # 2. Get every level of each relevant coordinate from the schema, including
        #    levels that have no explicit StyleRule and therefore use defaults.
        #
        # 3. Generate the Cartesian product of those level domains. Each combination
        #    becomes a concrete candidate selector.
        #
        # 4. For each candidate selector, resolve its effective props by starting with
        #    the plot-type defaults and applying every matching StyleRule in order.
        #
        # 5. Generate one label from the complete selector combination.
        #
        # 6. Construct one LegendEntry from each selector, label, and resolved props.
        if not input.style_rules:
            raise ValueError("You have called `legend` with no arguments, but have added no styles " \
            "from which to build a legend")
        style_rules = input.style_rules
        data_schema = input.data_source.data_schema
        entries = []

        coordinates = list(
            dict.fromkeys([k for rule in style_rules for k in rule.selector.keys()])
            )

        if not coordinates:
            raise ValueError(
                "Cannot infer legend entries because no style rule refers "
                "to a condition coordinate."
            )

        for coord_name in coordinates:
            coord = data_schema.coord_by_name(coord_name)
            if  not coord.is_condition:
                raise ValueError(f"You've attached a style rule to coordinate {coord_name}, which" 
                    "is not a condition, and are inferring a legend. Style rules"
                    "should only be attached to condition coordinates.")
            if  coord.levels is None:
                raise ValueError(f"You've attached a style rule to coordinate {coord_name} without" 
                    "levels, and are inferring a legend.  Style rules should only be " 
                    "attached to condition coordinates with levels.")

        coord_names_to_levels = data_schema.coord_name_levels_map(coordinates)
        coordinates = list(coord_names_to_levels)
        levels = coord_names_to_levels.values()

        selectors = [
            dict(zip(coordinates, combo)) for combo in itertools.product(*levels)
        ]

        defaults = PLOT_TYPE_TO_DEFAULTS[input.plot_type]

        for combination in selectors:
            props = self.resolve_style(combination, style_rules, defaults)
            label = " ".join(combination.values())
            entries.append(LegendEntry(label=label, props=props, selector=combination))

        return entries

    def resolve_legend_entries(self, input):
        style_rules = input.style_rules or []
        defaults = PLOT_TYPE_TO_DEFAULTS[input.plot_type]
        
        entries = input.legend_spec.entries
        resolved_entries = []
        for entry in entries:
            if entry.props is None:
                props = self.resolve_style(entry.selector, style_rules, defaults=defaults)
                if entry.traits is not None:
                    props = {k: v for k, v in props.items() if k in entry.traits}
                    resolved_entry = replace(entry, props=props)
                else:
                    resolved_entry = replace(entry, traits=list(props), props=props)
            else:
                resolved_entry = entry
            resolved_entries.append(resolved_entry)

        return resolved_entries
    

class LayoutResolver:
    
    def resolve(self, input):
        data_schema = input.data_source.data_schema
        self.validate(data_schema)
        conditions = data_schema.condition_coords
        if len(conditions) == 1:
            condition = conditions[0]
            num_rows = 1
            num_cols = len(condition.levels)
            flat_panels = [Panel(row=0, col=j, coords={condition.name: level}) 
                      for j, level in enumerate(condition.values())]
            panels = [flat_panels]
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
            flat_panels = [panel for row in panels for panel in row]
        else:
            condition_a, condition_b, condition_c = conditions
            num_rows = condition_a.levels * condition_b.levels
            num_cols = condition_c.levels
            panels = [[[
                Panel(
                    row = j*i + j, 
                    col=k, 
                    coords={
                        condition_a.name: a_level, 
                        condition_b.name: b_level, 
                        condition_c.name: c_level
                        }) 
                        for k, c_level in enumerate(condition_c.levels)] 
                        for j, b_level in enumerate(condition_b.levels)] 
                        for i, a_level in enumerate(condition_a.levels)
                    ]
            flat_panels = [panel for facet in panels for row in facet for panel in row]
        
        return Layout(
            num_rows=num_rows,
            num_cols=num_cols,
            panels = panels,
            flat_panels = flat_panels
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
        

class Render(PlotDirective):

    def __init__(
            self, 
            role_resolver=None, 
            label_resolver=None, 
            layout_resolver=None, 
            legend_resolver=None
            ):
        self.role_resolver = role_resolver or PlotRoleResolver()
        self.label_resolver = label_resolver or LabelResolver()
        self.layout_resolver = layout_resolver or LayoutResolver()
        self.legend_resolver = legend_resolver or LegendResolver()

    def direct(self, input):
        return self.make_figure(input)

    def make_figure(self, input):
        figsize = getattr(input, 'figsize', (8, 8))
        fig = plt.figure(figsize=figsize)
        data = self.get_plot_data(input)

        if input.layout_spec is None:
            layout = self.layout_resolver.resolve(input)
        else:
            layout = input.layout_spec
        style_rules = input.style_rules or []
        gs = fig.add_gridspec(layout.num_rows, layout.num_cols)
        panel_ax_map = {}
        role_source_map = self.role_resolver.resolve(input, layout)

        for panel in layout.flat_panels:
            ax = fig.add_subplot(gs[panel.row, panel.col])
            panel_ax_map[(panel.row, panel.col)] = ax
            func = self.plot_function_map[input.plot_type]
            func(input.plot_type, panel, ax, data, role_source_map, style_rules)

        label_plan =  self.label_resolver.resolve(input, role_source_map, layout)

        if label_plan:
            self.add_labels(layout, label_plan, fig, data, panel_ax_map, role_source_map)

        if input.legend_spec:
            legend_spec = self.legend_resolver.resolve(input)
            self.add_legend(legend_spec, fig)

        fig.show()

        return fig
    
    def bar_kwargs_from_props(self, props):
        # matplotlib accepts {'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
        # for hatch
        
        bar_kwargs = {k:v for k, v in props.items() if k not in ["pattern", "color"]}
        if "pattern" in props:
            bar_kwargs["hatch"] = props["pattern"]
        if "color" in props:
            bar_kwargs["facecolor"] = props["color"]
        return bar_kwargs
        
    
    def add_legend(self, legend_spec, fig):
        entries = legend_spec.entries
      
        handles = [Patch(**self.bar_kwargs_from_props(entry.props)) for entry in entries]
        labels = [entry.label for entry in entries]

        fig.legend(
            handles=handles, 
            labels=labels, 
            title=legend_spec.title, 
            loc=legend_spec.position,
            **legend_spec.kwargs
            )

    
    def add_labels(self, layout, label_plan, fig, data, panel_ax_map, role_source_map):
        labels = label_plan.explicit or []
        for label in labels:
            if label.scope == "figure":
                self.add_figure_label(label, data, fig, role_source_map)
            elif label.scope == "panel":
                self.add_panel_label(layout, label, data, panel_ax_map, role_source_map)
            else:
                raise ValueError(f"Unknown value {label.scope} for label scope.")


    def add_figure_label(self, label, data, fig, role_source_map):
        text = self.resolve_label_text(label, data, role_source_map)
        if label.axis == "x":
            fig.supxlabel(text, **label.kwargs)
        elif label.axis == "y":
            fig.supylabel(text, **label.kwargs)
        
    def add_panel_label(self, layout, label, data, panel_ax_map, role_source_map):
        panels_to_label = self.select_panels_to_label(layout, label.where)
        if label.side in ("bottom", "top"):
            for panel in panels_to_label:
                ax = panel_ax_map[(panel.row, panel.col)]
                text = self.resolve_label_text(label, data, role_source_map, panel=panel)
                ax.set_xlabel(text, **label.kwargs)
                ax.xaxis.set_label_position(label.side)
        if label.side in ("left", "right"):
            for panel in panels_to_label:
                ax = panel_ax_map[(panel.row, panel.col)]
                text = self.resolve_label_text(label, data, role_source_map, panel=panel)
                ax.set_ylabel(text, **label.kwargs)
                ax.yaxis.set_label_position(label.side)

    def resolve_label_text(self, label, data, role_source_map, panel=None):
        text = label.text
        if "{" in text and panel:
            text = text.format(**panel.coords)
        if label.units:

            if label.axis == "x" or label.side in ("top", "bottom"):
                source = role_source_map["x"]
            elif label.axis == "y" or label.side in ("left", "right"):
                source = role_source_map["y"]

            if source.kind == "coord":
                units = data.coords[source.name].pint.units
            else:
                if is_unitful(data):
                    units = data.pint.units
                else:
                    units = ''

            if units:
                text += f" ({units})"

        return text
    
    def select_panels_to_label(self, layout, where):
        panels = layout.panels
        flat_panels = layout.flat_panels
        if where == "all":
            return flat_panels
        elif where == "top":
            return panels[0]
        elif where == "bottom":
            return panels[-1]
        elif where == "left":
            return [p for row in panels for j, p in enumerate(row) if j == 0]
        elif where == "right":
            return [p for row in panels for j, p in enumerate(row) if j == len(row) - 1]
        else:
            raise ValueError(f"Unknown value for where {where}")
       

    def histogram(self, plot_type, panel, ax, data, role_source_map, style_rules):
        x_source = role_source_map["x"]
        panel_data = self.get_panel_data(panel, data)
        x = panel_data.coords[x_source.name].pint.magnitude
        y = panel_data.pint.magnitude
        width = np.median(np.diff(x))
        kwargs = self.get_merged_kwargs(plot_type, style_rules, panel_data)
        ax.bar(x, y, width=width, align="edge", **kwargs)
        ax.set_xlim(x[0], x[-1] + width)
        ax.margins(y = 0.08)

    def get_merged_kwargs(self, plot_type, style_rules, panel_data):
        defaults = PLOT_TYPE_TO_DEFAULTS[plot_type]
        selected_style_rules = [
            sr for sr in style_rules if self.panel_matches_style_rule(panel_data, sr)
            ]
        kwargs_list = [sr.props for sr in selected_style_rules]
        merged_kwargs = reduce(lambda acc, d: {**acc, **d}, kwargs_list, {})
        merged_kwargs = {**defaults, **merged_kwargs}
        merged_kwargs = self.bar_kwargs_from_props(merged_kwargs)
        return merged_kwargs

    def panel_matches_style_rule(self, panel_data, style_rule):
        return candidate_matches_selector(style_rule.selector, panel_data.coords)

    def get_plot_data(self, input):
        compiled_input = input.data_source.compile()
        return compiled_input.data
    
    def get_panel_data(self, panel, data):
        coords = panel.coords
        mask = reduce(
            and_, [(data.coords[coord] == coords[coord]) for coord in coords]
            )
        data = data.where(mask, drop=True).squeeze()
        return data
    
    @property
    def plot_function_map(self):

        return {
            "histogram": self.histogram,
            "time-histogram": self.histogram,
            "psth": self.histogram
        }

