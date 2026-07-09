import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from functools import reduce
from operator import and_
from collections.abc import Iterable
from collections import defaultdict

from k_onda.utils import is_unitful




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
class StyleRule:
    selector: dict = field(default_factory=dict)
    props: dict = field(default_factory=dict)


class PlotNode(PlotMixin):
    def __init__(
            self, 
            data_source=None, 
            plot_type=None, 
            layout=None, 
            coords=None, 
            labels=None,
            style_rules=None
            ):
        self.data_source = data_source
        self.plot_type = plot_type
        self.layout_spec = layout
        self.coords = coords
        self.label_specs = labels
        self.style_rules = style_rules





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
        layout = self.parse_layout()
        return PlotNode(
            data_source = input.data_source,
            plot_type=input.plot_type,
            coords=input.coords,
            labels=input.label_specs, 
            layout=layout,
            style_rules=input.style_rules)


    def parse_layout(self):
        """
        example layout string
        'control IN, control PN; defeat IN, defeat PN'
        """
        rows = self.panels_string.split(";")
        panel_array = [row.split(",") for row in rows]
        panels = []
        max_cols = 0
       
        for i, r in enumerate(panel_array):
            row = []
            cols_in_row = 0
            
            for j, c in enumerate(r):
                cols_in_row +=1
                
                conditions = c.strip().split()

                if len(conditions) != len(self.by):
                    raise ValueError(f"Length {self.by} does not equal length {c}")
                panel = Panel(row=i, col=j, coords={
                    k: v for k, v in zip(self.by, conditions)
                })
                row.append(panel)
            if max_cols < cols_in_row:
                max_cols = cols_in_row
            panels.append(row)

        layout = Layout(len(rows), max_cols, panels)
        return layout
    

class AddLabel(PlotDirective):

    def __init__(self, label):
        self.label = label

    def direct(self, input):
        return PlotNode(
            data_source = input.data_source,
            plot_type=input.plot_type,
            coords=input.coords,
            labels=self.parse_label(input), 
            layout=input.layout_spec,
            style_rules=input.style_rules)


    def parse_label(self, input):
        # my job is going to be merging or raising if there's a conflict
        # also I can resolve text maybe?

        existing_labels = input.label_specs or []
        if self.label.scope == "figure":
            self.check_figure_labels(existing_labels)
        else:
            self.check_panel_labels(existing_labels)

        return [*existing_labels, self.label]

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
            labels=input.label_specs, 
            layout=input.layout_spec,
            style_rules=[*style_rules, self.style_rule]
        )

    

class PlotRoleResolver:
    def resolve(self, plot_node):
        if plot_node.plot_type in ("histogram", "time-histogram", "psth"):
            return self.resolve_histogram(plot_node)
        raise NotImplementedError("Only histogram plots are currently implemented")

    def resolve_histogram(self, plot_node):
        role_source_map = {
            "x": PlotSource(kind="coord", name=self.infer_x_source(plot_node)),
            "y": PlotSource(kind="values")
        }
        return role_source_map

    def infer_x_source(self, input):
        if input.coords:
            # The user has supplied a coord name
            source = next(iter(input.coords.values()))
            
        else:
            # The default coord is the name of the one remaining axis.
            data_schema = input.data_source.data_schema
            panel_coord_names = set().union(
                *(panel.coords.keys() for panel in input.layout_spec.flat_panels)
            )
            default_axes = data_schema.axis_names_minus_axes_with_coords(panel_coord_names)

            if len(default_axes) != 1:
                raise ValueError("Unable to determine default axis.")
            source = default_axes[0]

        return source



class Render(PlotDirective):

    def __init__(self, role_resolver=None):
        self.role_resolver = role_resolver or PlotRoleResolver()

    def direct(self, input):
        return self.make_figure(input)

    def make_figure(self, input):
        figsize = getattr(input, 'figsize', (8, 8))
        fig = plt.figure(figsize=figsize)
        layout = input.layout_spec
        style_rules = input.style_rules or []
        gs = fig.add_gridspec(layout.num_rows, layout.num_cols)

        
        panel_ax_map = {}
        data = self.get_plot_data(input)
        role_source_map = self.role_resolver.resolve(input)

        for panel in layout.flat_panels:
            ax = fig.add_subplot(gs[panel.row, panel.col])
            panel_ax_map[(panel.row, panel.col)] = ax
            func = self.plot_function_map[input.plot_type]
            func(panel, ax, data, role_source_map, style_rules)

        self.add_labels(input, fig, data, panel_ax_map, role_source_map)


        fig.show()

        return fig
    
    def add_labels(self, input, fig, data, panel_ax_map, role_source_map):
        labels = input.label_specs or []
        for label in labels:
            if label.scope == "figure":
                self.add_figure_label(label, data, fig, role_source_map)
            elif label.scope == "panel":
                self.add_panel_label(input, label, data, panel_ax_map, role_source_map)
            else:
                raise ValueError(f"Unknown value {label.scope} for label scope.")


    def add_figure_label(self, label, data, fig, role_source_map):
        text = self.resolve_label_text(label, data, role_source_map)
        if label.axis == "x":
            fig.supxlabel(text, **label.kwargs)
        elif label.axis == "y":
            fig.supylabel(text, **label.kwargs)
        
    def add_panel_label(self, input, label, data, panel_ax_map, role_source_map):
        panels_to_label = self.select_panels_to_label(input, label.where)
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
    

    def select_panels_to_label(self, input, where):
        layout = input.layout_spec
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
       

    def histogram(self, panel, ax, data, role_source_map, style_rules):
        x_source = role_source_map["x"]
        panel_data = self.get_panel_data(panel, data)
        x = panel_data.coords[x_source.name].pint.magnitude
        y = panel_data.pint.magnitude
        width = np.median(np.diff(x))
        kwargs = self.get_merged_kwargs(style_rules, panel_data)
        ax.bar(x, y, width=width, align="edge", **kwargs)
        ax.set_xlim(x[0], x[-1] + width)
        ax.margins(y = 0.08)

    def get_merged_kwargs(self, style_rules, panel_data):
        selected_style_rules = [
            sr for sr in style_rules if self.panel_matches_style_rule(panel_data, sr)
            ]
        kwargs_list = [self.bar_kwargs_from_style_rule(sr) for sr in selected_style_rules]
        merged_kwargs = reduce(lambda acc, d: {**acc, **d}, kwargs_list, {})
        return merged_kwargs

    def bar_kwargs_from_style_rule(self, style_rule):
        kwargs = {}
        for prop in style_rule.props:
            if prop == "color":
                kwargs["color"] = style_rule.props["color"]

        return kwargs

        # TODO need to think about other kwargs later.  Not clear
        # that every style can just be passed through as a kwarg.

    def panel_matches_style_rule(self, panel_data, style_rule):
        for key in style_rule.selector:
            if key not in panel_data.coords:
                return False
            
            selected = style_rule.selector[key]

            if isinstance(selected, str):
                if panel_data.coords[key].item() != selected:
                    return False
            elif isinstance(selected, Iterable):
                if panel_data.coords[key].item() not in selected:
                    return False
            else:
                raise TypeError(f"Unknown type for {style_rule.selector[key]}")

        return True

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

