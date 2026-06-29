import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from functools import reduce
from operator import and_

from k_onda.utils import is_unitful




class PlotMixin:

    def plot(self, plot_type):
        return PlotSpecInit(plot_type=plot_type)(self)

    def layout(self, by=None, panels=None):
        return LayoutSpec(by=by, panels=panels)(self)

    def render(self):
        return Render()(self)
    
    def labels(self, x=None, y=None, units="auto"):
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


class PlotNode(PlotMixin):
    def __init__(
            self, 
            data_source=None, 
            plot_type=None, 
            layout=None, 
            coords=None, 
            labels=None
            ):
        self.data_source = data_source
        self.plot_type = plot_type
        self.layout_spec = layout
        self.coords = coords
        self.label_specs = labels


class PlotDirective:

    def __call__(self, input):
        return self.direct(input)
    

class PlotSpecInit(PlotDirective):

    def __init__(self, plot_type=None):
        self.plot_type = plot_type

    def direct(self, input):
        plot_node = PlotNode(
            data_source=input, 
            plot_type=self.plot_type)
        return plot_node


class LayoutSpec(PlotDirective):

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
            layout=layout)


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
        labels = self.parse_label(input)
        return PlotNode(
            data_source = input.data_source,
            plot_type=input.plot_type,
            coords=input.coords,
            labels=labels, 
            layout=input.layout_spec)


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
        gs = fig.add_gridspec(layout.num_rows, layout.num_cols)

        
        panel_ax_map = {}
        data = self.get_plot_data(input)
        role_source_map = self.role_resolver.resolve(input)

        for panel in layout.flat_panels:
            ax = fig.add_subplot(gs[panel.row, panel.col])
            panel_ax_map[(panel.row, panel.col)] = ax
            func = self.plot_function_map[input.plot_type]
            func(panel, ax, data, role_source_map)

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
            fig.supxlabel(text)
        elif label.axis == "y":
            fig.supylabel(text)
        
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
                text += f" {units}"

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
       

    def histogram(self, panel, ax, data, role_source_map):
        x_source = role_source_map["x"]
        panel_data = self.get_panel_data(panel, data)
        x = panel_data.coords[x_source.name].pint.magnitude
        y = panel_data.pint.magnitude
        width = np.median(np.diff(x))
        ax.bar(x, y, width=width, align="edge")
        ax.set_xlim(x[0], x[-1] + width)
        ax.margins(y = 0.08)

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

