import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from functools import reduce
from operator import and_




class PlotMixin:

    def plot(self, plot_type):
        return PlotSpecInit(plot_type=plot_type)(self)

    def layout(self, by=None, panels=None):
        return LayoutSpec(by=by, panels=panels)(self)

    def render(self):
        return Render()(self)
    
    def labels(self, x=None, y=None, units=True):
        return Label(x=x, y=y, units=units)


@dataclass
class Layout:
    num_rows: int
    num_cols: int
    panels: list | None
    flat_panels:  list = field(init=False)

    def __post_init__(self):
        self.flat_panels = [p for row in self.panels for p in row]


@dataclass
class Panel:
    row: int
    col: int
    coords: dict | None


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
        self.labels = labels


class PlotDirective:

    def __call__(self, input):
        return self.direct(input)
    

class PlotSpecInit(PlotDirective):

    def __init__(self, plot_type=None):
        self.plot_type = plot_type

    def direct(self, input):
        plot_node = PlotNode(data_source=input, plot_type=self.plot_type)
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
            labels=input.labels, 
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
                
                conditions = c.strip().split(" ")
                panel = Panel(row=i, col=j, coords={
                    k: v for k, v in zip(self.by, conditions)
                })
                row.append(panel)
            if max_cols < cols_in_row:
                max_cols = cols_in_row
            panels.append(row)

        layout = Layout(len(rows), max_cols, panels)
        return layout
    
class Label(PlotDirective):

    def __init__(self, x, y, units=True):
        self.x = x
        self.y = y
        self.units = units

    def direct(self, input):
        labels = self.parse_labels()
        return PlotNode(
            data_source = input.data_source,
            plot_type=input.plot_type,
            coords=input.coords,
            labels=labels, 
            layout=input.layout)


    def parse_labels(self):
        pass


class Render(PlotDirective):

    def direct(self, input):
        return self.make_figure(input)


    def make_figure(self, input):
        figsize = getattr(input, 'figsize', (8, 8))
        fig = plt.figure(figsize=figsize)
        layout = input.layout_spec
        gs = fig.add_gridspec(layout.num_rows, layout.num_cols)

        data = self.get_plot_data(input)

        for panel in layout.flat_panels:
            ax = fig.add_subplot(gs[panel.row, panel.col])
            func = self.plot_function_map[input.plot_type]
            func(input, panel, ax, data)

        fig.show()

        return fig

    def histogram(self, input, panel, ax, data):
        if input.coords:
            coord = next(iter(input.coords.values()))
        else:
            data_schema = input.data_source.data_schema
            default_axes = data_schema.axis_names_minus_axes_with_coords(list(panel.coords.keys()))
            if len(default_axes) > 1:
                raise ValueError("Data has multiple plottable axes. Only 1-D histograms " \
                "are currently supported.")
            coord = default_axes[0]

        panel_data = self.get_panel_data(panel, data)
        x = panel_data.coords[coord].pint.magnitude
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

