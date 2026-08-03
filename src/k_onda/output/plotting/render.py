from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
from matplotlib.axes import Axes as MPLAxes
from matplotlib.figure import Figure

from .axes import AxisResolver
from .band import BandRenderer
from .bars import BarRenderer
from .core import PlotDirective
from .labels import LabelRenderer, LabelResolver
from .layout import LayoutResolver, Panel
from .axes import AxisRenderer
from .legend import LegendRenderer, LegendResolver

if TYPE_CHECKING:
    from .node import PlotNode

Cell = tuple[int, int]


class Render(PlotDirective):

    def __init__(
        self,
        axis_resolver=None,
        label_resolver=None,
        layout_resolver=None,
        legend_resolver=None,
        bar_renderer=None,
        axis_renderer=None,
        label_renderer=None,
        legend_renderer=None,
        band_renderer=None
    ):
        self.axis_resolver = axis_resolver or AxisResolver()
        self.label_resolver = label_resolver or LabelResolver()
        self.layout_resolver = layout_resolver or LayoutResolver()
        self.legend_resolver = legend_resolver or LegendResolver()
        self.bar_renderer = bar_renderer or BarRenderer()
        self.axis_renderer = axis_renderer or AxisRenderer()
        self.label_renderer = label_renderer or LabelRenderer()
        self.legend_renderer = legend_renderer or LegendRenderer()
        self.band_renderer = band_renderer or BandRenderer()

    def direct(self, input: PlotNode) -> Figure:
        figsize = getattr(input, "figsize", (8, 8))
        fig = plt.figure(figsize=figsize, layout="constrained")
        data = self.get_plot_data(input)
        layout = (
            self.layout_resolver.resolve(input)
            if input.layout_spec is None
            else input.layout_spec
        )
        style_rules = input.style_rules or []
        grid = fig.add_gridspec(layout.num_rows, layout.num_cols)
        role_source_map = self.axis_resolver.resolve_role(input, layout)

        panel_to_x_anchor, panel_to_y_anchor, panel_to_tick_args = self.axis_renderer.get_tick_args(
            input, layout
        )

        panel_mpl_axis_map = {}

        def get_share_anchor(panel_to_anchor: dict[Cell, Panel], panel: Panel
                             ) -> MPLAxes | None:
            # starting with panel and map to the anchor panels, return the matplotlib
            # axis object that will actually be passed to the rendering function (or None)
            anchor_panel = panel_to_anchor[(panel.row, panel.col)]
            if anchor_panel == panel:
                return None
           
            return panel_mpl_axis_map[(anchor_panel.row, anchor_panel.col)]

        band_plan = self.band_renderer.build_band_plan(input, layout)

        for panel in layout.flat_panels:
            share_x = get_share_anchor(panel_to_x_anchor, panel)
            share_y = get_share_anchor(panel_to_y_anchor, panel)
            ax = fig.add_subplot(grid[panel.row, panel.col], sharex=share_x, sharey=share_y)
            ax.tick_params(**panel_to_tick_args[(panel.row, panel.col)])
            # It's a little weird that this map is being constructed in the 
            # same loop where get_share_anchor uses it to get the anchor axis, 
            # but the anchor axis is guaranteed to already be in the dictionary
            panel_mpl_axis_map[(panel.row, panel.col)] = ax

            self.bar_renderer.render(
                input.plot_type, panel, ax, data, role_source_map, style_rules,
            )

            for band in band_plan[(panel.row, panel.col)]:
                self.band_renderer.make_band(band, ax)

        if input.label_plan:
            label_plan = self.label_resolver.resolve(input, role_source_map, layout)

            self.label_renderer.render(
                layout,
                label_plan,
                fig,
                data,
                panel_mpl_axis_map,
                role_source_map,
            )

        if input.legend_spec:
            legend_spec = self.legend_resolver.resolve(input)
            self.legend_renderer.render(legend_spec, fig)

        fig.show()
        return fig

    def get_plot_data(self, input):
        return input.data_source.compile().data
