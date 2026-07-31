import matplotlib.pyplot as plt

from .axes import PlotRoleResolver
from .bars import BarRenderer
from .core import PlotDirective
from .labels import LabelRenderer, LabelResolver
from .layout import LayoutResolver
from .legend import LegendRenderer, LegendResolver


class Render(PlotDirective):

    def __init__(
        self,
        role_resolver=None,
        label_resolver=None,
        layout_resolver=None,
        legend_resolver=None,
        bar_renderer=None,
        label_renderer=None,
        legend_renderer=None,
    ):
        self.role_resolver = role_resolver or PlotRoleResolver()
        self.label_resolver = label_resolver or LabelResolver()
        self.layout_resolver = layout_resolver or LayoutResolver()
        self.legend_resolver = legend_resolver or LegendResolver()
        self.bar_renderer = bar_renderer or BarRenderer()
        self.label_renderer = label_renderer or LabelRenderer()
        self.legend_renderer = legend_renderer or LegendRenderer()

    def direct(self, input):
        return self.make_figure(input)

    def make_figure(self, input):
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
        panel_ax_map = {}
        role_source_map = self.role_resolver.resolve(input, layout)

        for panel in layout.flat_panels:
            ax = fig.add_subplot(grid[panel.row, panel.col])
            panel_ax_map[(panel.row, panel.col)] = ax
            self.bar_renderer.render(
                input.plot_type,
                panel,
                ax,
                data,
                role_source_map,
                style_rules,
            )

        if input.label_plan:
            label_plan = self.label_resolver.resolve(
                input,
                role_source_map,
                layout,
            )
            self.label_renderer.render(
                layout,
                label_plan,
                fig,
                data,
                panel_ax_map,
                role_source_map,
            )

        if input.legend_spec:
            legend_spec = self.legend_resolver.resolve(input)
            self.legend_renderer.render(legend_spec, fig)

        fig.show()
        return fig

    def get_plot_data(self, input):
        return input.data_source.compile().data
