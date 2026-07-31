from .api import PlotMixin


class PlotNode(PlotMixin):

    def __init__(
        self,
        data_source=None,
        plot_type=None,
        layout_spec=None,
        coords=None,
        label_plan=None,
        style_rules=None,
        legend_spec=None,
    ):
        self.data_source = data_source
        self.plot_type = plot_type
        self.layout_spec = layout_spec
        self.coords = coords
        self.label_plan = label_plan
        self.style_rules = style_rules
        self.legend_spec = legend_spec
