from .core import PlotDirective, new_plot_node
from .labels import LabelMixin
from .layout import LayoutMixin
from .legend import LegendMixin
from .render import Render
from .style import StyleMixin


class PlotMixin(
    LayoutMixin,
    LabelMixin,
    StyleMixin,
    LegendMixin,
):

    def plot(self, plot_type):
        return SetPlotType(plot_type=plot_type)(self)

    def render(self):
        return Render()(self)


class SetPlotType(PlotDirective):

    def __init__(self, plot_type=None):
        self.plot_type = plot_type

    def direct(self, input):
        return new_plot_node(
            data_source=input,
            plot_type=self.plot_type,
        )
