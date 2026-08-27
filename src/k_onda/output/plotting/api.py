from k_onda.central import type_registry as tr
from k_onda.utils import validate_types

from .core import PLOT_TYPE_TO_DEFAULTS, PlotDirective, new_plot_node
from .labels import LabelMixin
from .layout import LayoutMixin
from .axes import AxisMixin
from .legend import LegendMixin
from .render import Render
from .style import StyleMixin
from .band import BandMixin


class PlotMixin(
    LayoutMixin,
    AxisMixin,
    LabelMixin,
    StyleMixin,
    LegendMixin,
    BandMixin,
    ):

    @validate_types
    def plot(self, plot_type: str):
        return SetPlotType(plot_type=plot_type)(self)

    def render(self):
        return Render()(self)


class SetPlotType(PlotDirective):

    def __init__(self, plot_type: str):
        self._validate_configuration(plot_type)
        self.plot_type = plot_type

    def _validate_configuration(self, plot_type):
        self.validate_type_hints()
        self.validate_parameter("plot_type", plot_type, nonempty_string=True)
        if plot_type not in PLOT_TYPE_TO_DEFAULTS:
            supported = ", ".join(repr(name) for name in PLOT_TYPE_TO_DEFAULTS)
            raise NotImplementedError(
                f"{self.format_call()}: plot type {plot_type!r} is not implemented. "
                f"Supported plot types: {supported}."
            )

    def _validate_input(self, input):
        if not isinstance(input, tr.Signal):
            raise TypeError(
                f"{self.format_call()}: input must be a Signal, not "
                f"{type(input).__name__}."
            )

    def direct(self, input):
        return new_plot_node(
            data_source=input,
            plot_type=self.plot_type,
        )
