from dataclasses import dataclass
from collections import defaultdict

from .core import PlotSource, PlotDirective, replace_plot_node


class AxisSharingMixin:

    def share_axes(self, x=None, y=None):

        for axis_val in (x, y):
            if (axis_val not in [True, False, None, "all", "row", "col"] and
                axis_val not in self.data_source.data_schema.condition_coord_names):
                raise ValueError(f"You have passed {axis_val} to share_axes but this is neither a "
                                 "known value for x or y or a condition coord on the data_schema.")

        if x is None and y is None:
            axis_sharing = AxisSharing(x="all", y="all")
            node = SetAxisSharing(axis_sharing)(self)
            return node

        def normalize(value):
            if value is True:
                return "all"
            if value in (False, None):
                return False
            return value

        axis_sharing = AxisSharing(x=normalize(x), y = normalize(y))

        node = SetAxisSharing(axis_sharing)(self)

        return node


@dataclass
class AxisSharing:
    x: str | bool = False
    y: str | bool = False


class SetAxisSharing(PlotDirective):
    def __init__(self, axis_sharing):
        self.axis_sharing = axis_sharing

    def direct(self, input):
        return replace_plot_node(input, axis_sharing=self.axis_sharing)


class AxisSharingResolver:

    def resolve(self, plot_node, layout):
        x_anchors = {}
        y_anchors = {}
        panel_to_x_anchor = defaultdict(lambda: None)
        panel_to_y_anchor = defaultdict(lambda: None)

        def panel_to_key(panel, spec):

            if spec == "all":
                key = "all"
            elif spec == "row":
                key = panel.row
            elif spec == "col":
                key = panel.col
            else:
                if spec not in panel.coords:
                    raise ValueError(
                        f"At least one panel does not have condition {spec}"
                        )
                key = panel.coords[spec]
            return key

        sharing = plot_node.axis_sharing

        for panel in layout.flat_panels:
            if sharing and sharing.x:
                x_key = panel_to_key(panel, plot_node.axis_sharing.x)
                anchor_panel = x_anchors.get(x_key)
                x_anchors.setdefault(x_key, panel)
                panel_to_x_anchor[(panel.row, panel.col)] = anchor_panel
            if sharing and sharing.y:
                y_key = panel_to_key(panel, plot_node.axis_sharing.y)
                anchor_panel = y_anchors.get(y_key)
                y_anchors.setdefault(y_key, panel)
                panel_to_y_anchor[(panel.row, panel.col)] = anchor_panel

        return panel_to_x_anchor, panel_to_y_anchor
            


class PlotRoleResolver:

    def resolve(self, plot_node, layout_spec):
        if plot_node.plot_type in ("histogram", "time-histogram", "psth"):
            return self.resolve_histogram(plot_node, layout_spec)
        raise NotImplementedError("Only histogram plots are currently implemented")

    def resolve_histogram(self, plot_node, layout_spec):
        return {
            "x": PlotSource(
                kind="coord",
                name=self.infer_x_source(plot_node, layout_spec),
            ),
            "y": PlotSource(kind="values"),
        }

    def infer_x_source(self, input, layout_spec):
        if input.coords:
            return next(iter(input.coords.values()))

        data_schema = input.data_source.data_schema
        panel_coord_names = set().union(
            *(panel.coords.keys() for panel in layout_spec.flat_panels)
        )
        default_axes = data_schema.axis_names_minus_axes_with_coords(
            panel_coord_names
        )

        if len(default_axes) != 1:
            raise ValueError("Unable to determine default axis.")
        return default_axes[0]
