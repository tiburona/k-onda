from __future__ import annotations
from dataclasses import dataclass, replace, field
from collections.abc import Hashable
from typing import TYPE_CHECKING

from .core import PlotSource, PlotDirective, replace_plot_node
from .layout import Panel, Layout
from .utils import merge_dataclasses, UNSET

if TYPE_CHECKING:
    from .node import PlotNode


Cell = tuple[int, int]

class AxisMixin:

    def axes(
        self,
        share_x=UNSET,
        share_y=UNSET,
        x_ticks=UNSET,
        y_ticks=UNSET,
        x_tick_labels=UNSET,
        y_tick_labels=UNSET,
        x_spines=UNSET,
        y_spines=UNSET
    ):
        share_x, share_y = self.validate_and_normalize_share_args(share_x, share_y)

        return ConfigureAxes(
            x = AxisSpec(
                sharing=share_x,
                spine_visibility=x_spines,
                tick_visibility=x_ticks,
                tick_label_visibility=x_tick_labels
            ),
            y = AxisSpec(
                sharing=share_y,
                spine_visibility=y_spines,
                tick_visibility=y_ticks,
                tick_label_visibility=y_tick_labels
            )
        )(self)

    def validate_and_normalize_share_args(self, share_x, share_y):
        for axis_val in (share_x, share_y):
            if (axis_val not in [True, False, None, "all", "row", "col", UNSET] and
                axis_val not in self.data_source.data_schema.condition_coord_names):
                raise ValueError(
                    f"You have passed {axis_val} to share_axes but this is neither a "
                    "known value for x or y or a condition coord on the data_schema."
                    )
        def normalize(value):
            if value is True:
                return "all"
            if value in (False, None):
                return False
            return value

        return normalize(share_x), normalize(share_y)

    def share_axes(self, x="all", y="all"):
        share_x, share_y = self.validate_and_normalize_share_args(x, y)
        node = ConfigureAxes(x=AxisSpec(sharing=share_x), y=AxisSpec(sharing=share_y))(self)
        return node


@dataclass
class AxisSpec:
    sharing: str | bool | object = UNSET

    spine_visibility: str | bool | object = UNSET
    tick_visibility: str | bool | object = UNSET
    tick_label_visibility: str | bool | object = UNSET

    scale: str | None | object = UNSET
    limits: tuple | None | object = UNSET
    formatter: object | None = UNSET


@dataclass
class AxesSpec:
    x: AxisSpec = field(default_factory=AxisSpec)
    y: AxisSpec = field(default_factory=AxisSpec)





class ConfigureAxes(PlotDirective):
    def __init__(
        self, *, x: AxisSpec = AxisSpec(), y: AxisSpec = AxisSpec()):
            self.x = x
            self.y = y

    def direct(self, input):
        current = input.axes_spec or AxesSpec()
        updated = AxesSpec(
            x=merge_dataclasses(current.x, self.x),
            y=merge_dataclasses(current.y, self.y)
        )

        return replace_plot_node(input, axes_spec=updated)


class AxisRenderer:

    def resolve_config(self, axes_spec: AxesSpec) -> AxesSpec:
        keys = {"tick_visibility": True, "tick_label_visibility": "auto", "spine_visibility": True}
        
        def resolve(axis_str: str) -> AxisSpec:
            config = getattr(axes_spec, axis_str, AxisSpec())
            axis_config = replace(
                config, 
                **{k: keys[k] if getattr(config, k) == UNSET else getattr(config, k) for k in keys}
            )
            return axis_config

        return AxesSpec(x=resolve("x"), y=resolve("y")) 

    def get_tick_args(
            self, plot_node: PlotNode, layout: Layout
            ) -> tuple[dict[Cell, Panel], dict[Cell, Panel], dict[Cell, dict]]:

        axes_spec, panel_to_x_anchor, panel_to_y_anchor = self.build_sharing_plan(
            plot_node, layout
            )
        
        panel_to_tick_args = self.build_axis_plan(
            axes_spec, layout, panel_to_x_anchor, panel_to_y_anchor
            )

        return panel_to_x_anchor, panel_to_y_anchor, panel_to_tick_args

    def build_sharing_plan(
        self, 
        plot_node: PlotNode, 
        layout: Layout 
        ) -> tuple[AxesSpec, dict[Cell, Panel], dict[Cell, Panel]]:
        # A sharing plan, for both x and y axes, is a map of row, col panel coords
        # to the anchor panel for the sharing group of which the panel is a part.
        
        axes_spec = self.resolve_config(plot_node.axes_spec)
        
        def panel_to_key(panel: Panel, spec: str | bool | object) -> Hashable:

            if spec == "all":
                key = "all"
            elif spec == "row":
                key = panel.row
            elif spec == "col":
                key = panel.col
            elif spec in (None, False, UNSET):
                key = (panel.row, panel.col)
            else:
                if spec not in panel.coords:
                    raise ValueError(
                        f"At least one panel does not have condition {spec}"
                        )
                key = panel.coords[spec]
            return key

        def populate_panel_to_anchor(
            panel:Panel, 
            key_to_anchor: dict[Hashable, Panel], 
            panel_to_anchor: dict[Cell, Panel], 
            sharing: str
            ) -> None:
            # key can be row/col idx, False/None, or "all"
            key = panel_to_key(panel, sharing)
            # map keys to a single anchor panel
            key_to_anchor.setdefault(key, panel)
            # populate panel_to_anchor
            panel_to_anchor[(panel.row, panel.col)] = key_to_anchor[key]

        key_to_x_anchor: dict[Hashable, Panel] = {}
        key_to_y_anchor: dict[Hashable, Panel] = {}
        panel_to_x_anchor: dict[Cell, Panel] = {}
        panel_to_y_anchor: dict[Cell, Panel] = {}

        for panel in layout.flat_panels:
            populate_panel_to_anchor(
                panel, key_to_x_anchor, panel_to_x_anchor, axes_spec.x.sharing
                )

            populate_panel_to_anchor(
                panel, key_to_y_anchor, panel_to_y_anchor, axes_spec.y.sharing
            )
                
        return axes_spec, panel_to_x_anchor, panel_to_y_anchor

    def build_axis_plan(
            self, 
            axes_spec: AxesSpec, 
            layout: Layout, 
            panel_to_x_anchor: dict[Cell, Panel], 
            panel_to_y_anchor: dict[Cell, Panel]
            ) -> dict[Cell, dict]:
       
        # determine whether this panel is on the bottom of the share group
        def show_x_tick_labels(panel:Panel, share_group:list[Cell]) -> bool:
            return not any(
                other[1] == panel.col and other[0] > panel.row
                for other in share_group
                )

        # determine whether this panel is on the left of the share group
        def show_y_tick_labels(panel:Panel, share_group:list[Cell]) -> bool:
            return not any(
                other[0] == panel.row and other[1] < panel.col
                for other in share_group
            )

        # invert the panel to anchor map to get a map of anchor (row, col)s to a 
        # list of panel (row, col)s -- the share groups
        def make_share_groups(
                panel_to_anchor: dict[Cell, Panel]
                ) -> dict[Cell, list[Cell]]:
            groups = {}
            for panel, anchor in panel_to_anchor.items():
                groups.setdefault((anchor.row, anchor.col), []).append(panel)
            return groups

        # from the map of panel (row, col)s to anchor panels, and the map of 
        # anchor (row, col)s to share groups, get the share group of a panel
        def get_share_group(
            panel: Panel, panel_to_anchor: dict[Cell, Panel], groups: dict[Cell, list[Cell]]
            ) -> list[Cell]:
            anchor = panel_to_anchor[(panel.row, panel.col)]
            # groups maps anchor cells to a list of panels
            return groups[(anchor.row, anchor.col)]

        if axes_spec.x.tick_label_visibility == "auto":
            x_groups = make_share_groups(panel_to_x_anchor)
        if axes_spec.y.tick_label_visibility == "auto":
            y_groups = make_share_groups(panel_to_y_anchor)

        panel_to_tick_args = {}

        # for each of x and y, tick visibility is read from the axis config
        # if tick label visibility is "auto", then we get the group of panels with
        # which the panel axes, and compare the row and col indices to determine
        # whether this panel will have visible tick labels

        for panel in layout.flat_panels:
            args = {}

            args["bottom"] = axes_spec.x.tick_visibility

            if axes_spec.x.tick_label_visibility == "auto":
                share_group = get_share_group(panel, panel_to_x_anchor, x_groups)
                args["labelbottom"] = show_x_tick_labels(panel, share_group)

            else:
                args["labelbottom"] = axes_spec.x.tick_label_visibility

            args["left"] = axes_spec.y.tick_visibility

            if axes_spec.y.tick_label_visibility == "auto":
                share_group = get_share_group(panel, panel_to_y_anchor, y_groups)
                args["labelleft"] = show_y_tick_labels(panel, share_group)

            else:
                args["labelleft"] = axes_spec.y.tick_label_visibility

            panel_to_tick_args[(panel.row, panel.col)] = args

        return panel_to_tick_args


class AxisResolver:

    def resolve_role(self, plot_node, layout_spec):
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

     
      
    
