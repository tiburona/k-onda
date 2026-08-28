from dataclasses import dataclass, field, replace

from k_onda.utils import (
    is_unitful,
    validate_types,
    ValidationMixin,
)

from .core import PlotDirective, replace_plot_node
from .utils import UNSET, UnsetType


class LabelMixin:
    def label(self):
        raise NotImplementedError("Default label method not yet implemented.")

    def _normalize_label_input(
        self,
        method_name,
        location,
        value,
        *,
        fixed,
        shared,
    ):
        shared_values = {
            name: default if supplied is UNSET else supplied
            for name, (supplied, default) in shared.items()
        }
        if isinstance(value, str):
            return shared_values | {"text": value} | fixed

        if not value:
            raise ValueError(
                f"{method_name}(): the {location} label specification cannot be "
                "empty."
            )

        allowed_fields = {"text", "kwargs", *shared}
        unknown_fields = set(value) - allowed_fields
        if unknown_fields:
            raise ValueError(
                f"{method_name}(): unknown fields in the {location} label "
                f"specification: {sorted(unknown_fields)!r}."
            )
        if "text" not in value:
            raise ValueError(
                f"{method_name}(): the {location} label specification requires "
                "text."
            )

        conflicts = {
            name
            for name, (supplied, _) in shared.items()
            if supplied is not UNSET and name in value
        }
        if conflicts:
            raise ValueError(
                f"{method_name}(): pass {sorted(conflicts)!r} either as shared "
                "keywords or inside a label specification, not both."
            )

        return shared_values | value | fixed

    @validate_types
    def plot_labels(
        self,
        *,
        x: str | dict[str, object] | None = None,
        y: str | dict[str, object] | None = None,
        units: str | None | UnsetType = UNSET,
    ):
        if x is None and y is None:
            raise ValueError("plot_labels(): provide an x or y label.")

        labels = []

        for axis_value, axis_name in zip((x, y), ("x", "y")):
            if axis_value is None:
                continue
            params = self._normalize_label_input(
                "plot_labels",
                axis_name,
                axis_value,
                fixed={"scope": "figure", "axis": axis_name},
                shared={"units": (units, "auto")},
            )
            labels.append(Label(**params))
                
        node = self
        for label in labels:
            node = AddLabel(label)(node)

        return node
    
    @validate_types
    def panel_labels(
        self,
        *,
        top: str | dict[str, object] | None = None,
        left: str | dict[str, object] | None = None,
        right: str | dict[str, object] | None = None,
        bottom: str | dict[str, object] | None = None,
        where: str | UnsetType = UNSET,
        units: str | None | UnsetType = UNSET,
    ):
        if all(value is None for value in (top, left, right, bottom)):
            raise ValueError(
                "panel_labels(): provide a top, left, right, or bottom label."
            )

        labels = []
        side_values = (top, left, right, bottom)
        side_names = ("top", "left", "right", "bottom")
        for side_value, side_name in zip(side_values, side_names):
            if side_value is None:
                continue
            params = self._normalize_label_input(
                "panel_labels",
                side_name,
                side_value,
                fixed={"scope": "panel", "side": side_name},
                shared={"where": (where, "all"), "units": (units, None)},
            )
            labels.append(Label(**params))
                
        node = self
        for label in labels:
            node = AddLabel(label)(node)
            
        return node


@dataclass
class Label(ValidationMixin):
    text: str
    scope: str = "figure"
    axis: str | None = None
    side: str | None = None
    where: str = "all"
    units: str | None = "auto"
    kwargs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self.validate_type_hints()
        self.validate_parameter("text", self.text, nonempty_string=True)
        self.validate_parameter(
            "scope", self.scope, choices=("figure", "panel")
        )
        self.validate_parameter(
            "axis", self.axis, choices=("x", "y"), allow_none=True
        )
        self.validate_parameter(
            "side",
            self.side,
            choices=("top", "left", "right", "bottom"),
            allow_none=True,
        )
        self.validate_parameter(
            "where",
            self.where,
            choices=("all", "top", "left", "right", "bottom"),
        )
        if self.units not in (None, "auto"):
            raise NotImplementedError(
                f"{self.format_call()}: explicit label-unit formatting is not "
                "implemented; use 'auto' or None."
            )
        if self.scope == "figure":
            if self.axis is None:
                raise ValueError(
                    f"{self.format_call()}: a figure label requires axis='x' or "
                    "axis='y'."
                )
            if self.side is not None:
                raise NotImplementedError(
                    f"{self.format_call()}: positioning a figure label by side is "
                    "not yet implemented; use axis='x' or axis='y'."
                )
            if self.where != "all":
                raise ValueError(
                    f"{self.format_call()}: where applies only to panel labels."
                )
        else:
            if self.side is None:
                raise ValueError(
                    f"{self.format_call()}: a panel label requires side."
                )
            if self.axis is not None:
                raise ValueError(
                    f"{self.format_call()}: a panel label cannot specify axis."
                )


@dataclass
class LabelPlan:
    explicit: list[Label] = field(default_factory=list)
    infer_missing: bool = False


class AddLabel(PlotDirective):

    def __init__(self, label: Label | None, infer_missing: bool = False):
        self._validate_configuration(label, infer_missing)

        self.label = label 
        self.infer_missing = infer_missing

    def _validate_configuration(self, label, infer_missing):
        self.validate_type_hints()
        if label is None and not infer_missing:
            raise ValueError(
                f"{self.format_call()}: label cannot be None unless infer_missing "
                "is True."
            )

    def direct(self, input):
        self.validate_label(input)

        if input.label_plan is None:
            explicit = [self.label] if self.label is not None else []
            infer_missing = self.infer_missing
        else:
            if self.label:
                explicit = [*input.label_plan.explicit, self.label]
            else:
                explicit = input.label_plan.explicit
            infer_missing = input.label_plan.infer_missing or self.infer_missing
        
        label_plan = LabelPlan(explicit=explicit, infer_missing=infer_missing)
        
        return replace_plot_node(input, label_plan=label_plan)

    def validate_label(self, input):
        if input.label_plan is None or self.label is None:
            return
        existing_labels = input.label_plan.explicit or []
        if self.label.scope == "figure":
            self.check_figure_labels(existing_labels)
        else:
            self.check_panel_labels(existing_labels)

    def check_figure_labels(self, existing_labels):
        for label in existing_labels:
            if label.axis == "x" and self.label.axis == "x":
                raise NotImplementedError(
                    f"{self.format_call()}: multiple x-axis figure labels are not "
                    "yet implemented."
                )
            elif label.axis == "y" and self.label.axis == "y":
                raise NotImplementedError(
                    f"{self.format_call()}: multiple y-axis figure labels are not "
                    "yet implemented."
                )
            
    def check_panel_labels(self, existing_labels):
        panel_labels = [el for el in existing_labels if el.scope == "panel"]
        if any([self._is_conflicting(panel_label, self.label) for panel_label in panel_labels]):
            raise NotImplementedError(
                f"{self.format_call()}: multiple panel labels at the same placement "
                "are not yet implemented."
            )

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


class LabelResolver:
    
    def resolve(self, input, role_source_map, layout_spec):
       
        if input.label_plan.infer_missing:
            label_plan = self.infer_missing_labels(input, role_source_map, layout_spec)
        else:
            label_plan = input.label_plan

        return label_plan

    def infer_missing_labels(self, input,  role_source_map, layout_spec):
        label_plan = self.ensure_axis_labels(input.label_plan, role_source_map)
        label_plan = self.ensure_panel_labels(layout_spec, label_plan)
        return label_plan

    def ensure_axis_labels(self, label_plan, role_source_map):
        labels = list(label_plan.explicit)
        for axis in ("x", "y"):
            axis_label = [label for label in label_plan.explicit if label.axis == axis]
            if not axis_label:
                labels.append(Label(role_source_map[axis].name, axis= axis))

        return replace(label_plan, explicit=labels)
    
    def ensure_panel_labels(self, layout, label_plan):

        labels = list(label_plan.explicit)

        panel_labels_exist = bool([
            label for label in label_plan.explicit if label.scope == "panel"
            ])
        
        if not panel_labels_exist:
            panels = layout.flat_panels
            coord_sets = {set(panel.coords) for panel in panels}
            
            if len(coord_sets) != 1:
                raise ValueError(
                    "Automatic panel labels require all panels to use the same "
                    "condition coordinates. Use panel_labels() explicitly."
                    )
            
            labels.append(
                Label(
                    text = " ".join(f"{{{coord_name}}}" for coord_name in panels[0].coords),
                    scope="panel",
                    side="bottom",
                    where="all",
                    units=None
                    )
            )

        return replace(label_plan, explicit=labels)


class LabelRenderer:

    def render(self, layout, label_plan, fig, data, panel_ax_map, role_source_map):
        labels = label_plan.explicit or []
        for label in labels:
            if label.scope == "figure":
                self.add_figure_label(label, data, fig, role_source_map)
            elif label.scope == "panel":
                self.add_panel_label(
                    layout,
                    label,
                    data,
                    panel_ax_map,
                    role_source_map,
                )

    def add_figure_label(self, label, data, fig, role_source_map):
        text = self.resolve_label_text(label, data, role_source_map)
        if label.axis == "x":
            fig.supxlabel(text, **label.kwargs)
        elif label.axis == "y":
            fig.supylabel(text, **label.kwargs)

    def add_panel_label(
        self,
        layout,
        label,
        data,
        panel_ax_map,
        role_source_map,
    ):
        panels_to_label = self.select_panels_to_label(layout, label.where)
        if label.side in ("bottom", "top"):
            for panel in panels_to_label:
                ax = panel_ax_map[(panel.row, panel.col)]
                text = self.resolve_label_text(
                    label,
                    data,
                    role_source_map,
                    panel=panel,
                )
                ax.set_xlabel(text, **label.kwargs)
                ax.xaxis.set_label_position(label.side)
        if label.side in ("left", "right"):
            for panel in panels_to_label:
                ax = panel_ax_map[(panel.row, panel.col)]
                text = self.resolve_label_text(
                    label,
                    data,
                    role_source_map,
                    panel=panel,
                )
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
            elif is_unitful(data):
                units = data.pint.units
            else:
                units = ""

            if units:
                text += f" ({units})"

        return text

    def select_panels_to_label(self, layout, where):
        panels = layout.panels
        flat_panels = layout.flat_panels
        if where == "all":
            return flat_panels
        if where == "top":
            return panels[0]
        if where == "bottom":
            return panels[-1]
        if where == "left":
            return [panel for row in panels for panel in row if panel.col == 0]
        if where == "right":
            return [
                panel
                for row in panels
                for panel in row
                if panel.col == len(row) - 1
            ]
