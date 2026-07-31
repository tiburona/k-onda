from dataclasses import dataclass, field, replace

from k_onda.utils import is_unitful

from .core import PlotDirective, replace_plot_node


class LabelMixin:
    def label(self):
        raise NotImplementedError("Default label method not yet implemented.")

    def plot_labels(self, x=None, y=None, units="auto"):
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
class Label:
    text: str
    scope: str = "figure"
    axis: str = None
    side: str = None
    where: str = "all"
    units: str = "auto"
    kwargs: dict = field(default_factory=dict)


@dataclass
class LabelPlan:
    explicit: list[Label] = field(default_factory=list)
    infer_missing: bool = False


class AddLabel(PlotDirective):

    def __init__(self, label, infer_missing=False):
        self.label = label 
        self.infer_missing = infer_missing
       

    def direct(self, input):
        self.validate_label(input)

        if input.label_plan is None:
            explicit = [self.label] or []
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
        if input.label_plan is None:
            return
        existing_labels = input.label_plan.explicit or []
        if self.label.scope == "figure":
            self.check_figure_labels(existing_labels)
        else:
            self.check_panel_labels(existing_labels)

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
            else:
                raise ValueError(f"Unknown value {label.scope} for label scope.")

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
        raise ValueError(f"Unknown value for where {where}")
