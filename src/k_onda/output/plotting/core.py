from dataclasses import dataclass
from typing import Literal

from k_onda.utils import ValidationMixin


BAR_DEFAULT_PROPS = {
    "color": "#4C78A8",
}

PLOT_TYPE_TO_DEFAULTS = {
    "histogram": BAR_DEFAULT_PROPS,
    "time-histogram": BAR_DEFAULT_PROPS,
    "psth": BAR_DEFAULT_PROPS,
}


def validate_plot_node(input, call):
    from .node import PlotNode

    if not isinstance(input, PlotNode):
        raise TypeError(
            f"{call}: input must be a PlotNode; call plot() before applying a "
            "plot directive."
        )
    if input.data_source is None or input.plot_type is None:
        raise ValueError(
            f"{call}: input must be a PlotNode created by calling plot() on a "
            "Signal."
        )


class PlotDirective(ValidationMixin):

    def __call__(self, input):
        self._validate_input(input)
        return self.direct(input)

    def _validate_input(self, input):
        validate_plot_node(input, self.format_call())

    def _validate_coord_names(
        self,
        input,
        names,
        *,
        parameter,
        conditions_only=False,
    ):
        data_schema = input.data_source.data_schema
        valid_names = (
            data_schema.conditions_coord_names
            if conditions_only
            else data_schema.coord_names
        )
        unknown_names = set(names) - set(valid_names)
        if unknown_names:
            coord_kind = "condition coordinates" if conditions_only else "coordinates"
            raise ValueError(
                f"{self.format_call()}: unknown {coord_kind} in {parameter}: "
                f"{sorted(unknown_names)!r}."
            )


@dataclass(frozen=True)
class PlotSource:
    kind: Literal["coord", "values"]
    name: str | None = None


def new_plot_node(**kwargs):
    # Imported lazily so feature modules can depend on this module without
    # creating a cycle through the combined public API.
    from .node import PlotNode

    return PlotNode(**kwargs)


def replace_plot_node(input, **changes):
    values = {
        "data_source": input.data_source,
        "plot_type": input.plot_type,
        "layout_spec": input.layout_spec,
        "axes_spec": input.axes_spec,
        "coords": input.coords,
        "label_plan": input.label_plan,
        "style_rules": input.style_rules,
        "legend_spec": input.legend_spec,
        "overlays": input.overlays
    }
    values.update(changes)
    return new_plot_node(**values)
