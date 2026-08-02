from dataclasses import dataclass


BAR_DEFAULT_PROPS = {
    "color": "#4C78A8",
}

PLOT_TYPE_TO_DEFAULTS = {
    "histogram": BAR_DEFAULT_PROPS,
    "time-histogram": BAR_DEFAULT_PROPS,
    "psth": BAR_DEFAULT_PROPS,
}


class PlotDirective:

    def __call__(self, input):
        return self.direct(input)


@dataclass(frozen=True)
class PlotSource:
    kind: str
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
    }
    values.update(changes)
    return new_plot_node(**values)
