from functools import reduce

import numpy as np

from .core import PLOT_TYPE_TO_DEFAULTS
from .utils import candidate_matches_selector


def bar_kwargs_from_props(props):
    kwargs = {
        key: value
        for key, value in props.items()
        if key not in ("pattern", "color")
    }
    if "pattern" in props:
        kwargs["hatch"] = props["pattern"]
    if "color" in props:
        kwargs["facecolor"] = props["color"]
    return kwargs


class BarRenderer:

    supported_plot_types = ("histogram", "time-histogram", "psth")

    def render(
        self,
        plot_type,
        panel,
        ax,
        data,
        role_source_map,
        style_rules,
    ):
        x_source = role_source_map["x"]
        panel_data = self.get_panel_data(panel, data)
        x = panel_data.coords[x_source.name].pint.magnitude
        y = panel_data.pint.magnitude
        width = np.median(np.diff(x))
        kwargs = self.get_merged_kwargs(plot_type, style_rules, panel_data)
        ax.bar(x, y, width=width, align="edge", **kwargs)
        ax.set_xlim(x[0], x[-1] + width)
        ax.margins(y=0.08)

    def get_merged_kwargs(self, plot_type, style_rules, panel_data):
        defaults = PLOT_TYPE_TO_DEFAULTS[plot_type]
        selected_style_rules = [
            rule
            for rule in style_rules
            if candidate_matches_selector(rule.selector, panel_data.coords)
        ]
        props = reduce(
            lambda accumulated, rule: {**accumulated, **rule.props},
            selected_style_rules,
            defaults,
        )
        return bar_kwargs_from_props(props)

    def get_panel_data(self, panel, data):
        mask = reduce(
            lambda left, right: left & right,
            [
                data.coords[coord_name] == level
                for coord_name, level in panel.coords.items()
            ],
        )
        return data.where(mask, drop=True).squeeze()
