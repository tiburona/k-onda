import itertools
from dataclasses import dataclass, field, replace

from matplotlib.patches import Patch

from .bars import bar_kwargs_from_props
from .core import PLOT_TYPE_TO_DEFAULTS, PlotDirective, replace_plot_node
from .utils import candidate_matches_selector


class LegendMixin:

    def legend(self, *args, position="upper right", title="", **kwargs):
        if not args:
            infer_entries = True
            legend_entries = []
        else:
            if len(args) > 1:
                raise ValueError("`legend` accepts a maximum of one positional arg")
            if not isinstance(args[0], dict):
                raise ValueError(
                    "Positional arg passed to `legend` must be of type `dict`."
                )
            infer_entries = False
            legend_entries = self.construct_legend_entries_from_structured_input(
                args[0]
            )

        legend_spec = Legend(
            entries=legend_entries,
            title=title,
            position=position,
            infer_entries=infer_entries,
            kwargs=kwargs,
        )
        return AddLegend(legend_spec=legend_spec)(self)

    def construct_legend_entries_from_structured_input(self, entries):
        return [
            LegendEntry(
                label=label,
                selector={
                    key: value
                    for key, value in values.items()
                    if key != "traits"
                },
                traits=values.get("traits"),
            )
            for label, values in entries.items()
        ]


@dataclass
class LegendEntry:
    label: str
    selector: dict = field(default_factory=dict)
    traits: list[str] | None = None
    props: dict | None = None


@dataclass
class Legend:
    entries: list[LegendEntry] = field(default_factory=list)
    infer_entries: bool = False
    position: str = "upper right"
    title: str | None = None
    kwargs: dict = field(default_factory=dict)


class AddLegend(PlotDirective):
    def __init__(self, legend_spec):
        self.legend_spec = legend_spec

    def direct(self, input):
        return replace_plot_node(input, legend_spec=self.legend_spec)


class LegendResolver:

    def resolve(self, input):
        legend_spec = input.legend_spec

        if legend_spec.infer_entries:
            entries = self.infer_legend_entries(input)
        else:
            entries = self.resolve_legend_entries(input)

        return replace(legend_spec, entries=entries)

    def resolve_style(self, selector, style_rules, defaults=None):
        defaults = defaults or {}
        props = {**defaults}
        for rule in style_rules:
            if candidate_matches_selector(rule.selector, selector):
                props = {**props, **rule.props}
        return props
    
    def infer_legend_entries(self, input):
        # Combined legend inference:
        #
        # 1. Find the condition coordinates referenced by StyleRule selectors.
        #    Coordinates not referenced by any StyleRule do not affect the legend.
        #
        # 2. Get every level of each relevant coordinate from the schema, including
        #    levels that have no explicit StyleRule and therefore use defaults.
        #
        # 3. Generate the Cartesian product of those level domains. Each combination
        #    becomes a concrete candidate selector.
        #
        # 4. For each candidate selector, resolve its effective props by starting with
        #    the plot-type defaults and applying every matching StyleRule in order.
        #
        # 5. Generate one label from the complete selector combination.
        #
        # 6. Construct one LegendEntry from each selector, label, and resolved props.
        if not input.style_rules:
            raise ValueError("You have called `legend` with no arguments, but have added no styles " \
            "from which to build a legend")
        style_rules = input.style_rules
        data_schema = input.data_source.data_schema
        entries = []

        coordinates = list(
            dict.fromkeys([k for rule in style_rules for k in rule.selector.keys()])
            )

        if not coordinates:
            raise ValueError(
                "Cannot infer legend entries because no style rule refers "
                "to a condition coordinate."
            )

        for coord_name in coordinates:
            coord = data_schema.coord_by_name(coord_name)
            if  not coord.is_condition:
                raise ValueError(f"You've attached a style rule to coordinate {coord_name}, which" 
                    "is not a condition, and are inferring a legend. Style rules"
                    "should only be attached to condition coordinates.")
            if  coord.levels is None:
                raise ValueError(f"You've attached a style rule to coordinate {coord_name} without" 
                    "levels, and are inferring a legend.  Style rules should only be " 
                    "attached to condition coordinates with levels.")

        coord_names_to_levels = data_schema.coord_name_levels_map(coordinates)
        coordinates = list(coord_names_to_levels)
        levels = coord_names_to_levels.values()

        selectors = [
            dict(zip(coordinates, combo)) for combo in itertools.product(*levels)
        ]

        defaults = PLOT_TYPE_TO_DEFAULTS[input.plot_type]

        for combination in selectors:
            props = self.resolve_style(combination, style_rules, defaults)
            label = " ".join(combination.values())
            entries.append(LegendEntry(label=label, props=props, selector=combination))

        return entries

    def resolve_legend_entries(self, input):
        style_rules = input.style_rules or []
        defaults = PLOT_TYPE_TO_DEFAULTS[input.plot_type]
        
        entries = input.legend_spec.entries
        resolved_entries = []
        for entry in entries:
            if entry.props is None:
                props = self.resolve_style(entry.selector, style_rules, defaults=defaults)
                if entry.traits is not None:
                    props = {k: v for k, v in props.items() if k in entry.traits}
                    resolved_entry = replace(entry, props=props)
                else:
                    resolved_entry = replace(entry, traits=list(props), props=props)
            else:
                resolved_entry = entry
            resolved_entries.append(resolved_entry)

        return resolved_entries


class LegendRenderer:

    def render(self, legend_spec, fig):
        entries = legend_spec.entries
        handles = [
            Patch(**bar_kwargs_from_props(entry.props))
            for entry in entries
        ]
        labels = [entry.label for entry in entries]
        fig.legend(
            handles=handles,
            labels=labels,
            title=legend_spec.title,
            loc=legend_spec.position,
            **legend_spec.kwargs,
        )
