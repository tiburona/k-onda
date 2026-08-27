from collections.abc import Iterable
from dataclasses import dataclass, field
from collections import defaultdict

from k_onda.utils import validate_nonempty, validate_types, ValidationMixin

from .core import PlotDirective, replace_plot_node, validate_plot_node


StyleRuleInput = dict[str, object]
StyleRulesInput = StyleRuleInput | Iterable[StyleRuleInput]


class StyleMixin:
    def _style_data_schema(self, method_name):
        validate_plot_node(self, f"{method_name}()")
        return self.data_source.data_schema

    def _normalize_style_rules(self, rules, method_name):
        if isinstance(rules, dict):
            rules = [rules]
        else:
            try:
                rules = list(rules)
            except TypeError:
                raise TypeError(
                    f"{method_name}(): rules must be a dictionary or an iterable "
                    "of dictionaries."
                ) from None

        validate_nonempty(method_name, rules=rules)
        if any(not isinstance(rule, dict) for rule in rules):
            raise TypeError(
                f"{method_name}(): every style rule must be a dictionary."
            )
        return rules

    @validate_types
    def style(self, rules: StyleRulesInput):
        rules = self._normalize_style_rules(rules, "style")
        data_schema = self._style_data_schema("style")

        style_rules = []
        for rule in rules:
            selector = {}
            props = {}
            for key in rule:
                if key in data_schema.coord_names:
                    selector[key] = rule[key]
                else:
                    props[key] = rule[key]
            style_rule = StyleRule(selector=selector, props=props)
            style_rules.append(style_rule)
            
        node = self
        for rule in style_rules:
            node = AddStyle(style_rule=rule)(node)

        return node
        
    @validate_types
    def colors(
        self,
        rules: StyleRulesInput | None = None,
        **kwargs: object,
    ):
        # possible keys in kwargs
        # "control|defeat"
        # "control&tone"
        # "control"

        if rules is not None and kwargs:
            raise ValueError(
                "colors(): pass either structured rules or keyword colors, not "
                "both."
            )

        if rules is None:
            rules = []
        else:
            rules = self._normalize_style_rules(rules, "colors")

        if not rules and not kwargs:
            raise ValueError("colors(): provide rules or keyword colors.")
        if any("color" not in rule for rule in rules):
            raise ValueError(
                "colors(): every structured rule must contain a 'color' property."
            )

        data_schema = self._style_data_schema("colors")

        style_rules = [
            StyleRule(
                props={"color": rule["color"]}, 
                selector={k: v for k, v in rule.items() if k != "color"}
                )
            for rule in rules
        ]

        if kwargs:
            style_rules.extend(
                [self._parse_kwarg(kwargs, kwarg, data_schema) for kwarg in kwargs]
            )
        
        node = self
        for rule in style_rules:
            node = AddStyle(style_rule=rule)(node)

        return node
    
    def _get_coord_level_map(self, levels, data_schema):
        coord_level_map = defaultdict(list)
        for level in levels:
            matches = self._get_coord_matches(level, data_schema)
            coord_level_map[matches[0]].append(level)
        return dict(coord_level_map)
    
    def _get_coord_matches(self, level, data_schema):
        matches = data_schema.coord_names_by_level(level)
        if len(matches) == 0:
            raise ValueError(
                f"colors(): no coordinate level {level!r} was found in the data "
                "schema."
            )
        if len(matches) > 1:
            raise ValueError(
                f"colors(): condition value {level!r} is ambiguous; it occurs in "
                f"coordinates {matches!r}. Use structured input."
            )
        return matches
    
    def _parse_kwarg(self, kwargs, kwarg, data_schema):
        if "|" in kwarg:
            if "&" in kwarg:
                raise ValueError(
                    "colors(): use structured input when combining both '|' and "
                    "'&' conditions."
                )
            levels = kwarg.split("|")
            coord_level_map = self._get_coord_level_map(levels, data_schema)
            if len(coord_level_map) > 1:
                raise ValueError(
                    "colors(): '|' can only combine levels within one condition. "
                    "Use structured input for more complex combinations."
                )
            rule = StyleRule(
                selector = coord_level_map,
                props = {"color": kwargs[kwarg]}
            )
        elif "&" in kwarg:
            levels = kwarg.split("&")
            coord_level_map = self._get_coord_level_map(levels, data_schema)
            if any(len(coord_level_map[key]) > 1 for key in coord_level_map):
                raise ValueError(
                    "colors(): '&' can only combine levels across conditions. "
                    "Use structured input for more complex combinations."
                )

            rule = StyleRule(
                selector = coord_level_map,
                props = {"color": kwargs[kwarg]}
            )
            
        else:
            matches = self._get_coord_matches(kwarg, data_schema)
            coord = matches[0]
            
            rule = StyleRule(
                selector={coord: kwarg},
                props = {"color": kwargs[kwarg]}
            )

        return rule


@dataclass
class StyleRule(ValidationMixin):
    selector: dict[str, object] = field(default_factory=dict)
    props: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self.validate_type_hints()
        self.validate_parameter("props", self.props, nonempty=True)
        self.validate_string_iterable("selector key", self.selector)
        self.validate_string_iterable("property name", self.props)


class AddStyle(PlotDirective):
    def __init__(self, style_rule: StyleRule):
        self.validate_type_hints()
        self.style_rule = style_rule

    def direct(self, input):
        self._validate_coord_names(
            input,
            self.style_rule.selector,
            parameter="selector",
        )

        style_rules = input.style_rules or []
        return replace_plot_node(
            input,
            style_rules=[*style_rules, self.style_rule],
        )
