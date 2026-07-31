from dataclasses import dataclass, field
from collections import defaultdict

from .core import PlotDirective, replace_plot_node


class StyleMixin:
    def style(self, rules):
        
        style_rules = []
        for rule in rules:
            selector = {}
            props = {}
            for key in rule:
                if key in self.data_source.data_schema.coord_names:
                    selector[key] = rule[key]
                else:
                    props[key] = rule[key]
            style_rule = StyleRule(selector=selector, props=props)
            style_rules.append(style_rule)
            
        node = self
        for rule in style_rules:
            node = AddStyle(style_rule=rule)(node)

        return node
        
    def colors(self, rules=None, **kwargs):
        data_schema = self.data_source.data_schema
        # possible keys in kwargs
        # "control|defeat"
        # "control&tone"
        # "control"

        rules = rules or []

        style_rules = [
            StyleRule(
                props={"color": rule["color"]}, 
                selector={k: v for k, v in rule.items() if k != "color"}
                )
            for rule in rules
        ]

        if kwargs:
            data_schema = self.data_source.data_schema
            style_rules.extend([self._parse_kwarg(kwargs, kwarg, data_schema) for kwarg in kwargs])
        
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
            raise ValueError(f"No coord level {level!r} found in data schema.")
        if len(matches) > 1:
            raise ValueError(
                f"Condition value {level!r} is ambiguous; found in coords {matches}."
            )
        return matches
    
    def _parse_kwarg(self, kwargs, kwarg, data_schema):
        if "|" in kwarg:
            if "&" in kwarg:
                raise ValueError("Use structured input format if you have | and & conditions")
            levels = kwarg.split("|")
            coord_level_map = self._get_coord_level_map(levels, data_schema)
            if len(coord_level_map) > 1:
                raise ValueError("You can only use | keywords for combination within a condition." \
                "For more complex combinations, use structured input.")
            rule = StyleRule(
                selector = coord_level_map,
                props = {"color": kwargs[kwarg]}
            )
        elif "&" in kwarg:
            levels = kwarg.split("&")
            coord_level_map = self._get_coord_level_map(levels, data_schema)
            if any(len(coord_level_map[key]) > 1 for key in coord_level_map):
                raise ValueError("You can only use & keywords for combination across conditions." \
                "For more complex combinations, use structured input.")

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
class StyleRule:
    selector: dict = field(default_factory=dict)
    props: dict = field(default_factory=dict)


class AddStyle(PlotDirective):
    def __init__(self, style_rule):
        self.style_rule = style_rule

    def direct(self, input):
        style_rules = input.style_rules or []
        return replace_plot_node(
            input,
            style_rules=[*style_rules, self.style_rule],
        )
