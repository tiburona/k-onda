from collections.abc import Callable, Iterable, Sequence

import pint

from k_onda.transformers import KMeans, ExtractFeatures
from k_onda.central import operations
from k_onda.sources import Collection
from k_onda.utils import ValidationMixin


class Classify(ValidationMixin):
    rule_fields = {
        "classifier": {"type", "order", "sort_by", "labels"},
        "threshold": {
            "type",
            "feature",
            "operator",
            "value",
            "from_computed_features",
            "unit",
            "label",
        },
        "default": {"type", "value"},
    }
    required_rule_fields = {
        "classifier": {"type", "sort_by", "labels"},
        "threshold": {"type", "feature", "operator", "value", "label"},
        "default": {"type", "value"},
    }

    def __init__(
        self,
        label_name: str,
        *,
        spec: dict[str, object] | None = None,
        func: Callable[..., Iterable[object]] | None = None,
        order: str | None = None,
        sort_by: str | None = None,
        labels: Sequence[object] | None = None,
    ):
        self.validate_type_hints()
        self._validate_configuration(
            label_name, spec, func, order, sort_by, labels
        )
        self.label_name = label_name
        self.spec = spec
        self.func = func
        if spec is None and func is None:
            self.spec = {
                "rules": [
                    {
                        "type": "classifier",
                        "order": order or "ascending",
                        "sort_by": sort_by,
                        "labels": labels
                    }
                ]
            }

    def _validate_configuration(
        self, label_name, spec, func, order, sort_by, labels
    ):
        self.validate_parameter("label_name", label_name, nonempty_string=True)
        if order is not None:
            self.validate_parameter(
                "order", order, choices=("ascending", "descending")
            )
        if sort_by is not None:
            self.validate_parameter("sort_by", sort_by, nonempty_string=True)
        if labels is not None:
            if isinstance(labels, (str, bytes)):
                raise TypeError(
                    f"{self.format_call()}: labels must be a sequence of labels, "
                    "not a string."
                )
            self.validate_parameter("labels", labels, nonempty=True)

        if spec is not None:
            conflicts = [
                name
                for name, value in (
                    ("func", func),
                    ("order", order),
                    ("sort_by", sort_by),
                    ("labels", labels),
                )
                if value is not None
            ]
            if conflicts:
                raise ValueError(
                    f"{self.format_call()}: pass spec or "
                    f"{', '.join(conflicts)}, not both."
                )
            self._validate_spec(spec)
        elif func is not None:
            conflicts = [
                name
                for name, value in (
                    ("order", order),
                    ("sort_by", sort_by),
                    ("labels", labels),
                )
                if value is not None
            ]
            if conflicts:
                raise ValueError(
                    f"{self.format_call()}: pass func or "
                    f"{', '.join(conflicts)}, not both."
                )
        else:
            if sort_by is None or labels is None:
                raise ValueError(
                    f"{self.format_call()}: provide spec, func, or both sort_by "
                    "and labels."
                )

    def _validate_spec(self, spec):
        unknown_fields = set(spec) - {"rules"}
        if unknown_fields:
            raise ValueError(
                f"{self.format_call()}: unknown spec fields "
                f"{sorted(unknown_fields)!r}."
            )
        if "rules" not in spec:
            raise ValueError(f"{self.format_call()}: spec requires rules.")

        rules = spec["rules"]
        if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
            raise TypeError(
                f"{self.format_call()}: spec rules must be a sequence of "
                "dictionaries."
            )
        if not rules:
            raise ValueError(f"{self.format_call()}: spec rules cannot be empty.")

        for index, rule in enumerate(rules):
            self._validate_rule(rule, index)

    def _validate_rule(self, rule, index):
        if not isinstance(rule, dict):
            raise TypeError(
                f"{self.format_call()}: rule {index} must be a dictionary."
            )
        self.validate_string_iterable(f"rule {index} field", rule)
        rule_type = rule.get("type")
        if not isinstance(rule_type, str):
            raise TypeError(
                f"{self.format_call()}: rule {index} type must be a string."
            )
        if rule_type not in self.rule_fields:
            raise ValueError(
                f"{self.format_call()}: rule {index} has unknown type "
                f"{rule_type!r}."
            )

        unknown_fields = set(rule) - self.rule_fields[rule_type]
        if unknown_fields:
            raise ValueError(
                f"{self.format_call()}: rule {index} contains unknown fields "
                f"{sorted(unknown_fields)!r}."
            )
        missing_fields = self.required_rule_fields[rule_type] - set(rule)
        if missing_fields:
            raise ValueError(
                f"{self.format_call()}: rule {index} requires fields "
                f"{sorted(missing_fields)!r}."
            )

        if rule_type == "classifier":
            self._validate_classifier_rule(rule, index)
        elif rule_type == "threshold":
            self._validate_threshold_rule(rule, index)
        elif index != 0:
            raise ValueError(
                f"{self.format_call()}: a default rule must be first."
            )

    def _validate_classifier_rule(self, rule, index):
        sort_by = rule["sort_by"]
        if not isinstance(sort_by, str):
            raise TypeError(
                f"{self.format_call()}: rule {index} sort_by must be a string."
            )
        if not sort_by.strip():
            raise ValueError(
                f"{self.format_call()}: rule {index} sort_by cannot be empty."
            )

        labels = rule["labels"]
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise TypeError(
                f"{self.format_call()}: rule {index} labels must be a sequence."
            )
        if not labels:
            raise ValueError(
                f"{self.format_call()}: rule {index} labels cannot be empty."
            )

        order = rule.get("order", "ascending")
        if order not in {"ascending", "descending"}:
            raise ValueError(
                f"{self.format_call()}: rule {index} order must be 'ascending' "
                "or 'descending'."
            )

    def _validate_threshold_rule(self, rule, index):
        feature = rule["feature"]
        if not isinstance(feature, str):
            raise TypeError(
                f"{self.format_call()}: rule {index} feature must be a string."
            )
        if not feature.strip():
            raise ValueError(
                f"{self.format_call()}: rule {index} feature cannot be empty."
            )
        operator = rule["operator"]
        if not isinstance(operator, str):
            raise TypeError(
                f"{self.format_call()}: rule {index} operator must be a string."
            )
        if operator not in operations:
            raise ValueError(
                f"{self.format_call()}: rule {index} has unknown operator "
                f"{operator!r}."
            )
        if "from_computed_features" in rule and not isinstance(
            rule["from_computed_features"], bool
        ):
            raise TypeError(
                f"{self.format_call()}: rule {index} from_computed_features must "
                "be a boolean."
            )
        if "unit" in rule:
            if not isinstance(rule["unit"], str):
                raise TypeError(
                    f"{self.format_call()}: rule {index} unit must be a string."
                )
            if not rule["unit"].strip():
                raise ValueError(
                    f"{self.format_call()}: rule {index} unit cannot be empty."
                )

    def _validate_input(self, *inputs):
        from k_onda.signals import IndexedSignal

        if not inputs:
            raise ValueError(
                f"{self.format_call()}: provide at least one IndexedSignal."
            )
        for input in inputs:
            if not isinstance(input, IndexedSignal):
                raise TypeError(
                    f"{self.format_call()}: every input must be an IndexedSignal."
                )

    def __call__(self, *inputs):
        self._validate_input(*inputs)

        if self.spec:
            labeled_entities = self._parse_spec(*inputs)
        else:
            labeled_entities = self.func(*inputs)

        return Collection(labeled_entities)

    def _parse_spec(self, *chain):
        first_data = chain[0].data
        if "index" not in first_data.coords:
            raise ValueError(
                f"{self.format_call()}: the classified signal requires an index "
                "coordinate containing the entities to label."
            )
        entities_to_label = first_data.coords["index"].values

        # spec["rules"] is a list of dictionaries. each item in the list is a rule
        # each successive rule is allowed to override the next rule

        for rule in self.spec["rules"]:
            spec_type = rule["type"]  # threshold | classifier | default
            if spec_type == "classifier":
                # example classifier specification:
                # feature: 'fwhm'
                # order: 'ascending'  # ascending | descending
                # labels: ['IN', 'PN']  # in order of their value on the selected feature
                sort_by = rule["sort_by"]
                order = rule.get("order", "ascending")
                labels = rule["labels"]

                classification = self._find_transformer_output(chain, KMeans)

                # centers is a n_clusters, n_features numpy array
                centers = classification.data.attrs["kmeans_centers"]
                feature_names = classification.data.attrs["kmeans_feature_names"]

                if sort_by not in feature_names:
                    raise ValueError(
                        f"{self.format_call()}: classifier rule sort_by={sort_by!r} "
                        f"is not among the clustered features {feature_names!r}."
                    )
                if len(labels) != len(centers):
                    raise ValueError(
                        f"{self.format_call()}: classifier rule supplies "
                        f"{len(labels)} labels for {len(centers)} clusters."
                    )

                # the column ind of the feature that determines the label, e.g. 'fwhm'
                feature_ind = feature_names.index(sort_by)

                # sort row indexes on the value of the center at the selected feature.
                center_inds, feature_vals = (
                    list(range(len(centers))),
                    centers[:, feature_ind],
                )
                sorted_inds_and_vals = sorted(
                    zip(center_inds, feature_vals), key=lambda x: x[1]
                )
                if order != "ascending":
                    sorted_inds_and_vals = reversed(sorted_inds_and_vals)

                # map the sorted indices to the sorted labels
                inds_and_labels = {}
                for i, (ind, _) in enumerate(sorted_inds_and_vals):
                    inds_and_labels[ind] = labels[i]

                # assign labels to the entity in the index
                for i, entity in enumerate(entities_to_label):
                    int_label = int(classification.data.values[i])
                    label = inds_and_labels[int_label]
                    setattr(entity, self.label_name, label)
                    entity.set_annotation(
                        self.label_name,
                        label,
                        annotator=self,
                        source_signal=classification,
                    )

            elif spec_type == "threshold":
                # example threshold specification

                #     - feature: fwhm
                #       operator: '>'
                #       value: 400
                #       from_computed_features: True
                #       unit: us
                #       label: PN
                feature = rule["feature"]
                operator = rule["operator"]
                value = rule["value"]
                from_computed_features = rule.get("from_computed_features")
                unit = rule.get("unit")
                if unit:
                    value *= pint.application_registry(unit)
                label = rule["label"]

                feature_set = None
                if from_computed_features:
                    # value comes from a previously computed output of ExtractFeatures transformer
                    feature_set = self._find_transformer_output(
                        chain, ExtractFeatures
                    )
                    feature_names = feature_set.data.coords["feature"].values
                    if feature not in feature_names:
                        raise ValueError(
                            f"{self.format_call()}: threshold rule feature "
                            f"{feature!r} is not among the computed features "
                            f"{feature_names.tolist()!r}."
                        )
                    feature_vals = feature_set.data.sel(feature=feature)

                else:
                    # default: we'll look for the feature on the entity
                    feature_vals = [
                        getattr(entity, feature) for entity in entities_to_label
                    ]

                test_func = operations[operator]
                for entity, fval in zip(entities_to_label, feature_vals):
                    if test_func(fval, value):
                        setattr(entity, self.label_name, label)
                        entity.set_annotation(
                            self.label_name,
                            label,
                            annotator=self,
                            source_signal=feature_set,
                        )

            elif spec_type == "default":
                # example default specification
                # value: PN

                value = rule["value"]

                for entity in entities_to_label:
                    setattr(entity, self.label_name, value)
                    entity.set_annotation(
                        self.label_name, value, annotator=self, source_signal=None
                    )

        return entities_to_label

    def _find_transformer_output(self, chain, transformer_type):
        for node in chain:
            if isinstance(node.transformer, transformer_type):
                return node
        raise ValueError(
            f"{self.format_call()}: a rule requires output from "
            f"{transformer_type.__name__}, but the classified signal's chain "
            "does not contain it."
        )
