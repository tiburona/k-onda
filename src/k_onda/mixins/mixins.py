from collections.abc import MutableMapping
import re
from functools import reduce
from k_onda.utils import recursive_update


class DictDelegator(MutableMapping):
    _delegate_attr: str  # subclass sets this

    def __getitem__(self, key):
        try:
            return getattr(self, self._delegate_attr)[key]
        except KeyError:
            if hasattr(self, "__missing__"):
                return self.__missing__(key)
            raise

    def __setitem__(self, key, value):
        getattr(self, self._delegate_attr)[key] = value

    def __delitem__(self, key):
        del getattr(self, self._delegate_attr)[key]

    def __iter__(self):
        return iter(getattr(self, self._delegate_attr))

    def __len__(self):
        return len(getattr(self, self._delegate_attr))


class ConfigSetter:
    def resolve_config(self, config, config_source):
        if isinstance(config, dict):
            return config
        if config is None:
            config = []
        config_to_return = {}
        for key in config:
            config_to_return[key] = self.resolve_inheritance(key, config_source)
        return config_to_return

    def resolve_inheritance(self, config_key, config_source):
        if not config_source:
            return {}
        config = config_source[config_key]
        configs = [config]
        seen = set()
        while "inherits" in config:
            parent_key = config["inherits"]
            if parent_key in seen:
                raise ValueError("There's a cycle in the config inheritance chain.")
            seen.add(parent_key)
            config = config_source[parent_key]
            configs.append(config)
        configs.reverse()
        return reduce(recursive_update, configs, {})

    def fill_fields(
        self, constructor, experiment=None, subject=None, session=None, **kwargs
    ):
        if not constructor:
            return

        constructor = str(constructor)
        if "{" not in constructor:
            return constructor

        else:
            constructor = {
                "template": constructor,
                "fields": re.findall(r"\{(.*?)\}", constructor),
            }

        for field in constructor["fields"]:
            for entity, entity_name in zip(
                [experiment, subject, session], ["experiment", "subject", "session"]
            ):
                if field.startswith(entity_name) and entity is None:
                    raise ValueError(f"{entity_name} was not supplied")

        return constructor["template"].format(
            session=session, subject=subject, experiment=experiment, **kwargs
        )
