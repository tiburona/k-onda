from collections import defaultdict
from operator import attrgetter
from pathlib import Path
import uuid
import xarray as xr
import numpy as np

from ..transformers.transformer_mixins import (
    CalculateMixin, StackMixin, SelectMixin, AggregateMixin, PointProcessMixin)
from ..signals import Signal
from k_onda.utils import DictDelegator


class DataSource:
    """A file or resource containing experimental data."""

    def __init__(self, session, data_loader_config):
        self.session = session
        self.data_loader_config = data_loader_config
        self.file_path = Path(self.data_loader_config["file_path"])
        self.file_ext = self.data_loader_config.get("file_ext")
        self._raw_data = None
        self.subject = self.session.subject


class DataComponent(CalculateMixin, SelectMixin):
    output_class = Signal

    def __init__(self, source):
        self.data_source = source
        self.subject = self.data_source.subject
        self._data = None
        self.data_identity = None
        self.start = self.data_source.session.start
        self.duration = self.data_source.session.duration

    @property
    def data(self):
        if self._data is None:
            self._data = self.data_loader()
        return self._data
    
    @property
    def data_dims(self):
        return self.data_schema.dims

    def assign_to_data_identity(self, data_identity):
        if data_identity is None:
            if self.data_identity is None:
                return
            self.data_identity.remove_data_component(self)
            return
        data_identity.add_data_components([self])


class DataIdentity:
    name = "identity"

    def __init__(self, data_components=None):
        self.uuid = uuid.uuid4()
        self.data_components = set()
        self.subject = None
        if data_components is not None:
            self.add_data_components(data_components)

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        return isinstance(other, DataIdentity) and other.uuid == self.uuid

    def add_data_components(self, data_components):
        for i, data_component in enumerate(data_components):
            if i == 0:
                if self.subject is None:
                    self.subject = data_component.subject
                    self.subject.data_identities[self.name].append(self)
            if data_component.subject is not self.subject:
                raise ValueError(
                    "data components on the same data identity must share a subject."
                )
            self.data_components.add(data_component)
            old_identity = data_component.data_identity
            if old_identity is not None:
                if old_identity is not self:
                    old_identity.remove_data_component(data_component)
            data_component.data_identity = self

    def remove_data_component(self, data_component):
        self.data_components.discard(data_component)
        if self in self.subject.data_identities[self.name]:
            self.subject.data_identities[self.name].remove(self)
        data_component.data_identity = None


class Collection(StackMixin, CalculateMixin, SelectMixin, AggregateMixin, PointProcessMixin):
    def __init__(self, members):
        self.members = members
        self._signals = None

    def __iter__(self):
        return iter(self.members)

    def __len__(self):
        return len(self.members)

    def __getitem__(self, idx):
        return self.members[idx]

    @property
    def signals(self):
        if self._signals is None:
            self._signals = self.collect_signals()
        return self._signals
    
    def collect_base_components(self):
        if isinstance(self.members[0], DataIdentity):
            components = [c for m in self.members for c in m.data_components]
        elif isinstance(self.members[0], Collection):
            components = [m for mem in self.members for m in mem.collect_base_components()]
        else:
            components = self.members

        return components

    def collect_signals(self):
        base_components = self.collect_base_components()

        # If components aren't yet signals they need to be made into them.
        if isinstance(base_components[0], DataComponent):
            signals = [
                component.output_class(
                    component,
                    transform=lambda x: x,
                    transformer=None,
                    origin=component,
                    data_schema=component.data_schema
                )
                for component in base_components
            ]

            return signals

        return base_components

    def group_by(self, group_on, strict=True):
        return CollectionMap(self.members, group_on, strict=strict)
    







from k_onda.transformers import feature_registry

class MapMixin(DictDelegator):
    
    def extract_features(self, *features, registry=feature_registry):
        from k_onda.transformers import ExtractFeatures
        return ExtractFeatures(*features, registry=registry)((self,))


class SignalMap(MapMixin):
    _delegate_attr = 'map'

    def __init__(self, map):
        self.map = map
        self._cache = None

    @property
    def data(self):
        return self._materialize()

    def _materialize(self):
        if self._cache is None:
            self.cache = {k: signal.data for k, signal in self.map.items()}
        return self.cache
    




class CollectionMap(CalculateMixin, SelectMixin, AggregateMixin, MapMixin):
    _delegate_attr = 'groups'

    def __init__(self, members=None, group_on=None, strict=True, groups=None):
        if groups is None:
            if members is None or group_on is None:
                raise ValueError("members and group_on are required.")
            self.group_on = group_on
            self.members = list(members)
            self.groups = self.map_groups(strict=strict)

        else:
            self.group_on = group_on or getattr(groups, "group_on", None)
            self.groups = {
                k: (v if isinstance(v, Collection) else Collection(v))
                for k, v in groups.items()
            }
            self.members = [m for coll in self.groups.values() for m in coll]

    def map_groups(self, strict=True):
        groups = defaultdict(list)

        if type(self.group_on) == str:
            grouping_func = self.build_grouping_func(self.group_on, strict=strict)
        elif callable(self.group_on):
            grouping_func = self.group_on
        else:
            raise ValueError("grouping_factor must be a string or callable")

        for member in self.members:
            groups[grouping_func(member)].append(member)

        return {key: Collection(val) for key, val in groups.items()}

    @staticmethod
    def build_grouping_func(grouping, strict=True):
        def grouping_func(entity):
            if hasattr(entity, grouping):
                return getattr(entity, grouping)
            elif hasattr(entity, 'data_identity') and getattr(entity.data_identity, 'name', None) == grouping:
                return entity.data_identity

            try:
                return attrgetter(grouping)(entity)
            except AttributeError as e:
                if strict:
                    raise e
                return None

        return grouping_func

    def as_collection(self):
        return Collection([member for collection in self.values() for member in collection])
    




