from collections import defaultdict
from operator import attrgetter
from pathlib import Path
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Any
from frozendict import frozendict


from ..transformers.transformer_mixins import (
    CalculateMixin, StackMixin, SelectMixin, AggregateMixin, PointProcessMixin)
from ..signals import Signal
from k_onda.utils import DictDelegator
from k_onda.transformers import feature_registry


@dataclass
class ProvenanceContext:
    """Immutable snapshopt of facts known when a signal entered the DAG."""
    component_id: str
    data_identity_id: Optional[str] = None
    data_identity_snapshot: Optional[frozendict] = None
    session_id: Optional[str] = None
    annotations: frozendict = field(default_factory=frozendict)


class DataSource:
    """A file or resource containing experimental data."""

    def __init__(self, session, data_loader_config, label=None):
        self.uid = uuid.uuid4()
        self.session = session
        self.data_loader_config = data_loader_config
        self.label = label
        self.file_path = Path(self.data_loader_config["file_path"])
        self.file_ext = self.data_loader_config.get("file_ext")
        self._raw_data = None
        self.subject = self.session.subject

    @property
    def display_id(self):
        parts = [self.session.display_id]
        if self.label:
            parts.append(self.label)
        elif self.identifiers:
            parts.extend(self.identifiers)
        else:
            parts.append(str(self.uid)[:8])
        return ":".join(parts)

    @property
    def identifiers(self):
        return []


class DataComponent(CalculateMixin, SelectMixin):
    output_class = Signal

    def __init__(self, source, data_identity=None):
        self.uid = uuid.uuid4()
        self.data_source = source
        self.data_identity = data_identity
        self.subject = self.data_source.subject
        self._data = None
        self.start = self.data_source.session.start
        self.duration = self.data_source.session.duration

    @property
    def display_id(self):
        parts = [self.data_source.session.display_id]
        if self.label:
            parts.append(self.label)
        elif self.identifiers:
            parts.extend(self.identifiers)
        else:
            parts.append(str(self.uid)[:8])
        return ":".join(parts)
    
    @property
    def identifiers(self):
        return []

    def data_loader(self):
        if self._data is None:
            self._data = self._data_loader()
        return self._data
    
    # subclasses override
    def _data_loader(self):
        pass
    
    @property
    def signal(self):
        return self.to_signal()

    @property
    def data_dims(self):
        return self.data_schema.dims
    
    def to_signal(self) -> Signal:
        context = ProvenanceContext(
            component_id=self.uid,
            session_id=self.data_source.session.uid,
            data_identity_id=self.data_identity.uid if self.data_identity else None,
            data_identity_snapshot=self.data_identity.snapshot() if self.data_identity else None
        )
        loader = self.data_loader
        return self.output_class(
            inputs=(),
            context=context,
            transform=lambda: loader(),
            data_schema=self.data_schema,
            origin=self
        )
      

    def assign_to_data_identity(self, data_identity):
        if data_identity is None:
            if self.data_identity is None:
                return
            self.data_identity.remove_data_component(self)
            return
        data_identity.add_data_components([self])


class DataIdentity:
    name = "identity"
    _snapshot_fields = ()

    def __init__(self, data_components=None):
        self.uid = uuid.uuid4()
        self.data_components = set()
        self.subject = None
        self._annotations = []
        self._version = 0
        if data_components is not None:
            self.add_data_components(data_components)

    def __deepcopy__(self, memo):
        return self
        
    def __hash__(self):
        return hash(self.uid)

    def __eq__(self, other):
        return isinstance(other, DataIdentity) and other.uid == self.uid

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

    def set_annotation(self, key, value, annotator=None, source_signal=None):
        annotation = Annotation(
            key=key,
            value=value,
            annotator=annotator,
            source_signal=source_signal,
            timestamp=time.time(),
        )
        self._annotations.append(annotation)
        self._version += 1

    def snapshot(self):
        return frozendict({
            field: getattr(self, field)
            for field in self._snapshot_fields
        })
    
    @property
    def version(self):
        return self._version


@dataclass
class Annotation:
    key: str
    value: Any
    annotator: Optional[object] = None
    source_signal: Optional[Signal] = None
    timestamp: float = field(default_factory=time.time)


class FeatureMixin:
    def extract_features(self, *features, registry=feature_registry, group_by=None):
        from k_onda.transformers import ExtractFeatures
        return ExtractFeatures(*features, registry=registry, group_by=group_by)((self,))


class Collection(StackMixin, CalculateMixin, SelectMixin, AggregateMixin, PointProcessMixin,
                 FeatureMixin):
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
            return [component.signal for component in base_components]

        return base_components

    def group_by(self, group_on, strict=True):
        return CollectionMap(self.members, group_on, strict=strict)
    









class MapMixin(DictDelegator, FeatureMixin):
    pass


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
    




