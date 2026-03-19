from dataclasses import dataclass, field
from typing import Optional, Any
from frozendict import frozendict
import time

from k_onda.signals import Signal


@dataclass
class ProvenanceContext:

    component: object
   
    # derived in __post_init__ 
    component_id: str = field(init=False)
    session_id: str = field(init=False)
    data_identity_id: Optional[str] = field(init=False, default=None)
    data_identity_snapshot: Optional[frozendict] =field(init=False, default=None)
    subject_id: str = field(init=False)
    subject_snapshot: Optional[frozendict] = field(init=False, default=None)
    experiment_id: str = field(init=False)
    experiment_snapshot: Optional[frozendict] = field(init=False, default=None)

    def __post_init__(self):
        self.component_id = self.component.uid
        session = self.component.data_source.session
        if hasattr(self.component, 'data_identity'):
            data_identity = self.component.data_identity
            self.data_identity_id = data_identity.uid
            self.data_identity_snapshot = data_identity.snapshot()
        self.session_id = session.uid
        subject = session.subject
        self.subject_id = subject.uid
        self.subject_snapshot = subject.snapshot()
        experiment = subject.experiment
        self.experiment_id = experiment.uid
        self.experiment_snapshot = experiment.snapshot()
        self.component = None
    


@dataclass
class Annotation:
    key: str
    value: Any
    annotator: Optional[object] = None
    source_signal: Optional[Signal] = None
    timestamp: float = field(default_factory=time.time)


class AnnotatorMixin:

    def _init_annotations(self):
        self._annotations = []
        self._version = 0

    @property
    def version(self):
        return self._version

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