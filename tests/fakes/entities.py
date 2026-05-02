from types import SimpleNamespace
from frozendict import frozendict

from k_onda.sources import DataComponent


class SnapshotEntity:
    def __init__(self, snapshot, **attrs):
        self._snapshot = frozendict(snapshot)
        for key, value in attrs.items():
            setattr(self, key, value)

    def snapshot(self):
        return self._snapshot
    

class FakeDataComponent(DataComponent):
    data_schema = object()

    def _data_loader(self):
        return "loaded"
    
    

def make_lineage():
    experiment = SnapshotEntity(
        id="exp-1",
        snapshot={"subject_ids": frozenset({"subject-1"})}
    )
    subject = SnapshotEntity(
        id="subject-1",
        snapshot={"session_ids": frozenset({"session-1"})}
    )
    session = SimpleNamespace(
        uid="session-1",
        experiment=experiment,
        subject=subject,
        start=0,
        duration=10,
        ureg=getattr(experiment, "ureg", None)
    )
    data_source = SimpleNamespace(
        uid="source-1",
        session=session,
        subject=subject
    )
    data_identity = SnapshotEntity(
        uid="identity-1",
        snapshot={"component_ids": ("component-1",)}
    )
    component = SimpleNamespace(
        uid="component-1",
        data_source=data_source,
        data_identity=data_identity
    )
    return experiment, subject, session, data_source, data_identity, component


