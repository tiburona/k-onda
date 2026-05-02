import pytest

from k_onda.sources import DataComponent
from k_onda.central import type_registry

from tests.fakes import make_lineage, FakeDataComponent



experiment, subject, session, data_source, data_identity, _ = make_lineage()


def test_calling_to_signal_on_data_component_returns_signal():
   
    component = DataComponent(data_source, data_identity=data_identity)

    signal = component.to_signal()

    assert isinstance(signal, type_registry.Signal)
    
    with pytest.raises(ValueError, match=r"You must call \.compile\(\)"):
        signal.data

    assert signal.inputs == ()

    assert signal.origin is component

    assert signal.data_schema == component.data_schema

    assert isinstance(signal.context, type_registry.ProvenanceContext)
    assert signal.context.component is None
    assert signal.context.component_id == component.uid

