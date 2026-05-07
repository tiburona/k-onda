import pytest

from k_onda.central import AxisInfo, AxisKind, Schema
from k_onda.transformers import ReduceDim, Scale
from tests.fakes import FakeDataComponent, make_lineage


@pytest.fixture
def parent_signal():
    _, _, _, data_source, data_identity, _ = make_lineage()
    component = FakeDataComponent(data_source, data_identity=data_identity)
    component.data_schema = Schema(
        axes=[AxisInfo("time", AxisKind.AXIS, metadim="time")]
    )
    return component.to_signal()
experiment, subject, session, data_source, data_identity, _ = make_lineage()


class TestCalculatorSignalConstruction:
    def test_calculator_constructs_child_signal(self, parent_signal):
        calculator = Scale(2)
        
        child = calculator(parent_signal)

        assert child.inputs == (parent_signal,)
        assert child.transformer is calculator
        assert parent_signal.inputs == ()

    def test_schema_changing_calculator_does_not_mutate_parent_schema(self, parent_signal):
        parent_schema = parent_signal.data_schema
        calculator = ReduceDim("time")

        child = calculator(parent_signal)

        assert parent_signal.data_schema is parent_schema
        assert parent_signal.data_schema.has_name("time")
        assert not child.data_schema.has_name("time")

