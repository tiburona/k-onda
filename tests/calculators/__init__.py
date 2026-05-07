from k_onda.transformers import Scale

from tests.fakes import FakeDataComponent, make_lineage


experiment, subject, session, data_source, data_identity, component = make_lineage()


def test_calculator_constructs_child_signal_without_mutating_parent():
    component = FakeDataComponent(data_source, data_identity=data_identity)