from frozendict import frozendict

from k_onda.provenance import ProvenanceContext

from tests.fakes import make_lineage

_, _, _, _, _, component = make_lineage()


def test_provenance_context_snapshots_component_lineage_and_drops_component_ref():
   
    context = ProvenanceContext(component)

    assert context.component_id == "component-1"
    assert context.data_identity_id == "identity-1"
    assert context.data_identity_snapshot == frozendict(
        {"component_ids": ("component-1",)}
    )
    assert context.session_id == "session-1"
    assert context.subject_id == "subject-1"
    assert context.subject_snapshot == frozendict(
        {"session_ids": frozenset({"session-1"})}
    )
    assert context.experiment_id == "exp-1"
    assert context.experiment_snapshot == frozendict(
        {"subject_ids": frozenset({"subject-1"})}
    )
    assert context.component is None
