import traceback

import numpy as np
import pytest
import xarray as xr

from k_onda.central import AxisInfo, AxisKind, CoordInfo, DatasetSchema, Schema
from k_onda.signals import Signal
from k_onda.transformers import Add, Calculator, Transformer
from tests.fakes import FakeDataComponent, make_lineage


class RaiseInner(Calculator):
    def __init__(self, error):
        self.error = error

    def _apply_inner(self, data, **kwargs):
        raise self.error


class RaiseBinaryInner(RaiseInner):
    arity = "two"

    def _apply_inner(self, *data, **kwargs):
        raise self.error


class RaiseApply(Transformer):
    def __init__(self, error):
        self.error = error

    def _apply(self, data):
        raise self.error


class DropVariable(Transformer):
    def __init__(self, name):
        self.name = name

    def _apply(self, data):
        return data.drop_vars(self.name)

    def output_schema(self, input_schema):
        return input_schema


class RaiseWhileWrapping(Calculator):
    def __init__(self, error):
        self.error = error

    def _wrap_result(self, result, *args):
        raise self.error


class RaiseWhileMerging(Calculator):
    def __init__(self, error):
        self.error = error

    def merge_keys(self, data, result, key_spec):
        raise self.error


def _schema(dim="time"):
    return Schema(axes=[AxisInfo(dim, AxisKind.AXIS, metadim=dim)])


def _source_signal(data, schema=None, *, conditions=None):
    _, _, _, data_source, data_identity, _ = make_lineage()
    component = FakeDataComponent(data_source, data_identity=data_identity)
    component.data_schema = schema or _schema()
    component._data = data
    signal = component.to_signal()
    signal.conditions = conditions or {}
    return signal


def _notes(error):
    return getattr(error, "__notes__", ())


def _note_for(error, stage):
    matches = [
        note
        for note in _notes(error)
        if note.startswith(f"K-Onda execution context (stage={stage},")
    ]
    assert len(matches) == 1
    return matches[0]


def test_calculator_error_preserves_exception_and_adds_each_stage_once():
    data = xr.DataArray(
        [1.0, 2.0],
        dims="time",
        coords={"time": [0, 1]},
        attrs={"units": "mV"},
    )
    error = RuntimeError("calculation failed")
    failing = RaiseInner(error)(_source_signal(data))
    downstream = Add(1)(failing).compile()

    with pytest.raises(RuntimeError) as caught:
        downstream.data

    assert caught.value is error
    assert str(caught.value) == "calculation failed"
    assert [
        note.split("stage=", 1)[1].split(",", 1)[0]
        for note in _notes(error)
    ] == ["apply_inner", "apply", "materialize"]
    assert any(
        frame.name == "_apply_inner"
        for frame in traceback.extract_tb(caught.value.__traceback__)
    )

    inner_note = _note_for(error, "apply_inner")
    assert "transformer: RaiseInner(" in inner_note
    assert "data identity=identity-1" in inner_note
    assert "subject=subject-1" in inner_note
    assert "session=session-1" in inner_note
    assert "dimensions={'time': 2}" in inner_note
    assert "dtype=float64" in inner_note
    assert "units=mV" in inner_note


def test_multi_input_calculator_reports_every_input():
    first = xr.DataArray([1.0], dims="time", coords={"time": [0]})
    second = xr.DataArray([2], dims="time", coords={"time": [0]})
    error = RuntimeError("binary calculation failed")
    signal = RaiseBinaryInner(error)(
        _source_signal(first),
        _source_signal(second),
    ).compile()

    with pytest.raises(RuntimeError):
        signal.data

    note = _note_for(error, "apply_inner")
    assert "input 0:" in note
    assert "input 1:" in note
    assert "dtype=float64" in note
    assert "dtype=int64" in note


def test_validation_error_reports_apply_phase_without_apply_inner_note():
    data = xr.DataArray(
        [np.nan],
        dims="time",
        coords={"time": [0]},
    )
    signal = Add(1)(_source_signal(data)).compile()

    with pytest.raises(ValueError) as caught:
        signal.data

    assert not any("stage=apply_inner" in note for note in _notes(caught.value))
    assert "phase=validate materialized inputs" in _note_for(caught.value, "apply")
    _note_for(caught.value, "materialize")


def test_generic_transformer_error_gets_apply_and_materialize_notes():
    data = xr.DataArray([1.0], dims="time", coords={"time": [0]})
    error = LookupError("transform failed")
    signal = RaiseApply(error)(_source_signal(data)).compile()

    with pytest.raises(LookupError) as caught:
        signal.data

    assert caught.value is error
    assert "phase=run transformer application" in _note_for(error, "apply")
    _note_for(error, "materialize")
    assert not any("stage=apply_inner" in note for note in _notes(error))


def test_key_routing_failure_reports_dataset_structure_and_phase():
    item_schema = _schema()
    data = xr.Dataset(
        {
            "value": xr.DataArray(
                [1.0],
                dims="time",
                coords={"time": [0]},
                attrs={"units": "mV"},
            )
        }
    )
    source = _source_signal(data, DatasetSchema({"value": item_schema}))
    without_value = DropVariable("value")(source)
    signal = Add(1)(without_value, key="value").compile()

    with pytest.raises(KeyError) as caught:
        signal.data

    note = _note_for(caught.value, "apply")
    assert "phase=resolve dataset input key" in note
    assert "key routing: {'input': 'value', 'output_mode': 'replace'}" in note
    assert "Dataset(dimensions={'time': 1}, variables=[])" in note


@pytest.mark.parametrize(
    ("calculator_class", "phase"),
    [
        (RaiseWhileWrapping, "wrap calculator result"),
        (RaiseWhileMerging, "merge dataset output"),
    ],
)
def test_later_calculator_phases_are_identified(calculator_class, phase):
    error = RuntimeError(f"failed during {phase}")
    item_schema = _schema()
    data = xr.Dataset(
        {
            "value": xr.DataArray(
                [1.0],
                dims="time",
                coords={"time": [0]},
            )
        }
    )
    source = _source_signal(data, DatasetSchema({"value": item_schema}))
    signal = calculator_class(error)(source, key="value").compile()

    with pytest.raises(RuntimeError) as caught:
        signal.data

    assert caught.value is error
    assert f"phase={phase}" in _note_for(error, "apply")
    assert not any("stage=apply_inner" in note for note in _notes(error))


def test_source_loading_error_gets_source_materialization_context():
    _, _, _, data_source, data_identity, _ = make_lineage()
    component = FakeDataComponent(data_source, data_identity=data_identity)
    error = OSError("could not read source")

    def load():
        raise error

    signal = Signal(
        inputs=(),
        transform=load,
        data_schema=_schema(),
        origin=component,
    ).compile()

    with pytest.raises(OSError) as caught:
        signal.data

    assert caught.value is error
    note = _note_for(error, "materialize")
    assert "phase=load source data" in note
    assert "data identity=identity-1" in note
    assert not any("stage=apply" in item for item in _notes(error))


def test_source_schema_error_reports_produced_data():
    data = xr.DataArray([1.0], dims="wrong", coords={"wrong": [0]})
    signal = _source_signal(data, _schema()).compile()

    with pytest.raises(ValueError) as caught:
        signal.data

    note = _note_for(caught.value, "materialize")
    assert "phase=validate source data against schema" in note
    assert "produced data: DataArray(dimensions={'wrong': 1}, dtype=float64)" in note


def test_selected_epoch_context_is_summarized_without_full_values():
    selection_schema = Schema(
        axes=[
            AxisInfo(
                "trial",
                AxisKind.ORDINAL_INDEX,
                created_from_dim="epoch",
                created_from_metadim="time",
                coords=(
                    CoordInfo("trial_start_time", role="auxiliary"),
                    CoordInfo("trial_stop_time", role="auxiliary"),
                    CoordInfo(
                        "stimulus",
                        role="auxiliary",
                        is_condition=True,
                        levels=("tone", "noise"),
                        values_sequence=("tone", "noise"),
                    ),
                ),
            )
        ]
    )
    data = xr.DataArray(
        [1.0, 2.0],
        dims="trial",
        coords={
            "trial": [0, 1],
            "trial_start_time": ("trial", [10.0, 20.0]),
            "trial_stop_time": ("trial", [15.0, 25.0]),
            "stimulus": ("trial", ["tone", "noise"]),
        },
    )
    error = RuntimeError("selected calculation failed")
    signal = RaiseInner(error)(
        _source_signal(data, selection_schema, conditions={"block": "A"})
    ).compile()

    with pytest.raises(RuntimeError):
        signal.data

    note = _note_for(error, "apply_inner")
    assert "conditions={'block': 'A'}" in note
    assert "selection=trial<-epoch" in note
    assert "selection trial<-epoch, size=2, index=0..1" in note
    assert "trial_start_time=10.0..20.0" in note
    assert "trial_stop_time=15.0..25.0" in note
    assert "stimulus=['tone', 'noise']" in note
    assert "[1.0, 2.0]" not in note
