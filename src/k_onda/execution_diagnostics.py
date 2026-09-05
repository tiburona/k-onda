"""Compact context for exceptions raised while executing a lazy signal graph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import reprlib
from typing import Any

import numpy as np
import xarray as xr


_VALUE_REPR = reprlib.Repr()
_VALUE_REPR.maxdict = 6
_VALUE_REPR.maxlist = 6
_VALUE_REPR.maxset = 6
_VALUE_REPR.maxtuple = 6
_VALUE_REPR.maxstring = 100
_VALUE_REPR.maxother = 100

_NOTE_PREFIX = "K-Onda execution context"
_MAX_DATASET_VARIABLES = 6
_MAX_SELECTION_COORDS = 5
_MAX_COORD_LEVELS = 5


def _safe_getattr(obj: object, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _compact_repr(value: object) -> str:
    try:
        return _VALUE_REPR.repr(value)
    except Exception:
        return f"<{type(value).__name__}>"


def _diagnostic_identifier(entity: object | None) -> str | None:
    if entity is None:
        return None

    for attribute in ("display_id", "label", "id", "uid"):
        value = _safe_getattr(entity, attribute)
        if value is not None:
            try:
                return str(value)
            except Exception:
                return type(entity).__name__
    return type(entity).__name__


def _selection_axes(schema: object) -> tuple[dict[str, object], ...]:
    schemas: Sequence[object]
    if isinstance(schema, Mapping):
        schemas = tuple(schema.values())
    else:
        schemas = (schema,)

    selections: dict[str, dict[str, object]] = {}
    for item_schema in schemas:
        for axis in _safe_getattr(item_schema, "axes", ()) or ():
            source_dim = _safe_getattr(axis, "created_from_dim")
            source_metadim = _safe_getattr(axis, "created_from_metadim")
            if source_dim is None and source_metadim is None:
                continue

            name = _safe_getattr(axis, "name")
            if name is None:
                continue
            selection = selections.setdefault(
                name,
                {
                    "dim": name,
                    "source_dim": source_dim,
                    "source_metadim": source_metadim,
                    "coordinates": [],
                    "conditions": [],
                },
            )
            for coord in _safe_getattr(axis, "coords", ()) or ():
                coord_name = _safe_getattr(coord, "name")
                if coord_name is None or coord_name == name:
                    continue
                target = (
                    selection["conditions"]
                    if _safe_getattr(coord, "is_condition", False)
                    else selection["coordinates"]
                )
                if coord_name not in target:
                    target.append(coord_name)

    return tuple(selections.values())


def summarize_signal(signal: object) -> dict[str, object]:
    """Capture stable symbolic context without retaining a signal graph."""
    summary: dict[str, object] = {"signal": type(signal).__name__}
    try:
        entities = {
            "data identity": _safe_getattr(signal, "data_identity"),
            "origin": _safe_getattr(signal, "origin"),
            "subject": _safe_getattr(signal, "subject"),
            "session": _safe_getattr(signal, "session"),
        }
        for name, entity in entities.items():
            identifier = _diagnostic_identifier(entity)
            if identifier is not None:
                summary[name] = identifier

        conditions = _safe_getattr(signal, "conditions")
        if conditions:
            summary["conditions"] = _compact_repr(conditions)

        members = _safe_getattr(signal, "signals")
        if members is not None:
            try:
                summary["members"] = len(members)
            except Exception:
                pass

        schema = _safe_getattr(signal, "data_schema")
        if schema is not None:
            selections = _selection_axes(schema)
            if selections:
                summary["selections"] = selections
    except Exception:
        # A partial summary is better than allowing diagnostics to affect execution.
        pass
    return summary


def _format_call(transformer: object | None) -> str | None:
    if transformer is None:
        return None
    formatter = _safe_getattr(transformer, "format_call")
    if formatter is not None:
        try:
            return formatter()
        except Exception:
            pass
    return type(transformer).__name__


def build_transform_context(
    transformer: object,
    inputs: Sequence[object],
    key_spec: object | None,
) -> dict[str, object]:
    """Build context when a transform is created from symbolic inputs."""
    context: dict[str, object] = {
        "transformer": _format_call(transformer),
        "inputs": tuple(summarize_signal(item) for item in inputs),
    }
    if key_spec is not None:
        input_name = _safe_getattr(key_spec, "input_name")
        output_mode = _safe_getattr(key_spec, "output_mode")
        if input_name is not None or output_mode is not None:
            context["key routing"] = {
                "input": input_name,
                "output_mode": output_mode,
            }
    return context


def build_materialization_context(signal: object) -> dict[str, object]:
    """Build context for one signal's materialization boundary."""
    return {
        "transformer": _format_call(_safe_getattr(signal, "transformer")),
        "output": summarize_signal(signal),
    }


def _units(array: xr.DataArray) -> object | None:
    try:
        units = array.pint.units
    except Exception:
        units = None
    if units is None:
        units = array.attrs.get("units")
    return units


def _scalar_value(value: object) -> object:
    try:
        return value.item()
    except Exception:
        return value


def _magnitudes(array: xr.DataArray) -> np.ndarray:
    try:
        values = array.pint.magnitude
    except Exception:
        values = array.data
    return np.asarray(values)


def _format_coord_value(coord: xr.DataArray, index: int) -> str:
    try:
        value = _scalar_value(_magnitudes(coord).reshape(-1)[index])
    except Exception:
        return "?"
    units = _units(coord)
    rendered = _compact_repr(value)
    return f"{rendered} {units}" if units is not None else rendered


def _coord_range(coord: xr.DataArray) -> str | None:
    if coord.size == 0:
        return None
    first = _format_coord_value(coord, 0)
    if coord.size == 1:
        return first
    return f"{first}..{_format_coord_value(coord, -1)}"


def _coord_levels(coord: xr.DataArray) -> str | None:
    try:
        values = _magnitudes(coord).reshape(-1)
    except Exception:
        return None

    levels: list[object] = []
    rendered_levels: set[str] = set()
    truncated = False
    for value in values:
        value = _scalar_value(value)
        rendered = _compact_repr(value)
        if rendered in rendered_levels:
            continue
        if len(levels) == _MAX_COORD_LEVELS:
            truncated = True
            break
        rendered_levels.add(rendered)
        levels.append(value)
    suffix = ", ..." if truncated else ""
    return f"[{', '.join(_compact_repr(level) for level in levels)}{suffix}]"


def _selection_summary(
    data: xr.DataArray | xr.Dataset,
    selections: Sequence[Mapping[str, object]],
) -> list[str]:
    details: list[str] = []
    for selection in selections:
        dim = str(selection["dim"])
        if dim not in data.dims:
            continue

        source = selection.get("source_dim") or selection.get("source_metadim")
        parts = [f"{dim}<-{source}", f"size={data.sizes[dim]}"]
        if dim in data.coords and data.coords[dim].dims == (dim,):
            coord_range = _coord_range(data.coords[dim])
            if coord_range is not None:
                parts.append(f"index={coord_range}")

        coord_names = list(selection.get("coordinates", ()))[:_MAX_SELECTION_COORDS]
        for name in coord_names:
            if name in data.coords and data.coords[name].dims == (dim,):
                coord_range = _coord_range(data.coords[name])
                if coord_range is not None:
                    parts.append(f"{name}={coord_range}")

        condition_names = list(selection.get("conditions", ()))
        for name in condition_names[:_MAX_SELECTION_COORDS]:
            if name in data.coords and data.coords[name].dims == (dim,):
                levels = _coord_levels(data.coords[name])
                if levels is not None:
                    parts.append(f"{name}={levels}")

        details.append("selection " + ", ".join(parts))
    return details


def _summarize_array(
    array: xr.DataArray,
    selections: Sequence[Mapping[str, object]] = (),
) -> str:
    parts = []
    if array.name is not None:
        parts.append(f"name={array.name!r}")
    parts.extend((f"dimensions={dict(array.sizes)!r}", f"dtype={array.dtype}"))
    units = _units(array)
    if units is not None:
        parts.append(f"units={units}")
    parts.extend(_selection_summary(array, selections))
    return ", ".join(parts)


def summarize_data(
    data: object,
    signal_context: Mapping[str, object] | None = None,
) -> str:
    """Summarize xarray structure without including full data values."""
    selections = () if signal_context is None else signal_context.get("selections", ())
    if isinstance(data, xr.DataArray):
        return "DataArray(" + _summarize_array(data, selections) + ")"
    if isinstance(data, xr.Dataset):
        variables = []
        for index, (name, array) in enumerate(data.data_vars.items()):
            if index == _MAX_DATASET_VARIABLES:
                variables.append("...")
                break
            variable = f"{name}(dtype={array.dtype}"
            units = _units(array)
            if units is not None:
                variable += f", units={units}"
            variables.append(variable + ")")
        parts = [
            f"dimensions={dict(data.sizes)!r}",
            f"variables=[{', '.join(variables)}]",
        ]
        parts.extend(_selection_summary(data, selections))
        return "Dataset(" + ", ".join(parts) + ")"
    return type(data).__name__


def _format_signal(summary: Mapping[str, object]) -> str:
    excluded = {"signal", "selections"}
    parts = [str(summary.get("signal", "Signal"))]
    parts.extend(
        f"{name}={value}"
        for name, value in summary.items()
        if name not in excluded
    )
    selections = summary.get("selections", ())
    for selection in selections:
        source = selection.get("source_dim") or selection.get("source_metadim")
        parts.append(f"selection={selection['dim']}<-{source}")
    return ", ".join(parts)


def format_execution_note(
    *,
    stage: str,
    phase: str,
    context: Mapping[str, object] | None,
    data: Sequence[object] = (),
    output_data: object | None = None,
) -> str:
    """Format one human-readable exception note."""
    lines = [f"{_NOTE_PREFIX} (stage={stage}, phase={phase})"]
    context = context or {}

    transformer = context.get("transformer")
    if transformer:
        lines.append(f"  transformer: {transformer}")
    key_routing = context.get("key routing")
    if key_routing:
        lines.append(f"  key routing: {_compact_repr(key_routing)}")

    inputs = context.get("inputs", ())
    for index, input_context in enumerate(inputs):
        line = f"  input {index}: {_format_signal(input_context)}"
        if index < len(data):
            line += f"; data={summarize_data(data[index], input_context)}"
        lines.append(line)

    output = context.get("output")
    if output:
        lines.append(f"  output: {_format_signal(output)}")
    if output_data is not None:
        lines.append(f"  produced data: {summarize_data(output_data, output)}")

    return "\n".join(lines)


def add_execution_note(
    error: Exception,
    *,
    stage: str,
    phase: str,
    context: Mapping[str, object] | None,
    data: Sequence[object] = (),
    output_data: object | None = None,
) -> None:
    """Add at most one note per execution stage, never masking ``error``."""
    try:
        if has_execution_note(error, stage):
            return
        error.add_note(
            format_execution_note(
                stage=stage,
                phase=phase,
                context=context,
                data=data,
                output_data=output_data,
            )
        )
    except Exception:
        # Diagnostics must never replace the exception they are describing.
        return


def has_execution_note(error: Exception, stage: str) -> bool:
    """Return whether ``error`` already carries context for an execution stage."""
    try:
        marker = f"{_NOTE_PREFIX} (stage={stage},"
        return any(
            isinstance(note, str) and note.startswith(marker)
            for note in getattr(error, "__notes__", ())
        )
    except Exception:
        return False
