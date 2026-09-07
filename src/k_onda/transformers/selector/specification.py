from ..core import Transformer, Transform
from k_onda.central import type_registry, DimBounds

@type_registry.register
class SpecifySelection(Transformer):
    name = "selector"

    def __init__(
            self, 
            mode="local", 
            locus=None, 
            new_dim=None, 
            window=None, 
            ragged=None,
            ):
        mode = "local" if mode is None else mode
        self._validate_configuration(mode, locus, new_dim, window)
        self.mode = mode
        self.locus = locus
        self.new_dim = new_dim
        self.window = window
        self.ragged = ragged or {}

    def _validate_configuration(self, mode, locus, new_dim, window):
        if mode not in ("local", "pushdown"):
            raise ValueError(
                f"{self.format_call()}: mode must be 'local' or 'pushdown'."
            )
        if not isinstance(locus, (type_registry.Locus, type_registry.LocusSet)):
            raise TypeError(
                f"{self.format_call()}: locus must be a Locus or LocusSet."
            )
        if not hasattr(locus, "dim_bounds"):
            raise NotImplementedError(
                f"{self.format_call()}: selection of a point locus without a "
                "window is not implemented."
            )
        if new_dim is not None and (
            not isinstance(new_dim, str) or not new_dim.strip()
        ):
            raise TypeError(
                f"{self.format_call()}: new_dim must be a non-empty string or "
                "None."
            )
        if new_dim is not None and not isinstance(locus, type_registry.LocusSet):
            raise ValueError(
                f"{self.format_call()}: new_dim requires a LocusSet whose members "
                "supply the values along the new dimension."
            )
        if window is not None and not isinstance(window, DimBounds):
            raise TypeError(
                f"{self.format_call()}: window must be DimBounds or None."
            )

    def _call_on_signal(self, signal, key_spec):
        output = super()._call_on_signal(signal, key_spec)
        if hasattr(self.locus, "conditions"):
            output.conditions.update(self.locus.conditions)
        return output

    def _get_transform(self, *inputs, key_spec=None, **kwargs):
        return Transform(fn=lambda x: x, padlen=self.window, key_spec=key_spec)

    @property
    def fixed_output_class(self):
        return type_registry.SelectorSignal

    def _validate_input(self, signal, key_spec=None):

        super()._validate_input(signal, key_spec=key_spec)

        if key_spec and key_spec.input_name is not None:
            raise NotImplementedError(
                f"{self.format_call()}: Dataset key selection is not implemented; "
                "select a payload before calling select()."
            )

    def _validate_data_schema(self, input_schema):
        super()._validate_data_schema(input_schema)

        if not input_schema.is_selectable(self.locus.dim):
            raise ValueError(
                f"{self.format_call()}: input schema cannot be selected on "
                f"{self.locus.dim!r}."
            )
        if input_schema.is_point_process() and isinstance(
            self.locus, type_registry.LocusSet
        ):
            raise NotImplementedError(
                f"{self.format_call()}: this operation would produce a ragged "
                "array, and "
                "support for that is not yet implemented."
            )


