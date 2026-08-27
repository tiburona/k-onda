import numpy as np
import xarray as xr

from .core import Calculator
from k_onda.utils import np_from_xr


class Arithmetic(Calculator):
    name = "arithmetic"
    arity = "one_or_more"

    def __init__(self, operand=None, alignment="exact"):
        if operand is not None:
            from .transformer_mixins import _operand_kind

            if _operand_kind(operand) != "concrete_operand":
                raise ValueError(
                    f"{self.format_call()}: operand must be a concrete numerical "
                    "value, not a signal operand."
                )
        alignment_values = {"outer", "inner", "left", "right", "exact", "override"}
        if alignment not in alignment_values:
            raise ValueError(f"{self.format_call()}: alignment must be one of {alignment_values}.")
        self.operand = operand
        self.alignment = alignment

    def _validate_input(self, *inputs, **kwargs):
        super()._validate_input(*inputs, **kwargs)
        if self.operand is None and len(inputs) < 2:
            raise ValueError(
                f"{self.format_call()}: Arithmetic classes must be configured with an operand or "
                "provided with a second signal at call time."
                ) 
        if self.operand is not None and len(inputs) != 1:
            raise ValueError(
                f"{self.format_call()}: If arithmetic classes receive a concrete operand, they can "
                "receive a maximum of 1 signal operand."
            )
        
    def output_schema(self, *input_schemas):
        if len(input_schemas) == 1:
            return input_schemas[0]
        last_schema = input_schemas[0]
        for schema in input_schemas[1:]:
            if not last_schema.is_superset_of(schema):
                raise ValueError(
                    f"{self.format_call()}: Operands have incompatible data schemas."
                )
        return input_schemas[0]

    def _apply_inner(self, *input_data, **kwargs):
        data_to_align = [
            data for data in input_data if isinstance(data, (xr.DataArray, xr.Dataset))
            ]
        data_to_align = xr.align(*data_to_align, join=self.alignment)
        return data_to_align
  

class Add(Arithmetic):
    name = "add"
     
    def _apply_inner(self, *input_data, **kwargs):
        input_data = super()._apply_inner(*input_data, **kwargs)

        if self.operand is not None:
            return input_data[0] + self.operand
        else:
            _sum = input_data[0]
            for data in input_data[1:]:
                _sum = _sum + data
            return _sum
            


class Subtract(Arithmetic):
    name = "subtract"

    def _apply_inner(self, *input_data, **kwargs):
        input_data = super()._apply_inner(*input_data, **kwargs)

        if self.operand is not None:
            return input_data[0] - self.operand
        else:
            difference = input_data[0]
            for data in input_data[1:]:
                difference = difference - data
            return difference


class Multiply(Arithmetic):
    name = "multiply"

    def _apply_inner(self, *input_data, **kwargs):
        input_data = super()._apply_inner(*input_data, **kwargs)

        if self.operand is not None:
            return input_data[0] * self.operand
        else:
            product = input_data[0]
            for data in input_data[1:]:
                product = data * product
            return product

class Divide(Arithmetic):
    name = "divide"

    def _apply_inner(self, *input_data, **kwargs):
        input_data = super()._apply_inner(*input_data, **kwargs)

        if self.operand is not None:
            return input_data[0] / self.operand
        else:
            quotient = input_data[0]
            for data in input_data[1:]:
                quotient = quotient/data
            return quotient


class Normalize(Calculator):
    name = "normalize"
    methods = {"minmax", "rms", "zscore"}

    def __init__(self, method="rms", *, dim=None):
        normalized_dim = self._normalize_dim(dim)
        self._validate_configuration(method, normalized_dim)

        self.dim = normalized_dim
        self.method = method

    def _normalize_dim(self, dim):
        if dim is None or isinstance(dim, str):
            return dim

        try:
            return list(dim)
        except TypeError:
            raise TypeError(
                f"{self.format_call()}: dim must be None, a string, or an "
                "iterable of strings."
            ) from None

    def _validate_configuration(self, method, dim):
        if not isinstance(method, str):
            raise TypeError(f"{self.format_call()}: method must be a string.")
        if method not in self.methods:
            known_methods = ", ".join(sorted(self.methods))
            raise ValueError(
                f"{self.format_call()}: unknown normalization method {method!r}. "
                f"Available methods: {known_methods}."
            )

        if dim is None:
            return
        dims = [dim] if isinstance(dim, str) else dim
        if not dims:
            raise ValueError(
                f"{self.format_call()}: dim cannot be an empty collection."
            )
        if any(not isinstance(item, str) or not item for item in dims):
            raise TypeError(
                f"{self.format_call()}: every dimension must be a non-empty "
                "string."
            )
        if len(set(dims)) != len(dims):
            raise ValueError(
                f"{self.format_call()}: dimensions cannot be repeated."
            )

    def _validate_data_schema(self, input_schema):
        super()._validate_data_schema(input_schema)
        if not input_schema.dim_names:
            raise ValueError(
                f"{self.format_call()}: normalization requires at least one "
                "dimension."
            )
        if self.dim is None:
            return

        dims = [self.dim] if isinstance(self.dim, str) else self.dim
        missing = [dim for dim in dims if not input_schema.has_dim(dim)]
        if missing:
            raise ValueError(
                f"{self.format_call()}: cannot normalize over missing dimensions: "
                f"{missing}."
            )

    @staticmethod
    def _format_coord_value(coord):
        try:
            value = coord.pint.quantity.item()
        except Exception:
            value = coord.item()
        return repr(value)

    def _zero_denominator_locations(self, denominator, limit=5):
        arrays = (
            denominator.data_vars.items()
            if isinstance(denominator, xr.Dataset)
            else ((None, denominator),)
        )
        locations = []
        total = 0

        for variable_name, array in arrays:
            zero_mask = array == 0
            indices = np.argwhere(np.asarray(zero_mask))
            total += len(indices)

            for index in indices:
                if len(locations) >= limit:
                    continue

                parts = []
                if variable_name is not None:
                    parts.append(f"variable={variable_name!r}")

                selection = {
                    dim: int(position)
                    for dim, position in zip(zero_mask.dims, index)
                }
                selected = zero_mask.isel(selection) if selection else zero_mask
                reported_coords = set()
                for coord_name, coord in selected.coords.items():
                    if coord.ndim == 0:
                        parts.append(
                            f"{coord_name}={self._format_coord_value(coord)}"
                        )
                        reported_coords.add(coord_name)

                for dim, position in selection.items():
                    if dim not in reported_coords:
                        parts.append(f"{dim}_index={position}")

                locations.append(", ".join(parts) or "entire selected input")

        return locations, total

    def _validate_denominator(self, denominator, name):
        locations, total = self._zero_denominator_locations(denominator)
        if total:
            location_text = "; ".join(locations)
            if total > len(locations):
                location_text += f"; and {total - len(locations)} more"
            raise ZeroDivisionError(
                f"{self.format_call()}: {name} is zero at {location_text}, so "
                "normalization is undefined."
            )

    def _apply_inner(self, data, *args, **kwargs):
        dim = self.dim
        if self.method == "rms":
            rms = np.sqrt((data**2).mean(dim=dim))
            self._validate_denominator(rms, "RMS denominator")
            norm_params = {"rms": rms}
            result = data / rms
        elif self.method == "zscore":
            mean = data.mean(dim=dim)
            std = data.std(dim=dim)
            self._validate_denominator(std, "standard deviation")
            norm_params = {"mean": mean, "std": std}
            result = (data - mean) / std
        elif self.method == "minmax":
            data_min = data.min(dim=dim)
            data_max = data.max(dim=dim)
            data_range = data_max - data_min
            self._validate_denominator(data_range, "data range")
            norm_params = {"data_min": data_min, "data_max": data_max}
            result = (data - data_min) / data_range
        return result, {"norm_params": norm_params}

    def _wrap_result(self, result, data, norm_params=None):

        if norm_params is not None:
            stripped = {}
            units = {}

            for key, param in norm_params.items():
                stripped[key], units[key] = np_from_xr(param)

            prenorm_units = (
                data.attrs.get(
                    "feature_units"
                )  # e.g. {'fwhm': unit, 'firing_rate': unit}
                or {"data": next(iter(units.values()), None)}  # e.g. {'data': unit}
            )

            result = result.assign_attrs(
                norm_dim=self.dim, norm_params=stripped, prenorm_units=prenorm_units
            )

        result = super()._wrap_result(result)
        return result
