from mne.time_frequency import tfr_array_multitaper
import numpy as np
import xarray as xr
import pint
import inspect
from dataclasses import dataclass

from k_onda.central import DimBounds, DimPair, AxisInfo, AxisKind, type_registry as tr
from .core import PaddingCalculator
from ..utils import scalar


class SpectrogramCalculatorRegistry:

    def __init__(self):
        self._spectrogram_types = {}

    def register(self, method):
        def decorator(spectrogram_type):
            self._spectrogram_types[method] = spectrogram_type
            return spectrogram_type

        return decorator

    def create_calculator(self, method, **kwargs):
        try:
            spectrogram_type = self._spectrogram_types[method]
        except KeyError:
            known_types = ", ".join(sorted(self._spectrogram_types))
            raise ValueError(
                f"Unknown spectrogram method {method!r}. Registered methods: {known_types}."
                ) from None

        try:
            inspect.signature(spectrogram_type).bind(**kwargs)
        except TypeError as error:
            raise TypeError(
                f"Invalid parameters for spectrogram method {method!r}: {error}"
            ) from None
        
        return spectrogram_type(**kwargs)


spectrogram_registry = SpectrogramCalculatorRegistry()


@spectrogram_registry.register("multitaper")
@dataclass(frozen=True)
class MultitaperSpectrogram:
    n_cycles: int | float | list | tuple | np.ndarray
    freqs: tuple | list | np.ndarray
    decim: int
    time_bandwidth: int
    output: str = "power"

    def compute_spectrogram(self, data_3d, fs):
        power = tfr_array_multitaper(
            data_3d, 
            fs,
            self.freqs,
            n_cycles=self.n_cycles,
            time_bandwidth=self.time_bandwidth,
            decim=self.decim,
            output=self.output
            ).squeeze()
        return power

    def compute_padlen(self):
        n_cycles = (
            np.asarray(self.n_cycles) 
            if isinstance(self.n_cycles, (list, tuple)) 
            else self.n_cycles
            )
        freqs = self.freqs
        f_min = self.freqs[0]
        if isinstance(n_cycles, np.ndarray):
            pad_needed = np.max(n_cycles / freqs) / 2
        else:
            pad_needed = n_cycles / (2 * f_min)
        pad_seconds = pad_needed * pint.application_registry.seconds
        return pad_seconds


class Spectrogram(PaddingCalculator):
    name = "spectrogram"
    accepted_data_types = (xr.DataArray,)

    def __init__(self, method, **kwargs):
        self.method = method
        try:
            self.spectrogram_calculator = spectrogram_registry.create_calculator(method, **kwargs)
        except (TypeError, ValueError) as error:
            raise type(error)(f"{self.format_call()}: {error}") from None

    @property
    def fixed_output_class(self):
        return tr.TimeFrequencySignal

    def _validate_data_schema(self, input_schema):
        super()._validate_data_schema(input_schema)
        if input_schema.concrete_dim_from("time") is None:
            raise NotImplementedError(
                f"{self.format_call()}: spectrograms over dimensions other than "
                "time are not implemented yet."
            )
        if len(input_schema.dim_names) > 3:
            raise NotImplementedError(
                f"{self.format_call()}: spectrograms of inputs with more than "
                "three dimensions are not implemented yet."
            )

    def output_schema(self, input_schema):
        new_axis = AxisInfo(name="frequency", metadim="frequency", kind=AxisKind.AXIS)
        return input_schema.with_added(new_axis)

    def _compute_padlen(self, _, apply_kwargs):
        pad_seconds = self.spectrogram_calculator.compute_padlen()
        return DimBounds({"time": DimPair([-pad_seconds, pad_seconds])})

    def _get_extra_apply_kwargs(self, parent):
        return {"fs": scalar(parent.sampling_rate)}

    def _apply_inner(self, data, fs, data_schema=None, **kwargs):
        
        time_dim = data_schema.concrete_dim_from("time")
        leading_dims = [d for d in data.dims if d != time_dim]
        data = data.transpose(*leading_dims, time_dim)
        data_np = np.asarray(data.pint.magnitude)
        if data_np.ndim == 1:
            data_3d = data_np[np.newaxis, np.newaxis, :]
        elif data_np.ndim == 2:
            data_3d = data_np[:, np.newaxis, :]
        elif data_np.ndim == 3:
            data_3d = data_np

        power = self.spectrogram_calculator.compute_spectrogram(data_3d, fs)
        return (power, {"fs": fs, "data_schema": data_schema})

    def _wrap_result(self, result, data, fs, data_schema): 
        concrete_time_dim = data_schema.concrete_dim_from("time")
        spectrogram_dims = ("frequency", concrete_time_dim)
        other_dims = data.dims[:-1]
        result_dims = other_dims + spectrogram_dims

        # preceding dim coords, if any (epoch or channel)
        result_dim_coords = {
            k: data.coords[k] for k in other_dims
        }

        # frequency coord
        result_dim_coords["frequency"] = self.spectrogram_calculator.freqs
        
        # concrete time dim coord
        dt = self.spectrogram_calculator.decim / fs
        start = data.coords[concrete_time_dim].isel({concrete_time_dim: 0}).pint.magnitude
        concrete_time_dim_coord = np.arange(result.shape[-1]) * dt + start
        result_dim_coords[concrete_time_dim] = concrete_time_dim_coord

        da = xr.DataArray(
            result,
            dims=result_dims,
            coords=result_dim_coords
        )

        # auxiliary time coords
        def get_time_coords(coord):
            time_coord_base = np.arange(result.shape[-1]) * dt
            is_relative = data_schema.coord_by_name(coord).is_relative
            
            if is_relative:
                time_dim_coord = time_coord_base 
                start = data.coords[coord].isel({concrete_time_dim:0}).pint.magnitude
            else:
                leading_shape = data.shape[:-1]
                time_dim_coord = np.broadcast_to(
                    time_coord_base, 
                    (*leading_shape, time_coord_base.size)
                    ).copy()
                start = data.coords[coord].isel(
                    {concrete_time_dim:0}
                    ).pint.magnitude[..., np.newaxis]
                
            time_dim_coord += start
            time_dim_coord = pint.Quantity(time_dim_coord, 's')

            if is_relative:
                return ((concrete_time_dim,), time_dim_coord)
            else:
                return (data.dims[:-1] + (concrete_time_dim,), time_dim_coord)
    
        all_time_coords = {
            k: get_time_coords(k) 
            for k, v in data.coords.items() 
            if concrete_time_dim in v.dims
            }
        
        da = da.assign_coords(all_time_coords)

        preserved_aux_coords = {
            name: coord 
            for name, coord in data.coords.items()
            if name not in result_dim_coords 
            and concrete_time_dim not in coord.dims
            and set(coord.dims).issubset(set(result_dims))
        }

        da = da.assign_coords(preserved_aux_coords)

        da.attrs = data.attrs
        
        da = da.pint.quantify({"frequency": "Hz", concrete_time_dim: "s"})

        result = super()._wrap_result(da)
        return result
