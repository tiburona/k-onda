from copy import deepcopy
from mne.time_frequency import tfr_array_multitaper
import numpy as np
import xarray as xr
import pint

from k_onda.central import DimBounds, DimPair, AxisInfo, AxisKind, type_registry
from .core import PaddingCalculator
from ..utils import scalar


# TODO: Right now this calculator has a baked in assumption that we are computing
# a spectrogram on time.  That's the most common case, but there could be others
# and it would be nice to generalize it for consistency with the generalization
# of other calculators.


class Spectrogram(PaddingCalculator):
    name = "spectrogram"

    def __init__(self, config):
        self.config = config

    @property
    def fixed_output_class(self):
        return type_registry.TimeFrequencySignal

    def output_schema(self, input_schema):
        new_axis = AxisInfo(name="frequency", metadim="frequency", kind=AxisKind.AXIS)
        return input_schema.with_added(new_axis)

    def _compute_padlen(self, _, apply_kwargs):
        n_cycles = self.config["n_cycles"]
        freqs = self.config["freqs"]
        f_min = freqs[0]
        if isinstance(n_cycles, np.ndarray):
            # TODO: it would be safer here to test for iterables explicitly.
            pad_needed = np.max(n_cycles / freqs) / 2
        else:
            pad_needed = n_cycles / (2 * f_min)
        pad_seconds = pad_needed * pint.application_registry.seconds

        return DimBounds({"time": DimPair([pad_seconds, pad_seconds])})

    def _get_extra_apply_kwargs(self, parent):
        return {"fs": scalar(parent.sampling_rate)}

    def _apply_inner(self, data, fs, data_schema=None, **kwargs):
        config = deepcopy(self.config)
        config["sfreq"] = fs
        
        time_dim = data_schema.concrete_dim_from("time")
        leading_dims = [d for d in data.dims if d != time_dim]
        data.transpose(*leading_dims, time_dim)
        data_np = np.asarray(data)
        if data_np.ndim == 1:
            data_3d = data_np[np.newaxis, np.newaxis, :]
        elif data_np.ndim == 2:
            data_3d = data_np[:, np.newaxis, :]
        elif data_np.ndim == 3:
            data_3d = data_np
        else:
            raise ValueError("tfr_array_multitaper can not run on data with more than 3 dims.")
       
        power = tfr_array_multitaper(data_3d, **config).squeeze()
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

        # freequency coord
        result_dim_coords["frequency"] = self.config["freqs"]
        
        # concrete time dim coord
        dt = self.config["decim"] / fs
        start = data.coords[concrete_time_dim].isel({concrete_time_dim: 0}).values
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
                start = data.coords[coord].isel({concrete_time_dim:0}).values
            else:
                leading_shape = data.shape[:-1]
                time_dim_coord = np.broadcast_to(
                    time_coord_base, 
                    (*leading_shape, time_coord_base.size)
                    ).copy()
                start = data.coords[coord].isel({concrete_time_dim:0}).values[..., np.newaxis]
                
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

        da.attrs = data.attrs
        
        da = da.pint.quantify({"frequency": "Hz", concrete_time_dim: "s"})

        result = super()._wrap_result(da)
        return result
