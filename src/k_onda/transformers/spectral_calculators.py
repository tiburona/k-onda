from copy import deepcopy
from mne.time_frequency import tfr_array_multitaper
import numpy as np
import xarray as xr
import pint

from k_onda.central import DimBounds, DimPair, AxisInfo, AxisKind, types
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
        return types.TimeFrequencySignal
    
    def output_schema(self, input_schema):
        new_axis = AxisInfo(name='frequency', metadim='frequency', kind=AxisKind.AXIS)
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
        return {'fs': scalar(parent.sampling_rate)}

    def _apply_inner(self, data, fs):
        config = deepcopy(self.config)
        config["sfreq"] = fs
        data_np = np.asarray(data)
        data_3d = data_np[np.newaxis, np.newaxis, :]
        power = tfr_array_multitaper(data_3d, **config).squeeze()
        return (power, {'fs': fs})
    
    def _wrap_result(self, result, data, fs):

        start = data['time'][0].item().magnitude
        dt = (self.config['decim'] / fs)  
        freqs = self.config['freqs']
        time = (np.arange(result.shape[1]) * dt) + start

        def get_time_coords(coord):
            return (
                ((np.arange(result.shape[1]) * dt) + scalar(data.coords[coord][0].pint.to('s'))
                 ))

        da = xr.DataArray(
            result,
            dims=('frequency', 'time'),
            coords={
                'frequency': freqs,
                'time': time
            }
        )

        da.attrs = data.attrs

        # TODO: do I want to get spectrograms units?

        old_time_coords = {
            coord for coord in data.coords
            if 'time' in data.coords[coord].dims
        }

        time_coords = {
            coord: ('time', get_time_coords(coord))
            for coord in old_time_coords
            }
        
        da = da.assign_coords(**time_coords)
        da = da.pint.quantify({'frequency': 'Hz', 'time': 's'})

        # TODO include units here


        result = super()._wrap_result(da)
        return result
