from mne.time_frequency import tfr_array_multitaper
import numpy as np

from .signal import TimeFrequencySignal
from .calculator import PaddingCalculator, with_key_access
from .dataarray_factories import make_time_frequency_series
from .central import ureg


class Spectrogram(PaddingCalculator):

    obligate_output_class = TimeFrequencySignal

    def __init__(self, config):
        self.config = config

    def get_child_signal_class(self, _):
        return TimeFrequencySignal
    
    def _compute_padlen(self, _, apply_kwargs):
        n_cycles = self.config['n_cycles']
        freqs = self.config['freqs']
        f_min = freqs[0]
        if isinstance(n_cycles, np.ndarray):
            # todo: it would be safer here to test for my kinds of iterables and not assume it's numpy
            pad_needed = np.max(n_cycles / freqs) / 2
        else:
            pad_needed = n_cycles / (2 * f_min)
        pad_seconds = pad_needed * ureg.seconds

        return {"time": (pad_seconds, pad_seconds)}
    
    def _get_distinctive_apply_kwargs(self, parent_signal):
        return {'parent_signal': parent_signal}
    
    @with_key_access
    def _apply(self, data, parent_signal):
        config = self.config
        fs = parent_signal.sampling_rate.magnitude
        config['sfreq'] = fs
        data_np = np.asarray(data)
        data_3d = data_np[np.newaxis, np.newaxis, :]
        power = tfr_array_multitaper(data_3d, **config).squeeze()
        freqs = config["freqs"] 
        dt = config["decim"] / config["sfreq"]  # seconds per bin
        start = data['time'][0].item()
        da = make_time_frequency_series(power, freqs, start=start, dt=dt)
        return da
