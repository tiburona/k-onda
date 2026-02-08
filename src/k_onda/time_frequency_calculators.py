from mne.time_frequency import tfr_array_multitaper
import numpy as np

from .signal import TimeFrequencySignal
from .calculator import PaddingCalculator
from .dataarray_factories import make_time_frequency_series


class Spectrogram(PaddingCalculator):

    # TODO: this calculator should add padding

    def __init__(self, config):
        self.config = config

    def _get_child_signal_class(self, _):
        return TimeFrequencySignal
    
    def _get_apply_kwargs(self, parent_signal):
        return {
            'config': self.config,
            'sampling_rate': parent_signal.sampling_rate
        }
    
    def _apply(self, data, config, sampling_rate):
        fs = sampling_rate.magnitude
        config['sfreq'] = fs
        data_np = np.asarray(data)
        data_3d = data_np[np.newaxis, np.newaxis, :]
        power = tfr_array_multitaper(data_3d, **config).squeeze()
        freqs = config["freqs"] 
        dt = config["decim"] / config["sfreq"]  # seconds per bin
        start = data['time'][0].item()
        da = make_time_frequency_series(power, freqs, start=start, dt=dt)
        return da
