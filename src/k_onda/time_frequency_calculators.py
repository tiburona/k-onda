from mne.time_frequency import tfr_array_multitaper
import numpy as np

from .signal import TimeFrequencySignal
from .calculator import Calculator
from .dataarray_factories import make_frequency_time_series


class Spectrogram(Calculator):

    def __init__(self, config):
        self.config = {"config": config}

    def get_child_signal_class(self, _):
        return TimeFrequencySignal

    def _apply(self, data, config):

        data_3d = data[np.newaxis, np.newaxis, :]
        power = tfr_array_multitaper(data_3d, **config).squeeze()
        freqs = config["freqs"]
        dt = config["decim"] / config["sfreq"]  # seconds per bin
        times = np.arange(power.shape[-1]) * dt  # plain floats in seconds  
        da = make_frequency_time_series(power, freqs, times)
        return da