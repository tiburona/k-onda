from mne.time_frequency import tfr_array_multitaper
import numpy as np

from .signal import TimeFrequencySignal
from .calculator import Calculator
from .dataarray_factories import make_time_frequency_series


class Spectrogram(Calculator):

    def __init__(self, config):
        self.config = {"config": config}

    def get_child_signal_class(self, _):
        return TimeFrequencySignal

    def _apply(self, data, config):
        data_np = np.asarray(data)
        data_3d = data_np[np.newaxis, np.newaxis, :]
        power = tfr_array_multitaper(data_3d, **config).squeeze()
        freqs = config["freqs"]
        dt = config["decim"] / config["sfreq"]  # seconds per bin
        times = np.arange(power.shape[-1]) * dt  # plain floats in seconds  
        da = make_time_frequency_series(power, data.attrs['sampling_rate'], freqs)
        return da
