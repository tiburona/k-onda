from mne.time_frequency import tfr_array_multitaper
import numpy as np

from .signal import TimeFrequencySignal
from .calculator import Calculator
from .dataarray_factories import make_time_frequency_series


class Spectrogram(Calculator):

    def __init__(self, config):
        self.config = config

    def get_child_signal_class(self, _):
        return TimeFrequencySignal

    def _apply(self, data):
        data_np = np.asarray(data)
        data_3d = data_np[np.newaxis, np.newaxis, :]
        power = tfr_array_multitaper(data_3d, **self.config).squeeze()
        freqs = self.config["freqs"] 
        dt = self.config["decim"] / self.config["sfreq"]  # seconds per bin
        start = data['time'][0].item()
        da = make_time_frequency_series(power, freqs, start=start, dt=dt)
        return da
