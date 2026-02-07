from functools import lru_cache
import numpy as np
import pint
from scipy.signal import iirnotch, tf2sos, sosfreqz, filtfilt, sosfilt, sosfiltfilt

from .dataarray_factories import make_time_series
    

class Calculator:

    def __init__(self, config):
        self.config = config
        self.parent_signal = None

    # this method sometimes gets overridden
    def get_child_signal_class(self, parent_signal):
        return getattr(parent_signal, 'output_signal_class', type(parent_signal))

    def __call__(self, parent_signal):
        self.parent_signal = parent_signal
        child_signal_class = self.get_child_signal_class(parent_signal)
        return child_signal_class(
            parent=parent_signal,
            transform=self._apply,  # just the bound method
            calculator=self
        )
    
    @property
    def padlen(self):
        return self._compute_padlen()

    # this method sometimes gets overridden
    def _compute_padlen(self):
        return {}
    

class Normalize(Calculator):

    def __init__(self, method="rms"): 
        self.method = method
    
    def _apply(self, data):
        if self.method == "rms":
            result = data / np.sqrt(np.mean(data ** 2))
        elif self.method == "zscore":
            result = (data - np.mean(data)) / np.std(data)
        elif self.method == 'minmax':
            result = (data - data.min()) / (data.max() - data.min())
        else:
            raise ValueError("Unknown normalize method")
        
        da = make_time_series(
            result, 
            start=data.time[0].item(), 
            sampling_rate=self.parent_signal.sampling_rate)
        
        return da
        

class Filter(Calculator):
    def __init__(self, filter_config):
        self.config = filter_config
        self._designed_filter = None

    def __call__(self, parent_signal):
        self.parent_signal = parent_signal
        child_signal_class = self.get_child_signal_class(parent_signal)

        self.design_filter(self.config)
        
        return child_signal_class(
            parent=parent_signal,
            transform=self._apply,
            calculator=self
        )
    
    def design_filter(self, filter_config):
        fs = self.parent_signal.sampling_rate
        # todo: add in other kinds of filters
        self._designed_filter = self._design_sos(fs=fs.magnitude.item(), **filter_config)
        
    @staticmethod
    @lru_cache(maxsize=32)
    def _design_sos(method, f_lo, f_hi, fs, notch_Q=None, **_):
        if method == "iir_notch":
            f0 = 0.5 * (f_lo + f_hi)
            bw = max(1e-12, (f_hi - f_lo))
        
            if notch_Q is not None:
                Q = float(notch_Q)
            else:
                # Q = center frequency divided by bandwidth
                Q = float(f0 / bw)
            b, a = iirnotch(w0=f0, Q=Q, fs=fs)
            sos = tf2sos(b, a)
            return sos
        
    def _compute_padlen(self):
        # Generate impulse response
        N = int(self.parent_signal.sampling_rate.magnitude)  # 1 second worth of samples
        impulse = np.zeros(N)
        impulse[0] = 1.0
        h = sosfilt(self._designed_filter, impulse)

        # Find where it decays below some threshold 
        threshold = 1e-3  # -60 dB relative to peak
        peak = np.max(np.abs(h))
        settled = np.where(np.abs(h) > threshold * peak)[0]
        pad_needed = settled[-1] if len(settled) > 0 else 0
        pad_seconds = pad_needed/self.parent_signal.sampling_rate

        return {"time": (pad_seconds, pad_seconds)}
            
    def _apply(self, data):
        fs = self.parent_signal.sampling_rate
        axis = data.dims.index('time')
        result = sosfiltfilt(self._designed_filter, data, axis=axis)
        return make_time_series(result, fs, start=data['time'][0].item())


