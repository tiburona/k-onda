from functools import partial, lru_cache
import numpy as np
from scipy.signal import iirnotch, tf2sos, sosfreqz, filtfilt, sosfiltfilt

from .dataarray_factories import make_time_series, make_time_series
    

class Calculator:

    def __init__(self, config):
        self.config = config
        self.parent_signal = None

    # this method sometimes gets overwritten
    def get_child_signal_class(self, parent_signal):
        return getattr(parent_signal, 'output_signal_class', type(parent_signal))

    def __call__(self, parent_signal):
        self.parent_signal = parent_signal
        transform = partial(self._apply, **self.config)
        child_signal_class = self.get_child_signal_class(parent_signal)
        return child_signal_class(
            parent=parent_signal,
            transform=transform,
            config=self.config
        )
    

class Normalize(Calculator):

    def __init__(self, method="rms"): 
        self.config = {"method": method}
    
    def _apply(self, data, method):
        if method == "rms":
            result = data / np.sqrt(np.mean(data ** 2))
        elif method == "zscore":
            result = (data - np.mean(data)) / np.std(data)
        elif method == 'minmax':
            result = (data - data.min()) / (data.max() - data.min())
        else:
            raise ValueError("Unknown normalize method")
        
        da = make_time_series(result, self.parent_signal.sampling_rate)
        
        return da
        

class Filter(Calculator):
    def __init__(self, filter_config):
        self.config = {"filter_config": filter_config}
    
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
            
    
    def _apply(self, data, filter_config):

        method = filter_config["method"]
        fs = self.parent_signal.sampling_rate
        
        if method == "iir_notch":
            sos = self._design_sos(fs=fs.magnitude.item(), **filter_config)
        else:
            # TODO: implement other filter types
            pass
        
        result = sosfiltfilt(sos, data)

        return make_time_series(result, fs)
    


    

        





