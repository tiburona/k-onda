from functools import lru_cache, partial
import numpy as np
import pint
from scipy.signal import iirnotch, tf2sos, sosfreqz, filtfilt, sosfilt, sosfiltfilt

from .dataarray_factories import make_time_series
from .central import ureg



class Transform:
    """A transform function with optional metadata"""
    def __init__(self, fn, padlen=None):
        self.fn = fn
        self.padlen = padlen or {}

    def __call__(self, data):
        return self.fn(data)


class Calculator:

    def __init__(self, config):
        self.config = config

    def __call__(self, parent_signal):
        child_signal_class = self._get_child_signal_class(parent_signal)
        apply_kwargs = self._get_apply_kwargs(parent_signal)
        transform = self._get_transform(parent_signal, apply_kwargs)
        
        return child_signal_class(
            parent=parent_signal,
            transform=transform,  
            calculator=self
        )
    
    def _get_transform(self, _, apply_kwargs):
        return Transform(partial(self._apply, **apply_kwargs))
    
    def _get_child_signal_class(self, parent_signal):
        return getattr(parent_signal, 'output_signal_class', type(parent_signal))
    
    def _get_apply_kwargs(self, _):
         # Default: just pass config
        return {"config": self.config}
    

class PaddingCalculator(Calculator):

    def _get_transform(self, parent_signal, apply_kwargs):
        padlen = self._compute_padlen(parent_signal, apply_kwargs)
        transform = Transform(partial(self._apply, **apply_kwargs), padlen=padlen)
        return transform

    def _compute_padlen(self, parent_signal, apply_kwargs):
        return {}
    

class Normalize(Calculator):

    def __init__(self, method="rms"): 
        self.method = method

    def _get_apply_kwargs(self, parent_signal):
        return {
            'sampling_rate': parent_signal.sampling_rate
        }
    
    def _apply(self, data, sampling_rate):
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
            sampling_rate=sampling_rate)
        
        return da
        

class Filter(PaddingCalculator):

    def __init__(self, filter_config):
        self.config = filter_config
    
    def _get_apply_kwargs(self, parent_signal):
        designed_filter = self.design_filter(parent_signal)
        fs = parent_signal.sampling_rate
        return {
            'fs': fs,
            'designed_filter': designed_filter
        }
    
    def design_filter(self, parent_signal):
        fs = parent_signal.sampling_rate
        # todo: add in other kinds of filters
        return self._design_sos(fs=fs.magnitude, **self.config)
        
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
        
    def _compute_padlen(self, parent_signal, apply_kwargs):
        fs = parent_signal.sampling_rate.magnitude
        designed_filter = apply_kwargs['designed_filter']

        # Generate impulse response
        N = int(fs)  # 1 second worth of samples
        impulse = np.zeros(N)
        impulse[0] = 1.0
        h = sosfilt(designed_filter, impulse)

        # Find where it decays below some threshold 
        threshold = 1e-3  # -60 dB relative to peak
        peak = np.max(np.abs(h))
        settled = np.where(np.abs(h) > threshold * peak)[0]
        pad_needed = settled[-1] if len(settled) > 0 else 0
        pad_seconds = pad_needed/fs * ureg.s

        return {"time": (pad_seconds, pad_seconds)}
            
    def _apply(self, data, fs, designed_filter):
        axis = data.dims.index('time')
        result = sosfiltfilt(designed_filter, data, axis=axis)
        return make_time_series(result, fs, start=data['time'][0].item())
    

class Threshold(Calculator):

    def __init__(self, comparison, threshold):
        self.threshold = threshold
        self.comparison = comparison 
        self.operations = {
            'gt': lambda data, threshold: data > threshold,
            'lt': lambda data, threshold: data < threshold,
            'ge': lambda data, threshold: data >= threshold,
            'le': lambda data, threshold: data <= threshold
        }

    def _get_child_signal_class(self, _):
        from .signal import ValidityMask
        return ValidityMask
    
    def _get_apply_kwargs(self, parent_signal):
        return {
            'threshold': self.threshold,
            'comparison': self.comparison
        }
    
    def _apply(self, data, threshold, comparison):
        return self.operations[comparison](data, threshold)


class BinaryCalculatorMixin:
       
    def get_and_validate_sig_overlap(self, parent_data, other_data, dims=None):
        """
        Find overlapping region on shared dimensions.

        Args:
            dims: Dimension(s) to align on.  If None, uses all shared dimensions.
                  Can be a string ('time') or list (['time', 'frequency'])
        """

        shared_dims = set(parent_data.dims) & set(other_data.dims)

        # Determine which dims to align on
        if dims is None:
            dims = shared_dims
        else:
            dims = {dims} if isinstance(dims, str) else set(dims)
            dims = dims & shared_dims  # Only use dims that exist in both

        if not dims:
            raise ValueError("No shared dimensions to align on")
        
        # Build selection slices for each dimension
        slices = {}
        for dim in dims:
            coord_parent = parent_data.coords[dim]
            coord_other = other_data.coords[dim]

            overlap_start = max(coord_parent[0].item(), coord_other[0].item())
            overlap_end = min(coord_parent[-1].item(), coord_other[-1].item())

            if overlap_start >= overlap_end:
                raise ValueError(f"No overlap on dimension '{dim}'")
            
            slices[dim] = slice(overlap_start, overlap_end)
        
        # Select overlapping regions
        parent_overlap = parent_data.sel(**slices)
        other_overlap = other_data.sel(**slices)

        # Validate lengths match on aligned dimensions
        for dim in dims:
            if len(parent_overlap.coords[dim]) != len(other_overlap.coords[dim]):
                raise ValueError(f"Signals have different sampling on '{dim}'")
       
        return parent_overlap, other_overlap
    
    def validate_sig_types(self, signals):
        from .signal import BinarySignal
        for signal in signals:
            if not isinstance(signal, BinarySignal):
                raise TypeError(f"{signal} is not of type BinarySignal.")
            

class Intersection(Calculator, BinaryCalculatorMixin):

    def __init__(self, tolerance_decimals=9):
        self.tolerance = 10^-tolerance_decimals
        pass
    
    def __call__(self, parent, other):
        self.validate_sig_types([parent, other])

        child_signal_class = self._get_child_signal_class(parent)

        transform = partial(self._apply, parent, other)

        return child_signal_class(
            parent=parent,
            transform=transform,  
            calculator=self
        )

    def _apply(self, parent, other):
        parent_overlap, other_overlap = self.get_and_validate_sig_overlap(parent, other)
        return parent_overlap.data & other_overlap.data


class ApplyMask(Calculator, BinaryCalculatorMixin):

    def __init__(self, mask=None):
        self.mask = mask
  

    def __call__(self, parent_signal, mask=None):
        mask = mask or self.mask
        if mask is None:
            raise ValueError("mask must be provided at init or call time")
        self.validate_sig_types([mask])
        child_signal_class = self._get_child_signal_class(parent_signal)
        apply_kwargs = {'mask': mask}
        transform = self._get_transform(parent_signal, apply_kwargs)
   
        return child_signal_class(
            parent=parent_signal,
            transform=transform,
            calculator=self
        )
    
    def _apply(self, parent_data, mask):
        # TODO: Verify xarray alignment behavior - should give masked overlap + NaN outside
        _, mask_overlap = self.get_and_validate_sig_overlap(parent_data, mask.data)
        result = parent_data.where(mask_overlap, other=np.nan)
        return result
        

      

        




