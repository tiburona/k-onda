from functools import lru_cache, partial, wraps

import numpy as np
import pint
from scipy.signal import iirnotch, tf2sos, sosfreqz, filtfilt, sosfilt, sosfiltfilt, medfilt
import xarray as xr

from .dataarray_factories import make_time_series
from .central import ureg


def with_key_access(func):
    @wraps(func)
    def wrapper(self, data, *args, key=None, **kwargs):
        if key is None:
            return func(self, data *args, **kwargs)
        
        subset = data[key]
        result = func(self, subset, *args, **kwargs)

        new_data = data.copy()
        new_data[key] = result
        return new_data
    return wrapper


class Transform:
    """A transform function with optional metadata"""
    def __init__(self, fn, padlen=None, signal_class=None):
        self.fn = fn
        self.padlen = padlen or {}
        self.signal_class = signal_class

    def __call__(self, data):
        return self.fn(data)
    

class Transformer:
    pass


class Calculator(Transformer):

    obligate_output_class = None

    def __call__(self, parent, key=None):
  
        child_class = self.get_child_class(parent)
        kwargs = self._get_apply_kwargs(parent, key=key)

        if parent.is_bundle:
            kwargs.update(
                {'signal_class': self.obligate_output_class or 
                 self.get_output_class(parent.signals[0])}
                 )
            
        transform = self._get_transform(parent, **kwargs)

        child_signal = child_class(parent=parent, transform=transform, calculator=self)

        if not parent.is_bundle:
            child_signal.origin = getattr(parent, 'origin', None)
        
        return child_signal
    
    
    def _get_transform(self, _, **kwargs):
        signal_class = kwargs.pop('signal_class', None)
        return Transform(
            partial(self._apply, **kwargs), 
            signal_class=signal_class)
    
    def get_child_class(self, parent):
        # If we're operating on a SignalBundle, we'll return a SignalBundle.  
        # If this calculator has an obligate Signal type to return, 
        # we'll return one of those.
        # Otherwise, we'll query the parent signal to get the output class.
        if parent.is_bundle:
            return type(parent)
        else:
            return self.obligate_output_class or self.get_output_class(parent)
        
    @staticmethod
    def get_output_class(entity):
        return getattr(entity, 'output_class', type(entity))
    
    def _get_apply_kwargs(self, parent_signal, key=None):
        result = {'key': key}
        result.update(self._get_distinctive_apply_kwargs(parent_signal))
        return result
    
    def _get_distinctive_apply_kwargs(self, _):
        return {}
    

class ReduceDim(Calculator):

    def __init__(self, dim, method='mean'):
        self.dim = dim
        self.method = method

    @with_key_access
    def _apply(self, data):
        return getattr(data, self.method)(dim=self.dim)

    
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
    
    def _get_apply_kwargs(self, parent_signal, key=None, dim=None):
        result = super()._get_apply_kwargs(parent_signal, key=key)
        result.update({'dim': dim})
        return result
    
    def _get_distinctive_apply_kwargs(self, _):
        return {}
    
    @with_key_access
    def _apply(self, data, dim=None):
        if self.method == "rms":
            result = data / np.sqrt((data ** 2).mean(dim=dim))
        elif self.method == "zscore":
            result = (data - data.mean(dim=dim)) / data.std(dim=dim)
        elif self.method == 'minmax':
            result = (data - data.min(dim=dim)) / (data.max(dim=dim) - data.min(dim=dim))
        else:
            raise ValueError("Unknown normalize method")
        
        result.attrs = data.attrs
        
        return result
        

class Filter(PaddingCalculator):

    def __init__(self, filter_config, dim='time'):
        self.config = filter_config
        self.dim = dim
    
    def _get_apply_kwargs(self, parent_signal, key=None):
        result = super()._get_apply_kwargs(parent_signal, key=key)
        return result
    
    def _get_distinctive_apply_kwargs(self, parent_signal):
        designed_filter = self.design_filter(parent_signal)
        return {'designed_filter': designed_filter}
    
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
    
    @with_key_access
    def _apply(self, data, designed_filter):
        dim = self.dim
        if dim != 'time':
            raise NotImplementedError(
                "You can currently only filter along the time dimension.")
        axis = data.dims.index(dim)
        result = sosfiltfilt(designed_filter, data, axis=axis)
        result = xr.DataArray(
            result, 
            coords=data.coords, 
            dims=data.dims, 
            attrs=data.attrs
            )
        return result
    



    
    

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

    def get_child_class(self, _):
        from .signal import ValidityMask
        return ValidityMask
    
    @with_key_access
    def _apply(self, data):
        return self.operations[self.comparison](data, self.threshold)


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

        child_signal_class = self.get_child_class(parent)

        transform = partial(self._apply, parent, other)

        return child_signal_class(
            parent=parent,
            transform=transform,
            origin=(parent.origin, other.origin),  
            calculator=self
        )

    @with_key_access
    def _apply(self, parent, other):
        parent_overlap, other_overlap = self.get_and_validate_sig_overlap(parent, other)
        return parent_overlap.data & other_overlap.data


class ApplyMask(Calculator, BinaryCalculatorMixin):

    def __init__(self, mask=None):
        self.mask = mask
  
    # TODO: all these calls need to be verified to work with SignalBundle and not break acyclic graph representation
    def __call__(self, parent_signal, mask=None):
        mask = mask or self.mask
        if mask is None:
            raise ValueError("mask must be provided at init or call time")
        self.validate_sig_types([mask])
        child_signal_class = self.get_child_class(parent_signal)
        apply_kwargs = {'mask': mask}
        transform = self._get_transform(parent_signal, apply_kwargs)
   
        return child_signal_class(
            parent=parent_signal,
            transform=transform,
            origin=(parent_signal.origin, mask.origin),
            calculator=self
        )
    
    @with_key_access
    def _apply(self, parent_data, mask):
        # TODO: Verify xarray alignment behavior - should give masked overlap + NaN outside
        _, mask_overlap = self.get_and_validate_sig_overlap(parent_data, mask.data)
        result = parent_data.where(mask_overlap, other=np.nan)
        return result
        

class MedianFilter(Calculator):

    def __init__(self, kernel_sizes):
        self.kernel_sizes = kernel_sizes

    def _get_distinctive_apply_kwargs(self, _):
        return {
            'kernel_sizes': self.kernel_sizes
        }
        
    @with_key_access
    def _apply(self, data, kernel_sizes):
        # kernel_sizes is a dictionary like {'samples': 5}
        kernel_size = tuple(
            kernel_sizes.get(dim, 1) for dim in data.dims
        )
        result = medfilt(data, kernel_size=kernel_size)

        result = xr.DataArray(
            result, 
            coords=data.coords, 
            dims=data.dims, 
            attrs=data.attrs
            )
        return result



class GroupSignals(Transformer):
    """GroupSignals is passed a Collection and returns a SignalBundle.  The transform concatenates
    data along the provided stack_dim so that downstream calculations can be vectorized."""

    def __init__(self, stack_dim=None):
        self.stack_dim = stack_dim
        
    def __call__(self, parent):
        from .signal import SignalBundle
        
        return SignalBundle(
            parent=parent,
            transform=self._apply,
            calculator=self
        )
    
    def _gather_datasets(self, signals):
        keys = signals[0].data.keys()
        data = {}
        boundaries = [0]

        for i, key in enumerate(keys):
            arrays = []
            for signal in signals:
                arr = signal.data[key]
                arrays.append(arr)
                increment = arr.sizes[self.stack_dim] if self.stack_dim else 1
                if i == 0:
                    boundaries.append(boundaries[-1] + increment)
            
            data[key] = xr.concat(
                arrays, 
                dim=self.stack_dim or 'members', 
                combine_attrs='no_conflicts'
                )

        dataset = xr.Dataset(data)
        dataset.attrs['boundaries'] = boundaries
        dataset.attrs['stack_dim'] = self.stack_dim

        return dataset

    def _gather_arrays(self, signals):
        arrays = []
        boundaries = [0]
        
        for signal in signals:
            arr = signal.data
            arrays.append(arr)
            increment = arr.sizes[self.stack_dim] if self.stack_dim else 1
            boundaries.append(boundaries[-1] + increment)

        data = xr.concat(
            arrays, 
            dim=self.stack_dim or 'members', 
            combine_attrs='no_conflicts'
            )
        
        data.attrs['boundaries'] = boundaries
        data.attrs['stack_dim'] = self.stack_dim
        
        return data
        
    def _apply(self, signals):
        if isinstance(signals[0].data, xr.Dataset):
            return self._gather_datasets(signals)
        else:
            return self._gather_arrays(signals)
    

class SplitSignals(Transformer):
    
    def __init__(self, stack_dim=None):
        self.stack_dim = stack_dim or 'members'

    def get_child_class(self):
        from .sources import Collection
        return Collection

    def __call__(self, signal_bundle):
        signals = []

        for i in range(len(signal_bundle.signals)):
            signal_class = signal_bundle.signal_class
            transform = partial(self._apply, idx=i)
            origin = signal_bundle.signals[i].origin
            signal = signal_class(
                parent=signal_bundle,
                transform=transform,
                origin=origin,
                calculator=self
            )
            signals.append(signal)

        return self.get_child_class()(signals)

    def get_child_signal_class(self, signal):
        if not hasattr(signal, 'calculator'):
            return signal.output_class
        else:
            return signal.calculator.get_child_class()
    
    def _apply(self, data, idx):
        boundaries = data.attrs['boundaries']
        start, end = boundaries[idx], boundaries[idx + 1]
        return data.isel({self.stack_dim: slice(start, end)})