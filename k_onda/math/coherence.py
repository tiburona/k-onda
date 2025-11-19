import numpy as np
from scipy.signal import  welch, csd
import xarray as xr


def psd(x, fs, nperseg, noverlap, window='hann', detrend='constant'):
    f, Sxx = welch(x, fs, window=window, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
    return f, Sxx

def cross_spectral_density(x, y, fs, nperseg, noverlap, window='hann', detrend='constant'):
      f, Sxy = csd(x, y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
      return f, Sxy

def fisher_z_from_coherence(Cxy_da: xr.DataArray) -> xr.DataArray:
    r = xr.apply_ufunc(np.abs, Cxy_da, keep_attrs=True)                 # |coherence| in [0,1]
    r = xr.apply_ufunc(np.clip, r, 1e-12, 1 - 1e-12, keep_attrs=True)   # avoid ±inf
    z = xr.apply_ufunc(np.arctanh, r, keep_attrs=True)
    return z

def fisher_z_from_msc(msc_da: xr.DataArray) -> xr.DataArray:
    r = xr.apply_ufunc(np.sqrt, msc_da, keep_attrs=True)                # sqrt coherence
    r = xr.apply_ufunc(np.clip, r, 1e-12, 1 - 1e-12, keep_attrs=True)
    z = xr.apply_ufunc(np.arctanh, r, keep_attrs=True)
    return z

def back_transform_fisher_z(z_da: xr.DataArray) -> xr.DataArray:
    r = xr.apply_ufunc(np.tanh, z_da, keep_attrs=True)          # inverse of atanh
    r2 = xr.apply_ufunc(np.square, r, keep_attrs=True)          # square it
    return r2

def msc_from_spectra(Sxy, Sxx, Syy, eps=1e-20):
    Sxxr = np.maximum(np.real(Sxx), 0.0)
    Syyr = np.maximum(np.real(Syy), 0.0)
    denom = np.maximum(Sxxr, eps) * np.maximum(Syyr, eps)
    Cxy = (np.abs(Sxy)**2) / denom
    return np.clip(Cxy, 0.0, 1.0)



