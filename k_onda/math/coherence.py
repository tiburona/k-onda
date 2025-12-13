import numpy as np
from scipy.signal import  welch, csd
import mne
import xarray as xr

mne.set_log_level("WARNING")


def welch_psd(x, fs, **args):
    f, Sxx = welch(x, fs, **{k: v for k, v in args.items() if k in {
        "window", "nperseg", "noverlap", "detrend", "nfft", "scaling", "average"}})
    if np.ndim(Sxx) == 2:   # (n_epochs, n_freqs)
        Sxx = Sxx.mean(axis=0)
    return f, Sxx

def welch_csd(x, y, fs, **args):
    f, Sxy = csd(x, y, fs, **{k: v for k, v in args.items() if k in {
          "window", "nperseg", "noverlap", "detrend", "nfft", "scaling", "average"}})
    if np.ndim(Sxy) == 2:   # (n_epochs, n_freqs)
        Sxy = Sxy.mean(axis=0)
    return f, Sxy

def as_epochs_2d(x):
    """
    Ensure shape (n_epochs, n_times) for multitaper.

    Accepts:
    - (n_times,)
    - (n_epochs, n_times)
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")

def multitaper_psd(x, fs, output="power", avg_over_eps=True, **args):
    """
    Returns f (Hz), psd (n_freq,) averaged across epochs.
    """
    x2 = as_epochs_2d(x)

    # MNE expects (n_epochs, n_times); psd_array_multitaper returns (n_epochs, n_freqs)
    out = mne.time_frequency.psd_array_multitaper(
        x2,
        sfreq=fs,
        output=output,
        **{k: v for k, v in args.items() if k in {"bandwidth", "adaptive", "low_bias", "normalization"}},
    )
    psd_ep, freqs = out[0], out[1]
    val_to_return = psd_ep.mean(axis=0) if avg_over_eps else psd_ep
    return freqs, val_to_return

def multitaper_csd(x, y, fs, **args):
    # get complex spectra for both signals, keep epochs+tapers
    freqs, X = multitaper_psd(x, fs, output="complex", avg_over_eps=False, **args)
    _,     Y = multitaper_psd(y, fs, output="complex", avg_over_eps=False, **args)

    if X.ndim == 2:  # (n_epochs, n_freqs) no taper dim
        X = X[:, None, :]
        Y = Y[:, None, :]

    # X, Y: (n_epochs, n_tapers, n_freqs)
    Sxy_ep = np.mean(X * np.conj(Y), axis=1)         # -> (n_epochs, n_freqs)
    Sxx_ep = np.mean(np.abs(X) ** 2, axis=1)         # -> (n_epochs, n_freqs)
    Syy_ep = np.mean(np.abs(Y) ** 2, axis=1)

    Sxy = Sxy_ep.mean(axis=0)                        # -> (n_freqs,)
    Sxx = Sxx_ep.mean(axis=0)
    Syy = Syy_ep.mean(axis=0)

    return freqs, Sxy, Sxx, Syy

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



