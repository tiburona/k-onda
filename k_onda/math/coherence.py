import numpy as np
from scipy.signal import coherence, welch, csd

def calc_coherence(x, y, sampling_rate, low, high,
                   nperseg=2000, min_segments=8, overlap=0.5,
                   window='hann', detrend='constant', from_spectra=2):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1D.")
    n = min(x.size, y.size)
    if n < 4:
        raise ValueError("Signals too short.")

    # Drop any samples with NaNs in either channel (keeps them aligned)
    keep = np.isfinite(x[:n]) & np.isfinite(y[:n])
    x = x[:n][keep]
    y = y[:n][keep]
    n = x.size
    if n < 4:
        raise ValueError("Not enough finite samples after NaN filtering.")

    # Choose nperseg to ensure enough segments; fall back gracefully
    nper = min(nperseg, n)
    target_segments = max(2, int(min_segments))
    if n // nper < target_segments:
        nper = max(256, n // target_segments)  # 256 is a reasonable lower bound
        nper = max(8, min(nper, n))            # final clamp

    noverlap = int(round(overlap * nper))
    noverlap = min(noverlap, nper - 1)        # must be < nper

    if not from_spectra:
        f, Cxy = coherence(
            x, y, fs=sampling_rate, window=window,
            nperseg=nper, noverlap=noverlap,
            detrend=detrend
        )
    else:
        f, Cxy = coherence_from_spectra(
            x, y, fs=sampling_rate, window=window,
            nperseg=nper, noverlap=noverlap,
            detrend=detrend
        )

    mask = (f >= low) & (f <= high)
    return f[mask], Cxy[mask]


import numpy as np


def coherence_from_spectra(x, y, fs, nperseg, noverlap, window='hann', detrend='constant'):
    f, Sxx = welch(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
    _, Syy = welch(y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
    _, Sxy = csd(x, y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
    Cxy = (np.abs(Sxy)**2) / (Sxx * Syy)
    return f, Cxy

