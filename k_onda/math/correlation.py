import numpy as np
from scipy.signal import correlate, correlation_lags

def normalized_crosscorr(x, y, fs=None, mode="full"):
    """
    Normalized cross-correlation (Pearson) of two 1D arrays.
    Returns (corr, lags) where lags are in samples, or in seconds if fs is given.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be same-length 1D arrays.")

    # z-score with population std (ddof=0) so we can divide by overlap
    zx = (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0)
    zy = (y - y.mean()) / (y.std(ddof=0) if y.std(ddof=0) > 0 else 1.0)

    c = correlate(zx, zy, mode=mode, method="auto")

    if mode == "full":
        n = x.size
        lags = correlation_lags(n, n, mode="full")
        overlap = n - np.abs(lags)               # samples contributing at each lag
        corr = c / overlap
    elif mode == "same":
        # same-length output; approximate overlap near edges
        n = x.size
        lags = correlation_lags(n, n, mode="same")
        overlap = n - np.abs(lags)
        overlap[overlap < 1] = 1
        corr = c / overlap
    elif mode == "valid":
        # zero-lag only when lengths equal
        lags = np.array([0])
        corr = c / x.size
    else:
        raise ValueError("mode must be 'full', 'same', or 'valid'")

    if fs is not None:
        return corr, lags / float(fs)
    return corr, lags