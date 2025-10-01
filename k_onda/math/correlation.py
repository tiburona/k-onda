import numpy as np
from scipy.signal import correlate, correlation_lags

def normalized_xcorr(x, y, fs=None, mode="full"):
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



def pearson_xcorr(x, y, fs=None, min_overlap_frac=0.5, min_overlap=None):
    """
    Pearson cross-correlation at each lag (per-lag mean/std over the overlap).
    Returns (corr, lags) where lags are in seconds if fs is given, else samples.

    Args:
        x, y: 1D arrays of equal length.
        fs: sampling rate (Hz). If provided, lags are in seconds.
        min_overlap_frac: ignore lags with overlap < this fraction of len(x).
        min_overlap: integer override for minimum overlap samples (takes precedence).

    Notes:
        - Equivalent to np.corrcoef applied to each overlapping window per lag.
        - Robust to edge bias that shows up when using global z-scoring.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be same-length 1D arrays.")
    n = x.size

    # Lags: -(n-1) .. +(n-1)
    lags = correlation_lags(n, n, mode="full")

    
    i0 = np.where(lags >= 0, lags, 0)  # When lags are positive, y leads. the beginning of x is equivalent to the number of lags. When lags are negative, x leads, and the beginning of x is the first index.
    i1 = np.where(lags >= 0, n,     n + lags)  # When lags are positive, the end of x is equivalent to the size of x.  In other words the index starts at lags and ends at the end.  When lags are negative, and x leads, the second x index is n + lags (i.e. n - |lags|)
    j0 = np.where(lags >= 0, 0,     -lags)  # When lags are positive, y leads.  The beginning of y is 0.  When lags are negative the beginning of y is | lags | 
    j1 = np.where(lags >= 0, n - np.maximum(lags, 0), n) # When lags are positive, y leads, and the second y index is n - lags.  When lags are negative, x leads, and the second y index is n.
    m = (i1 - i0).astype(np.int64)  # overlap length per lag

    # Min-overlap mask
    if min_overlap is None:
        min_overlap = max(1, int(np.ceil((min_overlap_frac or 0.0) * n)))
    keep = m >= min_overlap

    # Cumulative sums for fast windowed stats
    cx  = np.concatenate(([0.0], np.cumsum(x)))
    cy  = np.concatenate(([0.0], np.cumsum(y)))
    cx2 = np.concatenate(([0.0], np.cumsum(x * x)))
    cy2 = np.concatenate(([0.0], np.cumsum(y * y)))

    # Overlap sums (vectorized)
    Sx  = cx[i1]  - cx[i0]
    Sy  = cy[j1]  - cy[j0]
    Sxx = cx2[i1] - cx2[i0]
    Syy = cy2[j1] - cy2[j0]

    mx = Sx / m
    my = Sy / m

    # Raw cross-sum over overlaps (not demeaned)
    xy_full = correlate(x, y, mode="full", method="auto")

    # Per-lag covariance and variances over the overlap
    cov  = xy_full - m * mx * my
    varx = Sxx - m * mx * mx
    vary = Syy - m * my * my
    denom = np.sqrt(np.maximum(varx * vary, 0.0))

    corr = np.full(2 * n - 1, np.nan, dtype=float)
    valid = keep & (denom > 0)
    corr[valid] = cov[valid] / denom[valid]

    if fs is not None:
        return corr, lags / float(fs)
    return corr, lags
