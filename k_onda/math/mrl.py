import numpy as np
from .hilbert import apply_hilbert_to_padded_data

def compute_phase(data, pad_len):
    analytic_signal = apply_hilbert_to_padded_data(data, pad_len)
    return np.angle(analytic_signal)


def compute_mrl(alpha, w, dim):
    # weighted vector sum (weights are 1 or NaN)
    sx = np.nansum(w * np.cos(alpha), axis=dim, keepdims=True)
    sy = np.nansum(w * np.sin(alpha), axis=dim, keepdims=True)
    wsum = np.nansum(w, axis=dim, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.sqrt(sx**2 + sy**2) / wsum

    # require at least 2 spikes
    count = np.nansum(w, axis=dim, keepdims=True)
    r = np.where(count < 2, np.nan, r)

    # drop the reduced axis
    return np.squeeze(r, axis=dim)