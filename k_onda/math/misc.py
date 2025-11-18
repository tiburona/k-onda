import numpy as np

def pool(vals, weights=None, normalize=True):
    """
    Weighted average across the first axis of `vals` (list/array of shape (n, f)),
    ignoring NaNs per-frequency. If all inputs at a frequency are NaN, returns NaN there.
    
    Parameters
    ----------
    vals : Sequence[np.ndarray] or np.ndarray
        Each item is a 1-D array of the same length (frequencies).
    weights : Sequence[float] or np.ndarray, optional
        Length n; weight for each row in `vals`. Defaults to uniform.
    normalize : bool, optional
        If True, pre-normalize weights to sum to 1 globally (not required for correctness).
    """
    V = np.asarray(vals, dtype=float)            # shape (n, f)
    if V.ndim != 2:
        raise ValueError("`vals` must stack to a 2D array of shape (n, f).")
    n, f = V.shape

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.shape[0] != n:
            raise ValueError("`weights` must be 1-D with length equal to len(vals).")

    if normalize:
        s = w.sum()
        if s != 0:
            w = w / s

    # Mask of valid entries and effective column-wise weights
    mask = ~np.isnan(V)                          # (n, f)
    w_col = w[:, None] * mask                    # zero out weights where V is NaN  → (n, f)
    sum_w = w_col.sum(axis=0)                    # per-frequency weight sum → (f,)

    # Replace NaNs with 0 so they contribute nothing to the numerator
    V_filled = np.where(mask, V, 0.0)

    num = (V_filled * w_col).sum(axis=0)         # (f,)
    out = np.divide(num, sum_w, out=np.full(f, np.nan), where=sum_w != 0)
    return out