import numpy as np


def scalar(value, t=float):
    return t(value.item().magnitude)


def is_uniformly_spaced(arr, tol=1e-5):
    """
    Checks if a 1D numpy array is uniformly spaced within a given tolerance.
    """
    if len(arr) <= 2:
        return True  # Arrays with 0, 1, or 2 elements are considered uniformly spaced

    # Calculate differences between consecutive elements
    diffs = np.diff(arr)
    
    # Check if all differences are close to the first difference
    # np.allclose handles potential floating-point errors
    return np.allclose(diffs, diffs[0], atol=tol)


def is_numeric(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False
    

def np_from_xr(xr_val):
    units = xr_val.pint.units
    if units is not None:
        arr = np.asarray(xr_val.pint.magnitude)
    else:
        arr = np.asarray(xr_val)  
    return arr, units