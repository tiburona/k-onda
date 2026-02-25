
import h5py
import numpy as np


def group_to_dict(group):
    result = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = group_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            result[key] = item[()]
        else:
            result[key] = item
    return result

   
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
    

def scalar(value, t=float):
    return t(value.item().magnitude)