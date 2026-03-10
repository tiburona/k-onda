
import h5py
import numpy as np
from collections.abc import MutableMapping


def group_to_dict(group):
    result = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = group_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            val = item[()]
            if isinstance(val, np.ndarray) and val.size == 1:
                result[key] = val.item()
            else:
                result[key] = val
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


class DictDelegator(MutableMapping):
    _delegate_attr: str  # subclass sets this

    def __getitem__(self, key):
        return getattr(self, self._delegate_attr)[key]

    def __setitem__(self, key, value):
        getattr(self, self._delegate_attr)[key] = value

    def __delitem__(self, key):
        del getattr(self, self._delegate_attr)[key]

    def __iter__(self):
        return iter(getattr(self, self._delegate_attr))

    def __len__(self):
        return len(getattr(self, self._delegate_attr))