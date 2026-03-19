import numpy as np
import h5py
import re
from pathlib import PosixPath
from copy import deepcopy

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


def recursive_update(d1, d2):
    """
    Recursively update dictionary d1 with values from dictionary d2.
    If a key in d1 and d2 has a dictionary as a value, merge them recursively.
    """
    for key, value in d2.items():
        if isinstance(value, dict) and key in d1 and isinstance(d1[key], dict):
            recursive_update(d1[key], value)  # Recursively update inner dictionary
        else:
            d1[key] = deepcopy(value)  # Overwrite or add new key-value pair
    return d1