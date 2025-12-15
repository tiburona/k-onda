import functools
import os
import shutil
from datetime import datetime
import json
import pickle
from copy import deepcopy
import importlib.util
import pathlib
import re
from collections.abc import Iterable
import numpy as np
import h5py
import xarray as xr
import numbers

DEBUG_MODE = 0
    

def sorted_prop(key):
    """Decorator to automatically fetch the sort key and apply sorting."""
    def decorator(func):
        @property
        @functools.wraps(func)
        def wrapper(self):
            items = func(self)
            sort = self.calc_opts.get('sort', {}).get(key)
            return self.sort(items, sort)
        return wrapper
    return decorator


def cache_method(method):
    """
    Decorator that allows the results of a method's calculation to be stored in the instance cache.
    """

    if DEBUG_MODE == 2:
        return method

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):

        # TODO it seems like this is doing nothing right now. Think about putting it back. 
        cache_level = self.calc_opts.get('cache', 2) 
        if cache_level == -1: # Do not cache
            return method(self, *args, **kwargs)

        # Define a level beyond which recursive functions don't cache
        if 'level' in kwargs and isinstance(kwargs['level'], int) and kwargs['level'] > cache_level:
            return method(self, *args, **kwargs)
            
        # TODO: make sure as selected period types evolve that there can't be any
        # key overlap between different calculations  
        
        if hasattr(self, 'period_type'):
            period_keys = [self.period_type]
        else:
            period_keys = self.selected_period_types + [self.selected_period_type]
          
        key_list = [self.calc_type, method.__name__, 
                    self.selected_neuron_type, *period_keys, 
                    *(f'{k}_{v}' for k, v in self.selected_conditions.items()), 
                    *(arg for arg in args), 
                    *(kwarg for kwarg in kwargs)]

        for obj in list(reversed(self.ancestors)):
            key_list.append(obj.name)
            key_list.append(obj.identifier)
        
        # TODO: does this work for selected period types?
        if self.selected_period_type in self.calc_opts.get('periods', {}):
            key_list.append(str(self.calc_opts['periods'][self.selected_period_type]))

        key = '_'.join([str(k) for k in key_list])
            
        if key not in self.cache[self.name]:
            self.cache[self.name][key] = method(self, *args, **kwargs)

        return self.cache[self.name][key]

    return wrapper


def log_directory_contents(log_directory):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_subdirectory = os.path.join(log_directory, timestamp)
    os.makedirs(new_subdirectory, exist_ok=True)

    for item in os.listdir(current_directory):
        if 'venv' in item:
            continue
        s = os.path.join(current_directory, item)
        d = os.path.join(new_subdirectory, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


def range_args(lst):
    if len(lst) < 2:
        return None

    start = lst[0]
    step = lst[1] - lst[0]

    for i in range(2, len(lst)):
        if lst[i] - lst[i-1] != step:
            return None

    return start, lst[-1] + step, step


def find_ancestor_attribute(obj, ancestor_type, attribute):
    current_obj = obj
    while hasattr(current_obj, 'parent'):
        if current_obj.name == ancestor_type or (
            ancestor_type == 'any' and hasattr(current_obj, attribute)
            ):
            return getattr(current_obj, attribute)
        current_obj = current_obj.parent
    return None


def to_serializable(val):
    """
    Convert non-serializable objects to serializable format.
    """
    if isinstance(val, range):
        # Convert range to list
        return list(val)
    elif isinstance(val, tuple):
        # Convert tuple to list
        return list(val)
    elif isinstance(val, dict):
        # Recursively apply to dictionary items
        return {key: to_serializable(value) for key, value in val.items()}
    elif isinstance(val, list):
        # Recursively apply to each item in the list
        return [to_serializable(item) for item in val]
    elif isinstance(val, np.ndarray):
        return [to_serializable(item) for item in val]
    elif isinstance(val, np.integer):
        return int(val)
    else:
        # Return the value as is if it's already serializable
        return val
    

def to_hashable(val):
    """
    Recursively coerce containers into hashable equivalents suitable for dict keys.
    """
    if isinstance(val, (tuple, list)):
        return tuple(to_hashable(item) for item in val)
    if isinstance(val, dict):
        # sort to guarantee equal dicts produce the same hashable tuple
        return tuple(sorted((to_hashable(k), to_hashable(v)) for k, v in val.items()))
    if isinstance(val, (set, frozenset)):
        return tuple(sorted(to_hashable(item) for item in val))
    if isinstance(val, np.ndarray):
        return tuple(to_hashable(item) for item in val.tolist())
    if isinstance(val, numbers.Number) or isinstance(val, str) or val is None:
        return val
    # Fallback: use repr so objects are at least distinguishable
    return repr(val)


def formatted_now():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")
    

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


class AlwaysLast:
    def __lt__(self, other):
        return False
    def __gt__(self, other):
        return True

always_last = AlwaysLast()


def safe_get(d, keys, default=None):
    current = d
    for key in keys[:-1]:  # Traverse all but the last key
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}  # Create an empty dictionary if the key is missing or not a dict
        current = current[key]
    
    # Try to get the last key's value; if not present, set it to default
    return current.get(keys[-1], default)


def load(store_path, store_type, force_recalc=False):
        if os.path.exists(store_path) and not force_recalc:
            with open(store_path, 'rb') as f:
                if store_type == 'pkl':
                    return_val = pickle.load(f)
                else:
                    return_val = json.load(f)
                return True, return_val, store_path
        else:
            return False, None, store_path


def save(result, store_path, store_type):
    mode = 'wb' if store_type == 'pkl' else 'w'
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    with open(store_path, mode) as f:
        if store_type == 'pkl':
            return pickle.dump(result, f)
        else:
            result_str = json.dumps([arr.tolist() for arr in result])
            f.write(result_str)

    
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


def is_truthy(obj, *, zero_ok: bool = False):
    """
    Recursively test “truthiness” of lists / NumPy / xarray objects.

    Parameters
    ----------
    obj : Any
        Object to check.
    zero_ok : bool, optional
        If True, treat numeric zeros (0, 0.0, …) as truthy.
        Defaults to False (original behaviour).

    Returns
    -------
    bool
    """
    # helper: decide scalar truthiness once
    def _scalar_truth(val):
        # Special-case numerics (but not bools) when zero_ok is on
        if zero_ok and isinstance(val, numbers.Number) and not isinstance(val, bool):
            return True
        return bool(val)

    if isinstance(obj, xr.DataArray):
        if obj.ndim == 0:                       # scalar DataArray
            return _scalar_truth(obj.item())
        return (
            obj.size > 0
            and any(is_truthy(el, zero_ok=zero_ok) for el in obj.values)
        )

    elif isinstance(obj, np.ndarray):
        if obj.ndim == 0:                       # scalar ndarray
            return _scalar_truth(obj.item())
        return (
            obj.size > 0
            and any(is_truthy(el, zero_ok=zero_ok) for el in obj)
        )

    elif isinstance(obj, list):
        return (
            len(obj) > 0
            and any(is_truthy(el, zero_ok=zero_ok) for el in obj)
        )

    else:                                       # Python scalars, pandas types, etc.
        return _scalar_truth(obj)


def is_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))


import numbers
import numpy as np
import xarray as xr
import math

def contains_nan(obj):
    """
    Recursively check whether a list / NumPy / xarray object
    or scalar number contains any NaN values.

    Parameters
    ----------
    obj : Any
        Object to check.

    Returns
    -------
    bool
        True if at least one NaN is found, False otherwise.
    """
    if isinstance(obj, xr.DataArray):
        if obj.ndim == 0:  # scalar DataArray
            return math.isnan(obj.item())
        return bool(np.isnan(obj.values).any())

    elif isinstance(obj, np.ndarray):
        if obj.ndim == 0:  # scalar ndarray
            return math.isnan(obj.item())
        return bool(np.isnan(obj).any())

    elif isinstance(obj, list):
        return any(contains_nan(el) for el in obj)

    elif isinstance(obj, numbers.Number):
        return math.isnan(obj)

    else:
        return False


operations = {
            '==': lambda a, b: a == b,
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b,
            'in': lambda a, b: a in b,
            '!=': lambda a, b: a != b,
            'not in': lambda a, b: a not in b, 
            'partial_dict_match': lambda a, b: all(a.get(k) == v for k, v in b.items())
        }


def load_config_py(path_to_py_file):
    path = pathlib.Path(path_to_py_file).resolve()
    module_name = path.stem  # just the filename without .py

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config file from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_make_dir(path):
    """
    Create all intermediate-level directories needed to contain the path.
    If 'path' is a file path, create the directory that would contain it.
    """
    dir_path = path if os.path.splitext(path)[1] == '' else os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def smart_title_case(s):
    lowercase_words = {'a', 'an', 'the', 'at', 'by', 'for', 'in', 'of', 'on', 'to', 'up', 'and', 
                       'as', 'but', 'or', 'nor', 'is', 's'}
    acronyms = {'psth', 'pl', 'hpc', 'bla', 'mrl', 'il', 'bf', 'mua', 'cs'}
    tokens = re.findall(r'\b\w+\b|[^\w\s]', s)  # Find words and punctuation separately
    title_words = []

    for i, word in enumerate(tokens):
        if word.lower() in lowercase_words and i != 0 and i != len(tokens) - 1:
            title_words.append(word.lower())
        elif word.lower() in acronyms:
            title_words.append(word.upper())
        elif not word.isupper():
            title_words.append(word.capitalize())
        else:
            title_words.append(word)

    # Join words carefully to avoid adding spaces before parentheses
    title = ''
    for i in range(len(title_words)):
        if i > 0 and title_words[i] not in {')', ',', '.', '!', '?', ':'} and title_words[i - 1] not in {'(', '-', '/'}:
            title += ' '
        title += title_words[i]
    return title


def find_container_with_key(data, target_key):
    if isinstance(data, dict):
        if target_key in data:
            return data
        for value in data.values():
            result = find_container_with_key(value, target_key)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_container_with_key(item, target_key)
            if result is not None:
                return result
    return None


def standardize(num_array):
    return np.round(num_array, 8).astype(np.float64)

