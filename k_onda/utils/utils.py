import functools
import os
import shutil
from datetime import datetime
import json
import pickle
from copy import deepcopy
import string 

from collections.abc import Iterable
import numpy as np
import h5py
import xarray as xr

DEBUG_MODE = 0

class classproperty(property):
    def __get__(self, instance, owner):
        return super().__get__(owner)

    def __set__(self, instance, value):
        return super().__set__(instance, value)


def make_class_property(attr_name, setter=True):
    def getter(cls):
        return getattr(cls, attr_name, None)  # Retrieve class-level attribute

    if setter:
        def setter(cls, value):
            setattr(cls, attr_name, value)  # Set class-level attribute
        return classproperty(getter, setter)
    else:
        return classproperty(getter)
    

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


def pad_axes_if_nec(arr, dim='row'):
    if arr.ndim == 1:
        if dim == 'row':
            return arr[np.newaxis, :]
        else:
            return arr[:, np.newaxis]
    else:
        return arr


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
    else:
        # Return the value as is if it's already serializable
        return val
    

def formatted_now():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")
    

def safe_get(d, keys, default=None):
    """
    Safely get a value from a nested dictionary using a list of keys.
    
    :param d: The dictionary to search.
    :param keys: A list of keys representing the path to the desired value.
    :param default: The default value to return if any key is missing.
    :return: The value found at the specified path or the default value.
    """
    assert isinstance(keys, list), "keys must be provided as a list"
    
    for key in keys:
        try:
            if isinstance(d, dict):
                d = d.get(key, default)
            else:
                return default
        except Exception:
            return default
    return d

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


def collect_dict_references(obj, prefix="root", refs=None):
    """Collect all dictionary references (with their paths) from obj.
    Returns a dict mapping: id(obj) -> list_of_paths."""
    if refs is None:
        refs = {}
    if isinstance(obj, dict):
        # Record this dictionary
        refs.setdefault(id(obj), []).append(prefix)
        # Recurse into dictionary values
        for k, v in obj.items():
            collect_dict_references(v, f"{prefix}.{k}", refs)
    elif isinstance(obj, list):
        # Recurse into list items
        for i, item in enumerate(obj):
            collect_dict_references(item, f"{prefix}[{i}]", refs)
    # For other types, do nothing
    return refs

 

def print_common_dict_references(obj1, obj2):
    # Collect references from both objects
    refs1 = collect_dict_references(obj1, prefix="obj1")
    refs2 = collect_dict_references(obj2, prefix="obj2")
    
    # Find common dictionary references
    common_ids = set(refs1.keys()) & set(refs2.keys())
    
    if not common_ids:
        print("No common dictionary references found.")
    else:
        for cid in common_ids:
            paths_in_obj1 = refs1[cid]
            paths_in_obj2 = refs2[cid]
            print(f"Common dictionary id={cid} found:")
            print(f"  In obj1 at: {paths_in_obj1}")
            print(f"  In obj2 at: {paths_in_obj2}")


def is_truthy(obj):
    if isinstance(obj, xr.DataArray):
        if obj.ndim == 0:
            return bool(obj.item())  
        return obj.size > 0 and any(is_truthy(el) for el in obj.values)

    elif isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return bool(obj.item())
        return obj.size > 0 and any(is_truthy(el) for el in obj)

    elif isinstance(obj, list):
        return len(obj) > 0 and any(is_truthy(el) for el in obj)

    else:
        return bool(obj)


def is_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))



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