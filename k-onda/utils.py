import functools
import numpy as np
import os
import shutil
from datetime import datetime
import h5py


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






def cache_method(method):
    """
    Decorator that allows the results of a method's calculation to be stored in the instance cache.
    """

    if DEBUG_MODE == 2:
        return method

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):

        cache_level = self.calc_spec.get('cache', 2)
        if cache_level == -1: # Do not cache
            return method(self, *args, **kwargs)

        # Define a level beyond which recursive functions don't cache
        if 'level' in kwargs and isinstance(kwargs['level'], int) and kwargs['level'] > cache_level:
            return method(self, *args, **kwargs)
            
        key_list = [self.calc_type, method.__name__, self.selected_neuron_type, 
                    self.selected_period_type, *(arg for arg in args), 
                    *(kwarg for kwarg in kwargs)]
        for obj in list(reversed(self.ancestors))[1:]:
            key_list.append(obj.name)
            key_list.append(obj.identifier)
        
        if self.selected_period_type in self.calc_spec.get('periods', {}):
            key_list.append(str(self.calc_spec['periods'][self.selected_period_type]))

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
