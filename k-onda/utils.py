import functools
import numpy as np
import os
import shutil
from datetime import datetime

DEBUG_MODE = 2


"""Cache Utils"""


def to_hashable(item, max_depth=5):
    """Converts a non hashable input into a hashable type for the purpose of using it as a part of the key in an
    instance's cache of calculated values."""
    if max_depth < 0:
        raise ValueError("Max recursion depth exceeded while trying to convert to hashable")

    if isinstance(item, dict):
        return tuple(sorted((k, to_hashable(v, max_depth - 1)) for k, v in item.items()))
    elif isinstance(item, (list, set)):
        return tuple(to_hashable(i, max_depth - 1) for i in item)
    elif isinstance(item, np.ndarray):
        return tuple(to_hashable(i, max_depth - 1) for i in item.tolist())
    elif hasattr(item, '__dict__'):  # Check if item is an object
        return id(item)  # return the memory address of the object
    else:
        return item


def cache_method(method):
    """
    Decorator that allows the results of a method's calculation to be stored in the instance cache.
    """

    if DEBUG_MODE == 2:
        return method

    cache_name = "_cache_" + method.__name__

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # If debug_mode is set to 2, the decorator will do nothing and just call the method.
        if self.data_opts.get('debug_mode') == 2:
            return method(self, *args, **kwargs)

        cache = getattr(self, cache_name) if hasattr(self, cache_name) else {}

        context_key = getattr(self, 'context').cache_id if hasattr(self, 'context') else ''

        cache_key = (id(self), context_key, method.__name__, tuple(to_hashable(arg) for arg in args),
                     tuple(sorted(kwargs.items())))

        if cache_key not in cache:
            cache[cache_key] = method(self, *args, **kwargs)
            setattr(self, cache_name, cache)
        return cache[cache_key]

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


def find_ancestor_attribute(obj, attr_name):
    current_obj = obj

    while hasattr(current_obj, 'parent'):
        if hasattr(current_obj, attr_name):
            return getattr(current_obj, attr_name)
        current_obj = current_obj.parent
    return None


def find_ancestor_id(obj, ancestor_type):
    current_obj = obj
    while hasattr(current_obj, 'parent'):
        if current_obj.name == ancestor_type:
            return current_obj.identifier
        current_obj = current_obj.parent
    return None


def get_ancestors(obj):
    """Fetch all ancestors."""

    if not hasattr(obj, 'parent'):
        return []

    # Include the current object and then get the ancestors of its parent
    return [obj] + obj.parent.ancestors


def get_descendants(obj, level=None):
    descendants = []
    # If the object does not have children, return the empty list
    if not hasattr(obj, 'children') or not obj.children:
        return descendants
    # Recursively get the descendants of each child
    for child in obj.children:
        # If level is None or the child's name matches level, add the child to the list
        if level is None or child.name == level:
            descendants.append(child)
        # Regardless, continue to check the child's descendants
        descendants.extend(get_descendants(child, level))
    return descendants



