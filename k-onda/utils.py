import re
import functools
import numpy as np
from datetime import datetime


"""Cache Utils"""


def to_hashable(item, max_depth=5):
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
    cache_name = "_cache_" + method.__name__

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        cache = getattr(self, cache_name, {})
        context_keys = (getattr(self, c).cache_id for c in ['neuron_type_context', 'data_type_context'] if
                        hasattr(self, c))
        cache_key = (id(self), context_keys, method.__name__, tuple(to_hashable(arg) for arg in args),
                     tuple(sorted(kwargs.items())))
        if cache_key not in cache:
            cache[cache_key] = method(self, *args, **kwargs)
            setattr(self, cache_name, cache)
        return cache[cache_key]

    return wrapper


"""Plot Utils"""


def formatted_now():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def smart_title_case(s):
    lowercase_words = {'a', 'an', 'the', 'at', 'by', 'for', 'in', 'of', 'on', 'to', 'up', 'and', 'as', 'but', 'or', 'nor', 'is'}
    acronyms = {'psth'}
    words = re.split(r'(\W+)', s)  # Split string on non-alphanumeric characters, preserving delimiters
    title_words = []
    for i, word in enumerate(words):
        if word.lower() in lowercase_words and i != 0 and i != len(words) - 1:
            title_words.append(word.lower())
        elif word.lower() in acronyms:
            title_words.append(word.upper())
        elif not word.isupper():
            title_words.append(word.capitalize())
        else:
            title_words.append(word)
    title = ''.join(title_words)
    return title


def ac_str(s):
    for (old, new) in [('pd', 'Pandas'), ('np', 'NumPy'), ('ml', 'Matlab')]:
        s = s.replace(old, new)
