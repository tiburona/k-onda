import functools
import re


def smart_title_case(s):
    lowercase_words = {'a', 'an', 'the', 'at', 'by', 'for', 'in', 'of', 'on', 'to', 'up', 'and', 'as', 'but', 'or', 'nor', 'is'}
    words = re.split(r'(\W)', s)  # Split string on non-alphanumeric characters, preserving delimiters
    title_words = [word if word.lower() not in lowercase_words or i == 0 or i == len(words) - 1 
                   else word.lower() 
                   for i, word in enumerate(words)]
    title = ''.join(title_words)
    return re.sub(r"[A-Za-z]+('[A-Za-z]+)?",
                  lambda mo: mo.group(0)[0].upper() + mo.group(0)[1:].lower() if not mo.group(0).isupper() else mo.group(0),
                  title)


import numpy as np

def cache_method(method):
    cache_name = "_cache_" + method.__name__

    def to_hashable(item, max_depth=5):
        if max_depth < 0:
            raise ValueError("Max recursion depth exceeded while trying to convert to hashable")

        if isinstance(item, dict):
            return tuple(sorted((k, to_hashable(v, max_depth - 1)) for k, v in item.items()))
        elif isinstance(item, (list, set)):
            return tuple(to_hashable(i, max_depth - 1) for i in item)
        elif isinstance(item, np.ndarray):
            return tuple(to_hashable(i, max_depth - 1) for i in item.tolist())
        else:
            return item

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        cache = getattr(self, cache_name, {})
        cache_key = (tuple(to_hashable(arg) for arg in args), tuple(sorted(kwargs.items())))
        if cache_key not in cache:
            cache[cache_key] = method(self, *args, **kwargs)
            setattr(self, cache_name, cache)
        return cache[cache_key]

    return wrapper


