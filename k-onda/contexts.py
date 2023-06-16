import numpy as np
import functools


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


class Context:
    def __init__(self):
        self.observers = []
        self.cache_id = None

    def subscribe(self, observer):
        self.observers.append(observer)

    def notify(self):
        for observer in self.observers:
            observer.update(self)


class OptsContext(Context):
    def __init__(self):
        super().__init__()
        self.opts = None
        self.selected_trial_indices = None

    def set_opts(self, new_opts):
        self.opts = new_opts
        self.selected_trial_indices = list(range(150))[slice(*self.opts.get.trials)]  # select only the trials indicated in opts
        self.cache_id = to_hashable(new_opts)
        self.notify()


class NeuronTypeContext(Context):
    def __init__(self):
        super().__init__()
        self.neuron_type = None

    def set_opts(self, neuron_type):
        self.neuron_type = neuron_type
        self.cache_id = neuron_type
        self.notify()


def cache_method(method):
    cache_name = "_cache_" + method.__name__

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        cache = getattr(self, cache_name, {})
        cache_key = (id(self), method.__name__, tuple(to_hashable(arg) for arg in args), tuple(sorted(kwargs.items())))
        if cache_key not in cache:
            cache[cache_key] = method(self, *args, **kwargs)
            setattr(self, cache_name, cache)
        return cache[cache_key]

    return wrapper
