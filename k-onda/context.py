import copy

from utils import to_hashable


class Context:
    """
    A Context stores information about, well, the context and communicates it to its subscribers when it changes.
    """
    def __init__(self):
        self.observers = []
        self.cache_id = None
        self.vals = {}
        self.old_vals = {}

    def subscribe(self, observer):
        self.observers.append(observer)

    def notify(self, name):
        for observer in self.observers:
            observer.update(name)

    def set_val(self, name, new_val):
        # Compare with old value if it exists
        self.update_dicts(name, new_val)
        self.cache_id = to_hashable(self.vals)
        self.notify(name)

    def set_vals(self, name_val_pairs):
        for name, new_val in name_val_pairs:
            old_val = self.old_vals.get(name)
            if old_val is not None and self._compare(old_val, new_val):
                return
            self.update_dicts(name, new_val)
        self.cache_id = to_hashable(self.vals)
        self.notify('_'.join([name for name, _ in name_val_pairs]))

    def update_dicts(self, name, new_val):
        old_val = self.old_vals.get(name)
        if old_val is not None and self._compare(old_val, new_val):
            return

        # Update the value and notify observers
        self.vals[name] = new_val
        self.old_vals[name] = copy.deepcopy(new_val)  # Store a deep copy

    def _compare(self, old_val, new_val):
        if isinstance(old_val, dict):
            return old_val.items() == new_val.items()
        else:
            return old_val == new_val


class Subscriber:

    def subscribe(self, context):
        setattr(self, 'context', context)
        context.subscribe(self)


experiment_context = Context()
