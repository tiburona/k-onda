from utils import to_hashable


class Context:
    """
    A Context stores information about, well, the context and communicates it to its subscribers when it changes.
    """
    def __init__(self, name):
        self.name = name
        self.observers = []
        self.cache_id = None
        self.val = None

    def subscribe(self, observer):
        self.observers.append(observer)

    def notify(self):
        for observer in self.observers:
            observer.update(self)

    def set_val(self, new_val):
        if isinstance(self.val, dict) and isinstance(new_val, dict):
            if self.val.items() == new_val.items():
                return
        else:
            if new_val in [self.val, None]:
                return
        self.val = new_val
        self.cache_id = to_hashable(new_val)
        self.notify()


neuron_type_context = Context('neuron_type_context')
data_type_context = Context('data_type_context')

