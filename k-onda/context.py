from utils import to_hashable


class Context:
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
        if new_val == self.val:
            return
        self.val = new_val
        self.cache_id = to_hashable(new_val)
        self.notify()



