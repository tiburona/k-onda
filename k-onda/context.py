from utils import to_hashable


class Base:

    """
    Base defines three properties that are useful for Experiment and its inheritors, Plotter, and Spreadsheet.
    They all access elements of the context.
    """

    @property
    def data_opts(self):
        return (self.data_type_context.val if self.data_type_context is not None else {}) or {}

    @data_opts.setter
    def data_opts(self, opts):
        self.data_type_context.set_val(opts)

    @property
    def data_type(self):
        return self.data_opts['data_type']

    @property
    def selected_neuron_type(self):
        return self.neuron_type_context.val

    @selected_neuron_type.setter
    def selected_neuron_type(self, neuron_type):
        self.neuron_type_context.set_val(neuron_type)


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
        if new_val == self.val:
            return
        self.val = new_val
        self.cache_id = to_hashable(new_val)
        self.notify()


class NeuronTypeMixin:

    def check_for_new_neuron_type(self, context):
        if context.name == 'neuron_type_context':
            if self.last_neuron_type != context.val:
                self.last_neuron_type = context.val
                self.update_neuron_type()

