from copy import deepcopy
from context import data_type_context as dt_context, neuron_type_context as nt_context
from utils import get_ancestors


class Base:

    data_type_context = dt_context
    neuron_type_context = nt_context

    @property
    def data_opts(self):
        return (self.data_type_context.val if self.data_type_context is not None else None) or None

    @data_opts.setter
    def data_opts(self, opts):
        self.data_type_context.set_val(opts)

    @property
    def data_type(self):
        return self.data_opts['data_type']

    @property
    def data_class(self):
        return self.data_opts.get('data_class')

    @data_type.setter
    def data_type(self, data_type):
        data_opts = deepcopy(self.data_opts)
        data_opts['data_type'] = data_type
        self.data_opts = data_opts

    @property
    def selected_neuron_type(self):
        return self.neuron_type_context.val

    @selected_neuron_type.setter
    def selected_neuron_type(self, neuron_type):
        self.neuron_type_context.set_val(neuron_type)

    @property
    def neuron_types(self):
        return ['IN', 'PN']

    @property
    def current_frequency_band(self):
        return self.data_opts.get('frequency_band')

    @current_frequency_band.setter
    def current_frequency_band(self, frequency_band):
        self.data_opts['frequency_band'] = frequency_band


class Data(Base):

    def __iter__(self):
        for child in self.children:
            yield child

    @property
    def data(self):
        return getattr(self, f"get_{self.data_type}")()

    @property
    def ancestors(self):
        return get_ancestors(self)


