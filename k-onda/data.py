from copy import deepcopy
from context import data_type_context as dt_context, neuron_type_context as nt_context, \
    period_type_context as pt_context
from utils import get_ancestors
import numpy as np
from math_functions import sem
from utils import cache_method


class Base:

    data_type_context = dt_context
    neuron_type_context = nt_context
    period_type_context = pt_context

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
    def selected_period_type(self):
        return self.period_type_context.val

    @selected_period_type.setter
    def selected_period_type(self, period_type):
        self.period_type_context.set_val(period_type)

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

    @property
    def sem(self):
        return self.get_sem()

    @property
    def scatter(self):
        return self.get_scatter_points()

    @cache_method
    def get_average(self, base_method, stop_at='trial', axis=0):  # Trial is the default base case, but not always
        if self.name == stop_at:
            return getattr(self, base_method)()
        else:
            child_vals = [child.get_average(base_method, stop_at=stop_at) for child in self.children]
            # Filter out nan values and arrays that are all NaN
            child_vals_filtered = [x for x in child_vals if
                                   not (isinstance(x, np.ndarray) and np.isnan(x).all()) and not (
                                               isinstance(x, float) and np.isnan(x))]
            if axis is None:
                return np.nanmean(np.array(child_vals_filtered))
            else:
                return np.nanmean(np.array(child_vals_filtered), axis=axis)

    @cache_method
    def get_sem(self):
        return sem(self.scatter)

    @cache_method
    def get_scatter_points(self):
        return [child.data for child in self.children]



