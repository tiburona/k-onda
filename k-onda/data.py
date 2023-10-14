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
        data_opts = deepcopy(self.data_opts)  # necessary to trigger the data type context notification
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
        """

        Returns:
        float or np.array: The mean of the data values from the object's descendants.
        """
        data = getattr(self, f"get_{self.data_type}")()
        if hasattr(self, 'evoked_value_calculator'):
            return self.evoked_value_calculator.get_evoked_data(data, self.data_opts.get('evoked'))
        else:
            return data

    @property
    def mean_data(self):
        return np.mean(self.data)

    @property
    def sd_data(self):
        return np.std(self.data)

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
        """
        Recursively calculates the average of the values of the computation in the base method on the object's
        descendants.

        Parameters:
        - base_method (str): Name of the method to be called when the recursion reaches the base case.
        - stop_at (str): The 'name' attribute of the base case object upon which `base_method` should be called.
        - axis (int or None): When the 'data' property of an object returns a vector or matrix instead of a single
        number, this specifies the axis across which to compute the mean. Default is 0, which preserves the original
        shape of the data and averages over dimensions like units and animals. If set to None, the mean is computed
        over all dimensions.

        Returns:
        float or np.array: The mean of the data values from the object's descendants.
        """
        if self.name == stop_at:  # we are at the base case and will call the base method
            return getattr(self, base_method)()
        else:  # recursively call
            child_vals = [child.get_average(base_method, stop_at=stop_at) for child in self.children]
            # Filter out nan values and arrays that are all NaN
            child_vals_filtered = [x for x in child_vals
                                   if not (isinstance(x, np.ndarray) and np.isnan(x).all())
                                   and not (isinstance(x, float) and np.isnan(x))]
            if axis is None:  # compute mean over all dimensions
                return np.nanmean(np.array(child_vals_filtered))
            else:  # compute mean over provided dimension
                return np.nanmean(np.array(child_vals_filtered), axis=axis)

    @cache_method
    def get_sem(self):
        """
        Calculates the standard error of an object's data. If object's data is a vector, it will always return a float.
        If object's data is a matrix, the `collapse_sem_data` opt will determine whether it returns the standard error
        of its children's average data points or whether it computes the standard error over children maintaining the
        original shape of children's data, as you would want, for instance, if graphing a standard error envelope around
        firing rate over time.

        Returns:
        float or np.array: The ."""
        if self.data_opts.get('collapse_sem_data'):
            return sem(self.scatter)
        else:
            return sem([child.data for child in self.children])

    @cache_method
    def get_scatter_points(self):
        """Returns a list of points of the data values for an object's children for use on, e.g. a bar graph"""
        return [np.mean(child.data) for child in self.children]


class EvokedValueCalculator:

    def __init__(self, ref):
        self.ref = ref

    @property
    def equivalent(self):
        other_stage = 'tone' if self.ref.period_type == 'pretone' else 'pretone'
        other_stage_children = getattr(self.ref.parent, f"all_{self.ref.name}s")[other_stage]
        return [p for p in other_stage_children if p.identifier == self.ref.identifier][0]

    def get_evoked_data(self, data, evoked):
        if evoked and self.ref.period_type == 'tone':
            data -= self.equivalent.data
        return data





