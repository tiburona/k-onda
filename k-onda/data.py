from copy import deepcopy
from context import experiment_context
from utils import get_ancestors, get_descendants
import numpy as np
from math_functions import sem
from utils import cache_method

# NEURON_TYPES = ['IN', 'PN']
NEURON_TYPES = ['PV_IN', 'ACH']
PERIOD_TYPES = ['pretone', 'tone']


class Base:

    context = experiment_context

    @property
    def data_opts(self):
        return self.context.vals.get('data')

    @data_opts.setter
    def data_opts(self, opts):
        self.context.set_val('data', opts)

    @property
    def data_type(self):
        return self.data_opts['data_type']

    @property
    def data_class(self):
        return self.data_opts.get('data_class')

    @data_type.setter
    def data_type(self, data_type):
        data_opts = deepcopy(self.data_opts)  # necessary to trigger the data context notification TODO: check that this is still necessay
        data_opts['data_type'] = data_type
        self.data_opts = data_opts

    @property
    def selected_neuron_type(self):
        return self.context.vals.get('neuron_type')

    @selected_neuron_type.setter
    def selected_neuron_type(self, neuron_type):
        self.context.set_val('neuron_type', neuron_type)

    @property  # TODO: reading from global vars here is terrible; undo t
    def neuron_types(self):
        return NEURON_TYPES

    @property
    def period_types(self):
        return PERIOD_TYPES

    @property
    def selected_block_type(self):
        return self.context.vals.get('neuron_type')

    @selected_block_type.setter
    def selected_block_type(self, block_type):
        self.context.set_val('block_type', block_type)

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
    def descendants(self):
        return get_descendants(self)

    @property
    def sem(self):
        return self.get_sem()

    @property
    def scatter(self):
        return self.get_scatter_points()

    @property
    def num_bins_per_event(self):
        pre_stim, post_stim, bin_size = (self.data_opts.get(opt) for opt in ['pre_stim', 'post_stim', 'bin_size'])
        return int((pre_stim + post_stim) / bin_size)

    @property
    def sampling_rate(self):
        if hasattr(self, '_sampling_rate'):
            return self._sampling_rate
        elif self.parent is not None:
            return self.parent.sampling_rate
        else:
            return None

    @property
    def event_duration(self):
        if self._event_duration is not None:
            return self._event_duration
        elif self.parent is not None:
            return self.parent.event_duration
        else:
            return None

    @cache_method
    def get_average(self, base_method, stop_at='event', axis=0):  # Trial is the default base case, but not always
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
        """

        if self.data_opts.get('sem_level'):
            sem_children = get_descendants(self, level=self.data_opts.get('sem_level'))
        else:
            sem_children = self.children

        if self.data_opts.get('collapse_sem_data'):
            return sem([np.mean(child.data) for child in sem_children])
        else:
            return sem([child.data for child in sem_children])

    @cache_method
    def get_scatter_points(self):
        """Returns a list of points of the data values for an object's children for use on, e.g. a bar graph"""
        if not self.children:
            return []
        return [np.nanmean(child.data) for child in self.children]



class TimeBin:
    name = 'time_bin'

    def __init__(self, i, val, parent):
        self.parent = parent
        self.identifier = i
        self.data = val
        self.mean_data = val
        self.ancestors = get_ancestors(self)

    def position_in_block_time_series(self):
        return self.parent.num_bins * self.parent.identifier + self.identifier





