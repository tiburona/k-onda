from context import experiment_context
from utils import get_ancestors, get_descendants
import numpy as np
from math_functions import sem
from utils import cache_method, is_empty


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

    @data_type.setter
    def data_type(self, data_type):
        self.update_data_opts(['data_type'], data_type)

    @property
    def data_class(self):
        return self.data_opts.get('data_class')

    @property
    def neuron_types(self):
        if hasattr(self, '_neuron_types'):
            return self._neuron_types
        elif hasattr(self, 'experiment'):
            return self.experiment.neuron_types
        elif hasattr(self, 'parent'):
            return self.parent.neuron_types
        else:
            return None

    @property
    def selected_neuron_type(self):
        return self.context.vals.get('neuron_type')

    @selected_neuron_type.setter
    def selected_neuron_type(self, neuron_type):
        self.context.set_val('neuron_type', neuron_type)

    @property
    def selected_block_type(self):
        return self.context.vals.get('block_type')

    @selected_block_type.setter
    def selected_block_type(self, block_type):
        self.context.set_val('block_type', block_type)

    @property
    def current_frequency_band(self):
        return self.data_opts.get('frequency_band')

    @current_frequency_band.setter
    def current_frequency_band(self, frequency_band):

        self.update_data_opts(['frequency_band'], frequency_band)

    @property
    def current_brain_region(self):
        return self.data_opts.get('brain_region')

    @current_brain_region.setter
    def current_brain_region(self, brain_region):
        self.update_data_opts(['brain_region'], brain_region)

    def update_data_opts(self, path, value):
        current_level = self.data_opts
        for key in path[:-1]:
            if key not in current_level or not isinstance(current_level[key], dict):
                current_level[key] = {}
            current_level = current_level[key]
        current_level[path[-1]] = value

        self.data_opts = self.data_opts  # Reassign to trigger the setter


class Evoked:
    def __init__(self, method):
        self.method = method

    def __call__(self, *args, **kwargs):
        self_instance = args[0]
        if self_instance.data_opts.get('evoked') == 'individual_reference':
            if 'stop_at' not in kwargs or kwargs['stop_at'] == self_instance.name:
                ref_data = self_instance.reference.data
                mean = np.mean(ref_data) if ref_data.shape[0] == 1 else np.mean(ref_data, axis=1)[:, np.newaxis]
                result = self.method(*args, **kwargs)
                return result - mean

        return self.method(*args, **kwargs)


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
        return getattr(self, f"get_{self.data_type}")()

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
    def sampling_rate(self):
        if hasattr(self, '_sampling_rate'):
            return self._sampling_rate
        elif self.parent is not None:
            return self.parent.sampling_rate
        else:
            return None

    @property
    def is_relative(self):
        if hasattr(self, '_is_relative'):
            return self._is_relative
        elif self.parent is not None:
            return self.parent.is_relative
        else:
            return None

    @property
    def reference(self):
        if hasattr(self, 'block') and not self.block.reference_block_type:
            return None
        elif hasattr(self, 'reference_block_type') and not self.reference_block_type:
            return None
        else:
            if self.name == 'block':
                return [blk for blk in self.parent.blocks[self.reference_block_type] if self is blk.target_block][0]
            if self.name == 'mrl_calculator':
                return [calc for calc in self.parent.mrl_calculators[self.reference_block_type]
                        if self is calc.block.target and self.unit is calc.unit][0]
            if self.name == 'event':
                return self.parent.reference.events[self.identifier]
        return None

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
    def current_reference_block_type(self):
        exp = self.find_experiment()
        return [blk for blk in exp.all_blocks if blk.block_type == self.selected_block_type][0].reference_block_type

    def refer(self, data, stop_at='', is_spectrum=False):
        if (  # all the conditions in which reference data should not be subtracted
                not self.data_opts.get('evoked') or
                self.block_type == self.current_reference_block_type or
                (stop_at and stop_at != self.name) or
                (self.data_type == 'spectrum' and not is_spectrum)
        ):
            return data
        if self.reference.name == 'block':
            ref_data = self.reference.data
        else:
            ref_data = self.reference.block.data
        mean_ref_data = np.mean(ref_data) if ref_data.shape[0] == 1 else np.mean(ref_data, axis=1)[:, np.newaxis]
        data -= mean_ref_data
        return data

    def get_average(self, base_method, stop_at='event', axis=0, **kwargs):
        """
        Recursively calculates the average of the values of the computation in the base method on the object's
        descendants.

        Parameters:
        - base_method (str): Name of the method to be called when the recursion reaches the base case.
        - stop_at (str): The 'name' attribute of the base case object upon which `base_method` should be called.
        - axis (int or None): Specifies the axis across which to compute the mean.
        - **kwargs: Additional keyword arguments to be passed to the base method.

        Returns:
        float or np.array: The mean of the data values from the object's descendants.
        """
        if self.name == stop_at:  # we are at the base case and will call the base method
            if hasattr(self, base_method) and callable(getattr(self, base_method)):
                return getattr(self, base_method)(**kwargs)
            else:
                raise ValueError(f"Invalid base method: {base_method}")

        else:  # recursively call
            child_vals = [child.get_average(base_method, axis=axis, stop_at=stop_at, **kwargs) for child in
                          self.children]
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
            return sem([child.mean_data for child in sem_children])
        else:
            return sem([child.data for child in sem_children])

    @cache_method
    def get_scatter_points(self):
        """Returns a list of points of the data values for an object's children for use on, e.g. a bar graph"""
        if not self.children:
            return []
        return [np.nanmean(child.data) for child in self.children]

    def find_experiment(self):
        if self.name == 'experiment':
            return self
        elif hasattr(self, 'experiment'):
            return self.experiment
        elif hasattr(self, 'parent'):
            return self.parent.find_experiment()
        else:
            return None


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


