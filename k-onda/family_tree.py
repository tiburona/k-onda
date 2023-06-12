from copy import deepcopy
import numpy as np
from signal_processing import compute_one_sided_spectrum, get_spectrum_fenceposts
from utils import cache_method


class FamilyTreeMixin:

    def get_data(self, opts, data_type, neuron_type=None):
        if data_type in ['autocorr', 'spectrum']:
            return getattr(self, f"get_{data_type}")(opts, neuron_type=neuron_type)
        else:  # data_type is psth
            return self.get_average(opts, 'get_psth', neuron_type=neuron_type)

    def get_autocorr(self, opts, neuron_type=None):
        try:
            return self.get_all_autocorrelations(opts, neuron_type=neuron_type)[opts['ac_key']]
        except KeyError as e:
            if str(e) == "'rates'":
                print("You probably tried to print a graph with an SEM where the unit autocorrelation was calculated "
                      "over rates. You can't do that, because there's only one value per unit in that calculation.")
        raise

    def get_spectrum(self, opts, neuron_type=None):
        result = self.get_autocorr(opts, neuron_type=neuron_type)
        if not np.all(np.isnan(result)):
            fft = np.fft.fft(self.get_autocorr(opts, neuron_type=neuron_type))
            oss = compute_one_sided_spectrum(fft)
            first, last = get_spectrum_fenceposts(opts)
            return oss[first:last]
        else:
            return np.array([])

    @cache_method
    def get_average(self, opts, base_method, neuron_type=None):
        child_vals = []
        for child in self.children:
            average = child.get_average(opts, base_method, neuron_type)
            if average.size > 0:
                child_vals.append(average)
        return np.nanmean(child_vals, axis=0) if len(child_vals) else np.array([])

    @staticmethod
    def sem(children_vals):
        all_series = np.vstack(children_vals)
        # Compute standard deviation along the vertical axis (each point in time)
        std_dev = np.std(all_series, axis=0, ddof=1)  # ddof=1 to compute sample standard deviation
        # Compute SEM = std_dev / sqrt(N)
        sem = std_dev / np.sqrt(len(all_series))
        return sem

    @cache_method
    def get_sem(self, opts, neuron_type=None):
        children_vals = []
        for child in self.children:
            data_func = getattr(child, f"get_{opts['data_type']}")
            child_opts = deepcopy(opts)
            if opts['data_type'] in ['autocorr', 'spectrum']:
                key_index = len(self.name + '_by_')
                child_opts['ac_key'] = opts['ac_key'][key_index:]
            child_vals = data_func(child_opts, neuron_type=neuron_type)
            if np.any(~np.isnan(child_vals)):
                children_vals.append(data_func(child_opts, neuron_type=neuron_type))

        return self.sem(children_vals)

