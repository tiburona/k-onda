from copy import deepcopy
import numpy as np
from signal_processing import compute_one_sided_spectrum, get_spectrum_fenceposts
from utils import cache_method


class Context:
    def __init__(self, opts):
        self.observers = []
        self.opts = opts

    def subscribe(self, observer):
        self.observers.append(observer)

    def set_opts(self, new_opts):
        self.opts = new_opts
        self.notify()

    def notify(self):
        for observer in self.observers:
            observer.update(self)


class SpikeRateMixin:

    @cache_method
    def get_hist(self, spikes, num_bins=None, spike_range=None):
        pre_stim, post_stim, bin_size = (self.context.opts.get(v) for v in ['pre_stim', 'post_stim', 'bin_size'])
        num_bins = num_bins if num_bins is not None else int((post_stim + pre_stim) / bin_size)
        spike_range = spike_range if spike_range is not None else (-pre_stim, post_stim)
        hist = np.histogram(spikes, bins=num_bins, range=spike_range)
        return hist

    @cache_method
    def get_rates(self, spikes, num_bins=None, spike_range=None):
        return self.get_hist(spikes, num_bins=num_bins, spike_range=spike_range)[0] / self.context.opts.get('bin_size')


class FamilyTreeMixin:

    @cache_method
    def get_data(self, opts, data_type, neuron_type=None):
        return getattr(self, f"get_{data_type}")(opts, neuron_type=neuron_type)

    @cache_method
    def get_average(self, opts, base_method, neuron_type=None):
        child_vals = []
        for child in self.children:
            average = child.get_average(opts, base_method, neuron_type)
            if average.size > 0:
                child_vals.append(average)
        return np.nanmean(child_vals, axis=0) if len(child_vals) else np.array([])

    @cache_method
    def get_psth(self, opts, neuron_type=None):
        return self.get_average(opts, 'get_pretone_corrected_trials', neuron_type=neuron_type)

    @cache_method
    def get_autocorr(self, opts, neuron_type=None):
        return self.get_all_autocorrelations(opts, neuron_type=neuron_type)[opts['ac_key']]

    @staticmethod
    def spectrum(series, opts):
        fft = np.fft.fft(series)
        oss = compute_one_sided_spectrum(fft)
        first, last = get_spectrum_fenceposts(opts)
        return oss[first:last]

    @cache_method
    def get_spectrum(self, opts, neuron_type=None):
        result = self.get_autocorr(opts, neuron_type=neuron_type)
        if not np.all(np.isnan(result)):
            return self.spectrum(result, opts)
        else:
            return np.array([])

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
        child_opts = deepcopy(opts)
        if opts['data_type'] in ['autocorr', 'spectrum']:
            key_index = len(self.name + '_by_')
            child_opts['ac_key'] = opts['ac_key'][key_index:]
        for child in self.children:
            child_vals = getattr(child, f"get_{opts['data_type']}")(child_opts, neuron_type=neuron_type)
            if np.any(~np.isnan(child_vals)):
                children_vals.append(child_vals)
        return self.sem(children_vals)

