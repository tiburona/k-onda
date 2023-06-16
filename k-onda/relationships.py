from copy import deepcopy
import numpy as np
from signal_processing import compute_one_sided_spectrum, get_spectrum_fenceposts
from contexts import cache_method


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

    def subscribe(self, context):
        setattr(self, 'context', context)
        context.subscribe(self)

    @cache_method
    def get_data(self, data_type):
        return getattr(self, f"get_{data_type}")()

    @cache_method
    def get_average(self, base_method):
        child_vals = []
        for child in self.children:
            average = child.get_average(base_method)
            if average.size > 0:
                child_vals.append(average)
        return np.nanmean(child_vals, axis=0) if len(child_vals) else np.array([])

    @cache_method
    def get_demeaned_rates(self):
        rates = self.get_average('get_rates')
        return rates - np.mean(rates)

    @cache_method
    def get_psth(self):
        return self.get_average('get_psth')

    @cache_method
    def get_autocorr(self):
        return self.get_all_autocorrelations()

    def spectrum(self, series):
        fft = np.fft.fft(series)
        oss = compute_one_sided_spectrum(fft)
        first, last = get_spectrum_fenceposts(self.context.opts)
        return oss[first:last]

    @cache_method
    def get_spectrum(self):
        result = self.get_autocorr()
        return self.spectrum(result)

    @staticmethod
    def sem(children_vals):
        all_series = np.vstack(children_vals)
        # Compute standard deviation along the vertical axis (each point in time)
        std_dev = np.std(all_series, axis=0, ddof=1)  # ddof=1 to compute sample standard deviation
        # Compute SEM = std_dev / sqrt(N)
        sem = std_dev / np.sqrt(len(all_series))
        return sem

    @cache_method
    def get_sem(self):
        children_vals = []
        child_opts = deepcopy(self.context.opts)
        if child_opts['data_type'] in ['autocorr', 'spectrum']:
            key_index = len(self.name + '_by_')
            child_opts['ac_key'] = self.context.opts['ac_key'][key_index:]
        for child in self.children:
            child_vals = getattr(child, f"get_{self.context.opts['data_type']}")()
            if np.any(~np.isnan(child_vals)):
                children_vals.append(child_vals)
        return self.sem(children_vals)


