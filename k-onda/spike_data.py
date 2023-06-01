from bisect import bisect_left as bs_left, bisect_right as bs_right
from collections import defaultdict as dd

import numpy as np
import pandas as pd

from plotters import *
from utils import cache_method
from signal_processing import compute_one_sided_spectrum


class FamilyTree:

    @cache_method
    def get_average(self, opts, base_method, neuron_type=None):
        if isinstance(self, Unit):
            child_vals = getattr(self, base_method)(opts) if neuron_type is None or self.neuron_type == neuron_type else np.array([])
        else:
            child_vals = []
            for child in self.children:
                average = child.get_average(opts, base_method, neuron_type)
                if average.size > 0:
                    child_vals.append(average)
        return np.nanmean(child_vals, axis=0) if len(child_vals) else np.array([])

    def plot_children(self, opts):
        pass


class AutocorrelationCalculator:
    def __init__(self):
        pass

    @cache_method
    def _autocorr_np(self, x, max_lag):
        result = np.correlate(x, x, mode='full')
        mid = result.size // 2
        return result[mid + 1:mid + max_lag + 1] / result[mid]

    @cache_method
    def _autocorr_pd(self, x, max_lag):
        return [pd.Series(x).autocorr(lag=lag) for lag in range(max_lag + 1)]

    def _calculate_autocorrelation(self, rates, opts, method):
        if not len(rates):
            return np.array([])
        return getattr(self, f"_autocorr_{method}")(rates, opts['max_lag'])

    def calculate_all_autocorrelations(self, opts, method=None, neuron_type=None):
        result = {}
        if isinstance(self, Unit):
            result['self_over_children'] = np.mean([self._calculate_autocorrelation(rates, opts, method)
                                                    for rates in self.get_trials_rates(opts)], axis=0)
        else:
            child_autocorr = {child.name: child.calculate_all_autocorrelations(opts, method, neuron_type) for child in
                              self.children}
            for key, value in child_autocorr.items():
                result[key + '_over_children'] = value['self_over_children']
                result[key + '_over_rates'] = value['self_over_rates']
            child_vals = [self._calculate_autocorrelation(
                child.get_average(opts, base_method='get_trials_rates', neuron_type=neuron_type), opts, method)
                for child in self.children]
            result['self_over_children'] = np.mean([arr for arr in child_vals if arr.size > 0], axis=0)
        result['self_over_rates'] = self._calculate_autocorrelation(self.get_average(
            opts, base_method='get_trials_rates', neuron_type=neuron_type), opts, method)
        return result


class Experiment(FamilyTree, AutocorrelationCalculator):
    def __init__(self, conditions=None):
        if conditions is not None:
            self.conditions, self.groups = conditions.keys(), conditions.values()
        self.children = self.groups
        self.children_name = 'groups'

    def plot_groups(self, opts, data_type=None):
        Plotter(opts, data_type=data_type).plot_groups(self.groups, ['PN', 'IN'])


class Group(FamilyTree, AutocorrelationCalculator):
    def __init__(self, name, animals=None):
        self.name = name
        self.animals = animals if animals else []
        self.children_name = 'animals'
        self.children = self.animals
        
    def plot_animals(self, opts, data_type=None):
        [Plotter(opts, data_type=data_type).plot_animals(self, neuron_type=neuron_type) for neuron_type in ['PN', 'IN']]


class Animal(FamilyTree, AutocorrelationCalculator):
    def __init__(self, name, condition, tone_period_onsets=None, tone_onsets_expanded=None, units=None):
        self.name = name
        self.condition = condition
        self.units = units if units is not None else dd(list)
        self.children = self.units['good']
        self.children_name = 'units'
        self.tone_onsets_expanded = tone_onsets_expanded if tone_onsets_expanded is not None else []
        self.tone_period_onsets = tone_period_onsets if tone_period_onsets is not None else []

    @cache_method
    def get_spectrum(self, autocorr_key, opts):
        autocorr = self.calculate_all_autocorrelations[autocorr_key]
        fft = np.fft.fft(autocorr)
        return SignalProcessing.compute_one_sided_spectrum(fft, opts['lags'])

    def plot_units(self, opts):
        Plotter(opts).plot_units(self)


class Unit(FamilyTree, AutocorrelationCalculator):
    def __init__(self, animal, category, spike_times, neuron_type=None):
        self.animal = animal
        self.category = category
        self.spike_times = spike_times
        self.animal.units[category].append(self)
        self.index = self.animal.units[category].index(self)
        self.name = str(self.index + 1)
        self.neuron_type = neuron_type
        self.children = None

    @cache_method
    def find_spikes(self, start, stop):
        return self.spike_times[bs_left(self.spike_times, start): bs_right(self.spike_times, stop)]
    
    @cache_method
    def get_trials_spikes(self, opts):
        return [[(spike - start)/30000 for spike in
                 self.find_spikes(start - opts['pre_stim'] * 30000, start + opts['post_stim'] * 30000)]
                for start in self.animal.tone_onsets_expanded[slice(*opts['trials'])]]

    @cache_method
    def get_hist(self, spikes, opts, num_bins=None, spike_range=None):
        num_bins = num_bins if num_bins is not None else int((opts['post_stim'] + opts['pre_stim']) / opts['bin_size'])
        spike_range = spike_range if spike_range is not None else (-opts['pre_stim'], opts['post_stim'])
        hist = np.histogram(spikes, bins=num_bins, range=spike_range)
        return hist

    @cache_method
    def get_rates(self, spikes, opts, num_bins=None, spike_range=None):
        return self.get_hist(spikes, opts, num_bins=num_bins, spike_range=spike_range)[0] / opts['bin_size']

    @cache_method
    def get_trials_rates(self, opts, num_bins=None, spike_range=None):
        return [self.get_rates(trial_spikes, opts, num_bins=num_bins, spike_range=spike_range)
                for trial_spikes in self.get_trials_spikes(opts)]

    @cache_method
    def get_pretone_means(self, opts):
        rate_set = []
        for onset in self.animal.tone_period_onsets:
            start = onset - 30 * 30000
            stop = onset - 1
            rate_set.append(self.get_rates(self.find_spikes(start, stop), opts, spike_range=(start, stop),
                                           num_bins=int(30/opts['bin_size'])))
        return np.mean(np.array(rate_set), axis=1)

    @cache_method
    def get_psth(self, opts):
        pretone_means = self.get_pretone_means(opts)
        trial_indices = list(range(150))[slice(*opts['trials'])]  # select only the trials indicated in opts
        trials_rates = self.get_trials_rates(opts)
        rate_set = [trials_rates[i] - pretone_means[trial_index // 30] for i, trial_index in enumerate(trial_indices)]
        return np.mean(np.array(rate_set), axis=0).flatten()

    @cache_method
    def get_fft(self, opts):
        return np.fft.fft(self.get_autocorr(opts))

    @cache_method
    def get_spectrum(self, opts):
        return compute_one_sided_spectrum(self.get_fft(opts))
