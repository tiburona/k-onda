from bisect import bisect_left as bs_left, bisect_right as bs_right
from collections import defaultdict as dd

from family_tree import FamilyTreeMixin
from plotters import *
from autocorrelation import AutocorrelationNode
from utils import cache_method


class Experiment(FamilyTreeMixin, AutocorrelationNode):
    name = 'experiment'

    def __init__(self, conditions=None):
        if conditions is not None:
            self.conditions, self.groups = conditions.keys(), conditions.values()
        self.children = self.groups
        self.children_name = 'groups'
        for group in self.groups:
            group.parent = self

    def plot_groups(self, opts, data_type=None):
        Plotter(self, opts, data_type=data_type).plot_groups()


class Group(FamilyTreeMixin, AutocorrelationNode):
    name = 'group'

    def __init__(self, name, animals=None):
        self.identifier = name
        self.animals = animals if animals else []
        self.children_name = 'animals'
        self.children = self.animals
        for child in self.children:
            child.parent = self
        self.parent = None
        
    def plot_animals(self, opts, data_type=None):
        [Plotter(opts, data_type=data_type).plot_animals(self, neuron_type=neuron_type) for neuron_type in ['PN', 'IN']]


class Animal(FamilyTreeMixin, AutocorrelationNode):
    name = 'animal'

    def __init__(self, name, condition, tone_period_onsets=None, tone_onsets_expanded=None, units=None):
        self.identifier = name
        self.condition = condition
        self.units = units if units is not None else dd(list)
        self.children = self.units['good']
        self.children_name = 'units'
        self.parent = None
        self.tone_onsets_expanded = tone_onsets_expanded if tone_onsets_expanded is not None else []
        self.tone_period_onsets = tone_period_onsets if tone_period_onsets is not None else []

    def plot_units(self, opts):
        Plotter(opts).plot_units(self)


class Unit(FamilyTreeMixin, AutocorrelationNode):
    name = 'unit'

    def __init__(self, animal, category, spike_times, neuron_type=None):
        self.animal = animal
        self.category = category
        self.spike_times = spike_times
        self.animal.units[category].append(self)
        # self.index = self.animal.units[category].index(self)
        self.parent = self.animal
        self.identifier = str(self.animal.units[category].index(self) + 1)
        self.neuron_type = neuron_type
        self.children = []

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

    def get_average(self, opts, base_method, neuron_type=None):
        if neuron_type is None or self.neuron_type == neuron_type:
            child_vals = getattr(self, base_method)(opts)
        else:
            child_vals = np.array([])
        return np.nanmean(child_vals, axis=0) if len(child_vals) else np.array([])

    def get_all_autocorrelations(self, opts, method, neuron_type=None, demean=False):
        if neuron_type is not None and self.neuron_type != neuron_type:
            return {f'{self.name}_by_rates': float('nan'), f'{self.name}_by_trials': float('nan')}
        else:
            results = {}
            rates = self.get_average(opts, 'get_trials_rates', neuron_type=neuron_type)
            demeaned_rates = rates - (np.mean(rates))
            results[f'{self.name}_by_rates'] = self._calculate_autocorrelation(demeaned_rates, opts, method,
                                                                               demean=demean)
            demeaned_rates = [rates - np.mean(rates) for rates in self.get_trials_rates(opts)]
            results[f'{self.name}_by_trials'] = np.nanmean([
                self._calculate_autocorrelation(rates, opts, method, demean=demean)
                for rates in demeaned_rates], axis=0)
        return results


