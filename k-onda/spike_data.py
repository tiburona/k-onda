from bisect import bisect_left as bs_left, bisect_right as bs_right
from collections import defaultdict as dd
from copy import deepcopy

from relationships import FamilyTreeMixin, SpikeRateMixin
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
        for nt in ['PN', 'IN']:
            setattr(self, nt, [unit for unit in self.units['good'] if unit.neuron_type == nt])
        self.tone_onsets_expanded = tone_onsets_expanded if tone_onsets_expanded is not None else []
        self.tone_period_onsets = tone_period_onsets if tone_period_onsets is not None else []

    def plot_units(self, opts):
        Plotter(opts).plot_units(self)


class Unit(FamilyTreeMixin, AutocorrelationNode, SpikeRateMixin):
    name = 'unit'

    def __init__(self, context, animal, category, spike_times, neuron_type=None):
        self.context = context
        context.subscribe(self)
        self.trials = []
        self.animal = animal
        self.parent = self.animal
        self.category = category
        self.animal.units[category].append(self)
        self.identifier = str(self.animal.units[category].index(self) + 1)
        self.trials = None
        self.children = self.trials
        self.spike_times = spike_times
        self.neuron_type = neuron_type

    def update(self, context):
        self.trials = []
        trials_slice = slice(*context.opts.get('trials'))
        pre_stim = context.opts.get('pre_stim')
        post_stim = context.opts.get('post_stim')
        for start in self.animal.tone_onsets_expanded[trials_slice]:
            spikes = self.find_spikes(start - pre_stim * 30000, start + post_stim * 30000)
            self.trials.append([(spike-start)/30000 for spike in spikes])
        self.children = self.trials

    @cache_method
    def find_spikes(self, start, stop):
        return self.spike_times[bs_left(self.spike_times, start): bs_right(self.spike_times, stop)]


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
    def get_psth(self, opts, neuron_type=None):
        if neuron_type is not None and self.neuron_type != neuron_type:
            return np.array([])
        return np.mean(np.array(self.get_pretone_corrected_trials(opts)), axis=0).flatten()

    @cache_method
    def get_pretone_corrected_trials(self, opts):
        pretone_means = self.get_pretone_means(opts)
        trial_indices = list(range(150))[slice(*opts['trials'])]  # select only the trials indicated in opts
        trials_rates = self.get_trials_rates(opts)
        return [trials_rates[i] - pretone_means[trial_index // 30] for i, trial_index in enumerate(trial_indices)]

    def get_average(self, opts, base_method, neuron_type=None):
        if neuron_type is not None and self.neuron_type != neuron_type:
            return np.array([])
        return np.nanmean(getattr(self, base_method)(opts), axis=0)


class Trial(FamilyTreeMixin, SpikeRateMixin, AutocorrelationNode):

    name = 'trial'

    def __init__(self, unit):
        self.parent = unit
        self.context = self.parent.context
        self.demeaned = []
        self.id = None



    def get_psth(self):


    def get_all_autocorrelations(self, opts):
        return {'trials': self.get_autocorrelation(opts)}

    def get_autocorr(self, opts):
        return self._calculate_autocorrelation(self.demeaned, opts)




