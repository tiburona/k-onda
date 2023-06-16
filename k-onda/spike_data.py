from bisect import bisect_left as bs_left, bisect_right as bs_right
from collections import defaultdict as dd

from relationships import FamilyTreeMixin, SpikeRateMixin
from plotters import *
from autocorrelation import AutocorrelationNode
from contexts import cache_method


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
        self.context = None

    def update(self, context):
        self.children = [unit for unit in self.units['good'] if unit.neuron_type == context.neuron_type]

    def plot_units(self, opts):
        Plotter(opts).plot_units(self)


class Unit(FamilyTreeMixin, AutocorrelationNode, SpikeRateMixin):
    name = 'unit'

    def __init__(self, animal, category, spike_times, neuron_type=None):
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
        self.context = None

    def update(self, context):
        self.trials = []
        trials_slice = slice(*context.opts.get('trials'))
        pre_stim = context.opts.get('pre_stim')
        post_stim = context.opts.get('post_stim')
        for i, start in enumerate(self.animal.tone_onsets_expanded[trials_slice]):
            spikes = self.find_spikes(start - pre_stim * 30000, start + post_stim * 30000)
            self.trials.append(Trial(self, [(spike-start)/30000 for spike in spikes], i))
        self.children = self.trials

    @cache_method
    def find_spikes(self, start, stop):
        return self.spike_times[bs_left(self.spike_times, start): bs_right(self.spike_times, stop)]

    @cache_method
    def get_pretone_means(self):
        rate_set = []
        for onset in self.animal.tone_period_onsets:
            start = onset - 30 * 30000
            stop = onset - 1
            rate_set.append(self.get_rates(self.find_spikes(start, stop), spike_range=(start, stop),
                                           num_bins=int(30/opts['bin_size'])))
        return np.mean(np.array(rate_set), axis=1)


class Trial(FamilyTreeMixin, SpikeRateMixin, AutocorrelationNode):
    name = 'trial'

    def __init__(self, unit, spikes, index):
        self.unit = unit
        self.parent = self.unit
        self.context = self.unit.context
        self.spikes = spikes
        self.index = index
        self.identifier = self.context.selected_trial_indices[self.index]

    @cache_method
    def get_average(self, base_method):
        return getattr(self, base_method)()

    @cache_method
    def get_psth(self):
        # self.identifier // 30 can have one of five values; it's the index of the tone period
        return self.get_rates() - self.unit.get_pretone_means()[self.identifier // 30]

    def get_all_autocorrelations(self):
        return {'trials': self.get_autocorrelation()}

    def get_autocorr(self, opts):
        return self._calculate_autocorrelation(self.get_demeaned_rates())




