from bisect import bisect_left as bs_left, bisect_right as bs_right
from collections import defaultdict
import pandas as pd
import numpy as np

from context import Base
from matlab_interface import xcorr
from utils import cache_method
from math_functions import calc_rates, spectrum, sem, trim_and_normalize_ac


"""
This module defines Experiment, Group, Animal, Unit, and Trial. Experiment inherits from Base, which defines a few 
common properties. Group, Unit, and Trial inherit from Experiment, but Animal shares a unique method with Group, and 
inherits from it. Several of Experiment's methods are recursive, and are overwritten by the base case, most frequently 
Trial.  
"""

SAMPLES_PER_SECOND = 30000
TONES_PER_PERIOD = 30  # The pip sounds 30 times in a tone period
TONE_PERIOD_DURATION = 30
INTER_TONE_INTERVAL = 30  # number of seconds between tones
PIP_DURATION = .05


class Level(Base):
    @property
    def data(self):
        return getattr(self, f"get_{self.data_type}")()

    @property
    def autocorr_key(self):
        return self.get_autocorr_key()

    def subscribe(self, context):
        setattr(self, context.name, context)
        context.subscribe(self)

    @cache_method
    def get_average(self, base_method, stop_at='trial'):  # Trial is the default base case, but not always
        if self.name == stop_at:
            return getattr(self, base_method)()
        else:
            child_vals = [child.get_average(base_method, stop_at=stop_at) for child in self.children]
        return np.mean(child_vals, axis=0)

    @cache_method
    def get_demeaned_rates(self):
        rates = self.get_average('get_rates')
        return rates - np.mean(rates)

    @cache_method
    def get_psth(self):
        return self.get_average('get_psth')

    @cache_method
    def proportion_score(self):
        return [1 if rate > 0 else 0 for rate in self.get_psth()]

    @cache_method
    def get_proportion_score(self):
        return self.get_average('proportion_score', stop_at=self.data_opts.get('base'))

    @cache_method
    def get_autocorr(self):
        return self.get_all_autocorrelations()[self.autocorr_key]

    @cache_method
    def get_spectrum(self):
        freq_range, max_lag, bin_size = (self.data_opts[opt] for opt in ['freq_range', 'max_lag', 'bin_size'])
        return spectrum(self.get_autocorr(), freq_range, max_lag, bin_size)

    @cache_method
    def get_sem(self):
        return sem([child.data for child in self.children])

    def get_autocorr_key(self):
        key = self.data_opts.get('ac_key')
        if key is None:
            return key
        else:
            # if self is the level being plotted, this will return the key in opts, or else it will return the
            # appropriate key for the child, the latter portion of the parent's key
            return key[key.find(self.name):]

    def _calculate_autocorrelation(self, x):
        opts = self.data_opts
        max_lag = opts['max_lag']
        if not len(x):
            return np.array([])
        if opts['ac_program'] == 'np':
            return trim_and_normalize_ac(np.correlate(x, x, mode='full'), max_lag)
        elif opts['ac_program'] == 'ml':
            return trim_and_normalize_ac(xcorr(x, max_lag), max_lag)
        elif opts['ac_program'] == 'pd':
            return np.array([pd.Series(x).autocorr(lag=lag) for lag in range(max_lag + 1)])[1:]
        else:
            raise "unknown autocorr type"

    @cache_method
    def get_all_autocorrelations(self):

        # Calculate the autocorrelation of the rates for this node
        ac_results = {f"{self.name}_by_rates": self._calculate_autocorrelation(self.get_demeaned_rates())}

        # Calculate the autocorrelation by children for this node, i.e. the average of the children's autocorrelations
        # We need to ask each child to calculate its autocorrelations first.
        children_autocorrs = [child.get_all_autocorrelations() for child in self.children]

        for key in children_autocorrs[0]:  # Assuming all children have the same autocorrelation keys
            ac_results[f"{self.name}_by_{key}"] = np.mean(
                [child_autocorrs[key] for child_autocorrs in children_autocorrs], axis=0)
        return ac_results


class Experiment(Level):
    """The experiment. Parent of groups."""

    name = 'experiment'

    def __init__(self, conditions):
        self.conditions, self.groups = conditions.keys(), conditions.values()
        self.children = self.groups
        for group in self.groups:
            group.parent = self
        self.all_units = None


class Group(Level):
    """A group in the experiment, i.e., a collection of animals assigned to a condition, the child of an Experiment,
    parent of Animals. Subscribes to neuron_type_context that defines which neuron type, PN or IN, is currently active.
    Limits its children to animals who have neurons of that type.  Also subscribes to data_type_context but doesn't need
    to update its children property when it changes."""

    name = 'group'

    def __init__(self, name, animals=None):
        self.identifier = name
        self.animals = animals if animals else []
        self.children = self.animals
        self.data_type_context = None
        self.neuron_type_context = None
        self.last_neuron_type = None

    def update(self, context):
        self.check_for_new_neuron_type(context)

    def check_for_new_neuron_type(self, context):
        if context.name == 'neuron_type_context':
            if self.last_neuron_type != context.val:
                self.last_neuron_type = context.val
                self.update_neuron_type()

    def update_neuron_type(self):
        if self.selected_neuron_type is None:
            self.children = self.animals
        else:
            self.children = [animal for animal in self.animals if len(getattr(animal, self.selected_neuron_type))]


class Animal(Group):
    """An animal in the experiment, the child of a Group, parent of Units. Subscribes to neuron_type_context that
    defines which neuron type, PN or IN, is currently active. Updates its children, i.e., the active units for analysis,
    when the context changes. Also subscribes to the data_type_context but doesn't need to update its children property
    when it changes.

    Note that the `units` property is not a list, but rather a dictionary with keys for different categories in list.
    It would also be possible to implement context changes where self.children updates to self.units['MUA'].
    """

    name = 'animal'

    def __init__(self, name, condition, units=None, tone_period_onsets=None, tone_onsets_expanded=None):
        self.identifier = name
        self.condition = condition
        self.units = units if units else defaultdict(list)
        self.children = None
        self.tone_onsets_expanded = tone_onsets_expanded if tone_onsets_expanded is not None else []
        self.tone_period_onsets = tone_period_onsets if tone_period_onsets is not None else []
        self.data_type_context = None
        self.neuron_type_context = None
        self.PN = []
        self.IN = []
        self.last_neuron_type = None

    def update(self, context):
        if self.children is None:
            self.update_neuron_type()
        self.check_for_new_neuron_type(context)

    def update_neuron_type(self):
        if self.last_neuron_type is None:
            self.children = self.units['good']
        else:
            self.children = getattr(self, self.selected_neuron_type)


class Unit(Level):
    """A unit that was recorded from in the experiment, the child of an Animal, parent of Trials. Subscribes to
    data_type_context with an opts property and updates its trials when the trial definitions in the context change."""

    name = 'unit'

    def __init__(self, animal, category, spike_times, neuron_type=None, data_type_context=None):
        self.trials = []
        self.animal = animal
        self.category = category
        self.animal.units[category].append(self)
        self.identifier = str(self.animal.units[category].index(self) + 1)
        self.neuron_type = neuron_type
        self.trials = None
        self.children = self.trials
        self.spike_times = spike_times
        self.data_type_context = data_type_context
        self.trials_opts = None
        self.selected_trial_indices = None

    def update(self, _):
        trials_opts = (self.data_opts.get(opt) for opt in ['trials', 'post_stim', 'pre_stim'])
        if self.trials_opts != trials_opts:
            self.trials_opts = trials_opts
            self.update_trials()

    def update_trials(self):
        self.trials = []
        pre_stim, post_stim = (self.data_opts.get(opt) for opt in ['pre_stim', 'post_stim'])
        trials_slice = slice(*self.data_opts.get('trials'))
        self.selected_trial_indices = list(range(150))[slice(*self.data_opts.get('trials'))]
        for i, start in enumerate(self.animal.tone_onsets_expanded[trials_slice]):
            spikes = self.find_spikes(
                start - pre_stim * SAMPLES_PER_SECOND, start + post_stim * SAMPLES_PER_SECOND)
            self.trials.append(Trial(self, [(spike - start) / SAMPLES_PER_SECOND for spike in spikes], i))
        self.children = self.trials

    @cache_method
    def find_spikes(self, start, stop):
        return self.spike_times[bs_left(self.spike_times, start): bs_right(self.spike_times, stop)]

    @cache_method
    def get_spikes_by_trials(self):
        return [trial.spikes for trial in self.trials]

    @cache_method
    def get_pretone_means(self):
        bin_size = self.data_opts.get('bin_size')
        num_bins = int(INTER_TONE_INTERVAL / bin_size)
        rate_set = []
        for onset in self.animal.tone_period_onsets:
            start = onset - INTER_TONE_INTERVAL * SAMPLES_PER_SECOND
            stop = onset - 1
            rate_set.append(calc_rates(self.find_spikes(start, stop), num_bins, (start, stop), bin_size))
        return np.mean(rate_set, axis=1)

    @cache_method
    def get_global_std_dev(self):
        bin_size = self.data_opts.get('bin_size')
        start = self.animal.tone_period_onsets[0] - INTER_TONE_INTERVAL * SAMPLES_PER_SECOND
        stop = self.animal.tone_period_onsets[-1] + TONE_PERIOD_DURATION * SAMPLES_PER_SECOND
        num_bins = int(len(self.animal.tone_period_onsets) * (INTER_TONE_INTERVAL + TONE_PERIOD_DURATION) / bin_size)
        return np.std(calc_rates(self.find_spikes(start, stop), num_bins, (start, stop), bin_size))

    @cache_method
    def get_tone_period_std_dev(self):
        bin_size = self.data_opts.get('bin_size')
        num_bins = int(TONE_PERIOD_DURATION/bin_size)
        rates = []
        for start in self.animal.tone_period_onsets:
            stop = start + TONE_PERIOD_DURATION * SAMPLES_PER_SECOND
            p_rates = calc_rates(self.find_spikes(start, stop), num_bins, (start, stop), bin_size)
            rates.append(p_rates)
        return np.std([rate for period in rates for rate in period])

    @cache_method
    def upregulated_to_pip(self):
        if self.data_type == 'psth':
            first_pip_bin_ind = int(self.data_opts['pre_stim']/self.data_opts['bin_size'])
            last_pip_bin_ind = int(first_pip_bin_ind + .05/self.data_opts['bin_size'])
            pip_activity = np.mean(self.data[first_pip_bin_ind:last_pip_bin_ind])
            if pip_activity > .5 * np.std(self.data):
                return 1
            elif pip_activity < -.5 * np.std(self.data):
                return -1
            else:
                return 0


class Trial(Level):
    """A single trial in the experiment, the child of a unit. Aspects of a trial, for instance, the start and end of
    relevant data, can change when the parent unit's data_type_context is updated. All methods on Trial are the base
    case of the recursive methods on Level."""

    name = 'trial'

    def __init__(self, unit, spikes, index):
        self.unit = unit
        self.data_type_context = self.unit.data_type_context
        self.spikes = spikes
        self.index = index
        self.identifier = self.unit.selected_trial_indices[self.index]

    @cache_method
    def get_psth(self):
        # self.identifier is a pip's index in the original full series of pips. self.identifier // TONES_PER_PERIOD can
        # have one of five values; it's the index of the tone period
        relative_rates = self.get_rates() - self.unit.get_pretone_means()[self.identifier // TONES_PER_PERIOD]
        if self.data_opts.get('adjustment') == 'relative':
            return relative_rates
        else:
            return relative_rates / self.unit.get_tone_period_std_dev()  # same as dividing unit psth by std dev

    @cache_method
    def get_rates(self):
        pre_stim, post_stim, bin_size = (self.data_opts.get(opt) for opt in ['pre_stim', 'post_stim', 'bin_size'])
        num_bins = int((post_stim + pre_stim) / bin_size)
        spike_range = (-pre_stim, post_stim)
        return calc_rates(self.spikes, num_bins, spike_range, bin_size)

    @cache_method
    def get_all_autocorrelations(self):
        return {'trials': self._calculate_autocorrelation(self.get_demeaned_rates())}






