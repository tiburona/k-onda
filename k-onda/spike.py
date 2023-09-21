from bisect import bisect_left as bs_left, bisect_right as bs_right
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

from data import Data
from context import Subscriber
from matlab_interface import MatlabInterface
from utils import cache_method, get_ancestors
from plotting_helpers import formatted_now
from math_functions import calc_rates, spectrum, trim_and_normalize_ac, sem

"""
This module defines Level, Experiment, Group, Animal, Unit, and Trial. Level inherits from Base, which defines a few 
common properties. The rest of the classes inherit from Level and some incorporate NeuronTypeMixin for methods related 
to updating the selected neuron type. Several of Level's methods are recursive, and are overwritten by the base case, 
most frequently Trial.  
"""

SAMPLING_RATE = 30000
TONES_PER_PERIOD = 30  # The pip sounds 30 times in a tone period
TONE_PERIOD_DURATION = 30
INTER_TONE_INTERVAL = 30  # number of seconds between tone periods
PIP_DURATION = .05
NUM_TRIALS = 150
TRIAL_DURATION = 1


class Level(Data):

    @property
    def time_bins(self):
        return [TimeBin(i, data_point, self) for i, data_point in enumerate(self.data)]

    @property
    def mean(self):
        return np.mean(self.data)

    @property
    def autocorr_key(self):
        return self.get_autocorr_key()

    @cache_method
    def get_demeaned_rates(self):
        rates = self.get_average('get_rates')
        return rates - np.mean(rates)

    def get_psth(self):
        return self.get_average('get_psth')

    @cache_method
    def proportion_score(self):
        return [1 if rate > 0 else 0 for rate in self.get_psth()]

    @cache_method
    def get_proportion(self):
        return self.get_average('proportion_score', stop_at=self.data_opts.get('base'))

    @cache_method
    def get_autocorr(self):
        return self.get_all_autocorrelations()[self.autocorr_key]

    @cache_method
    def get_spectrum(self):
        freq_range, max_lag, bin_size = (self.data_opts[opt] for opt in ['freq_range', 'max_lag', 'bin_size'])
        return spectrum(self.get_autocorr(), freq_range, max_lag, bin_size)

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
            ml = MatlabInterface()
            return trim_and_normalize_ac(ml.xcorr(x, max_lag), max_lag)
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

    @cache_method
    def upregulated_to_pip(self):
        if self.data_type == 'psth':
            first_pip_bin_ind = int(self.data_opts['pre_stim'] / self.data_opts['bin_size'])
            last_pip_bin_ind = int(first_pip_bin_ind + PIP_DURATION / self.data_opts['bin_size'])
            pip_activity = np.mean(self.data[first_pip_bin_ind:last_pip_bin_ind])
            if pip_activity > .5 * np.std(self.data):
                return 1
            elif pip_activity < -.5 * np.std(self.data):
                return -1
            else:
                return 0

    def get_sem(self):
        return sem([child.data for child in self.children])


class Experiment(Level, Subscriber):
    """The experiment. Parent of groups."""

    name = 'experiment'

    def __init__(self, conditions):
        self.identifier = 'Itamar_Safety_' + formatted_now()
        self.conditions, self.groups = list(conditions.keys()), list(conditions.values())
        self.children = self.groups
        for group in self.groups:
            group.parent = self
        self.all_groups = self.groups
        self.all_animals = [animal for group in self.groups for animal in group]
        self.all_units = [unit for animal in self.all_animals for unit in animal.units['good']]
        self.last_trial_vals = None
        self.last_neuron_type = 'uninitialized'

    @property
    def all_periods(self):
        return [period for unit in self.all_units for period in unit.all_periods]

    @property
    def all_trials(self):
        return [trial for period in self.all_periods for trial in period]

    def update(self, context):
        if context.name == 'data_type_context':
            trial_vals = [self.data_opts[key] for key in ['pre_stim', 'post_stim', 'bin_size', 'trials']
                          if key in self.data_opts]
            if trial_vals != self.last_trial_vals:
                [unit.update_children() for unit in self.all_units]
                self.last_trial_vals = trial_vals
        if context.name == 'neuron_type_context':
            if self.selected_neuron_type != self.last_neuron_type:
                [entity.update_children() for entity in self.all_groups + self.all_animals]
                self.last_neuron_type = self.selected_neuron_type

    def categorize_neurons(self):
        firing_rates = [unit.firing_rate for unit in self.all_units]
        fwhm = [unit.fwhm_microseconds for unit in self.all_units]
        scaler = StandardScaler()
        firing_rates = scaler.fit_transform(np.array(firing_rates).reshape(-1, 1))
        fwhm = scaler.fit_transform(np.array(fwhm).reshape(-1, 1))
        X = np.column_stack([firing_rates, fwhm])

        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(X)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        highest_center_index = np.argmax(centers[:, 0])
        # The label of this cluster is the same as the index
        IN_label = highest_center_index
        for unit, label in zip(self.all_units, labels):
            unit.neuron_type == 'IN' if label == IN_label else 'PN'


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
        for animal in self.animals:
            animal.parent = self

    def update_children(self):
        if self.neuron_type_context.val is None:
            self.children = self.animals
        else:
            self.children = [animal for animal in self.animals if len(getattr(animal, self.neuron_type_context.val))]


class Animal(Level):
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
        self.PN = []
        self.IN = []
        self.raw_lfp = None
        self.update_children()  # Needs to be called on initialization so units gets populated

    def update_children(self):
        if self.neuron_type_context.val is None:
            self.children = self.units['good']
        else:
            self.children = getattr(self, self.neuron_type_context.val)


class Unit(Level):
    """A unit that was recorded from in the experiment, the child of an Animal, parent of Trials. Subscribes to
    data_type_context with an opts property and updates its trials when the trial definitions in the context change."""

    name = 'unit'

    def __init__(self, animal, category, spike_times, neuron_type=None):
        self.trials = []
        self.animal = animal
        self.category = category
        self.animal.units[category].append(self)
        self.identifier = str(self.animal.units[category].index(self) + 1)
        self.neuron_type = neuron_type
        self.periods = {}
        self.all_periods = []
        self.children = None
        self.spike_times = np.array(spike_times)
        self.trials_opts = None
        spikes_for_fr = self.spike_times[self.spike_times > SAMPLING_RATE * 30]
        self.firing_rate = SAMPLING_RATE * len(spikes_for_fr) / float(spikes_for_fr[-1] - spikes_for_fr[0])
        self.fwhm_microseconds = None
        self.parent = animal

    def update_children(self):
        self.periods = {'tone': [], 'pretone': []}
        trials = self.data_opts.get('trials', (1, 150))
        selected_trial_indices = list(range(NUM_TRIALS))[slice(*trials)]
        num_periods = len(set([trial // TONES_PER_PERIOD for trial in range(*trials)]))
        for i in range(num_periods):
            trials = [trial for trial in selected_trial_indices if trial // TONES_PER_PERIOD == i]
            for period_type in ['tone', 'pretone']:
                self.periods[period_type].append(Period(self, i, trials, period_type=period_type))
        self.children = self.periods['tone']
        self.all_periods = self.periods['tone'] + self.periods['pretone']

    @cache_method
    def find_spikes(self, start, stop):
        return np.array(self.spike_times[bs_left(self.spike_times, start): bs_right(self.spike_times, stop)])

    @cache_method
    def get_spikes_by_trials(self):
        return [trial.spikes for trial in self.trials]

    @cache_method
    def get_global_std_dev(self):
        bin_size = self.data_opts.get('bin_size')
        start = self.animal.tone_period_onsets[0] - INTER_TONE_INTERVAL * SAMPLING_RATE
        stop = self.animal.tone_period_onsets[-1] + TONE_PERIOD_DURATION * SAMPLING_RATE
        num_bins = int(len(self.animal.tone_period_onsets) * (INTER_TONE_INTERVAL + TONE_PERIOD_DURATION) / bin_size)
        return np.std(calc_rates(self.find_spikes(start, stop), num_bins, (start, stop), bin_size))

    @cache_method
    def get_tone_period_std_dev(self):
        return np.std([rate for period in self.children for rate in period.get_unadjusted_rates()])


class Period(Level):
    name = 'period'

    def __init__(self, unit, index, trials, period_type='tone'):
        self.unit = unit
        self.identifier = index
        self.trial_indices = trials
        self.period_type = period_type
        self.animal = self.unit.animal
        self.trials = []
        self.full_trials = []
        pip_onsets = np.array([start for i, start in enumerate(self.animal.tone_onsets_expanded) if i in trials])
        if self.period_type == 'pretone':
            pip_onsets -= TONE_PERIOD_DURATION * SAMPLING_RATE
        self.start = pip_onsets[0]
        pre_stim, post_stim = (self.data_opts.get(opt, default) * SAMPLING_RATE
                               for opt, default in [('pre_stim', 0), ('post_stim', 1)])
        # TODO: check that i is the right value here (original index)
        for i, start in enumerate(pip_onsets):
            for spike_type, pre, post in [(self.trials, pre_stim, post_stim),
                                          (self.full_trials, 0, TRIAL_DURATION*SAMPLING_RATE)]:
                spikes = self.unit.find_spikes(start - pre_stim, start + post_stim)
                spike_type.append(Trial(self, unit, [(spike - start) / SAMPLING_RATE for spike in spikes], i))
        self.children = self.trials
        self.parent = unit

    @cache_method
    def get_spikes_by_trials(self):  # TODO: This method repeated for unit and period could be mixin
        return np.array([trial.spikes for trial in self.trials])

    @cache_method
    def get_unadjusted_rates(self):
        bin_size = self.data_opts.get('bin_size')
        return [calc_rates(trial.spikes, int(TRIAL_DURATION / bin_size), (0, TRIAL_DURATION), bin_size)
                for trial in self.trials]

    @cache_method
    def mean_firing_rate(self):
        return np.mean(self.get_unadjusted_rates())

    def get_spikes(self, border=0):
        if self.period_type == 'pretone':
            start = self.start - 3 * border * SAMPLING_RATE
        else:
            start = self.start - border * SAMPLING_RATE
        end = start + (TONE_PERIOD_DURATION + 2 * border) * SAMPLING_RATE
        return self.unit.find_spikes(start, end), start, end


class Trial(Level):
    """A single trial in the experiment, the child of a unit. Aspects of a trial, for instance, the start and end of
    relevant data, can change when the parent unit's data_type_context is updated. All methods on Trial are the base
    case of the recursive methods on Level."""

    name = 'trial'
    instances = []

    def __init__(self, period, unit, spikes, index):
        self.unit = unit
        if self.unit.category == 'good':
            self.instances.append(self)
        self.spikes = spikes
        self.identifier = index
        self.period = period
        self.period_type = self.period.period_type
        self.children = None
        self.parent = period

    @cache_method
    def get_psth(self):
        rates = self.get_rates()
        if self.period.period_type == 'pretone' or self.data_opts.get('adjustment') == 'none':
            return rates
        rates -= self.unit.periods['pretone'][self.period.identifier].mean_firing_rate()
        if self.data_opts.get('adjustment') == 'relative':
            return rates
        rates /= self.unit.get_tone_period_std_dev()  # same as dividing unit psth by std dev
        return rates

    @cache_method
    def get_rates(self):
        pre_stim, post_stim, bin_size = (self.data_opts.get(opt) for opt in ['pre_stim', 'post_stim', 'bin_size'])
        num_bins = int((post_stim + pre_stim) / bin_size)
        spike_range = (-pre_stim, post_stim)
        return calc_rates(self.spikes, num_bins, spike_range, bin_size)

    @cache_method
    def get_all_autocorrelations(self):
        return {'trials': self._calculate_autocorrelation(self.get_demeaned_rates())}


class TimeBin:

    name = 'time_bin'
    instances = []

    def __init__(self, i, val, parent):
        self.instances.append(self)
        self.parent = parent
        self.identifier = i
        self.data = val

    @property
    def ancestors(self):
        return get_ancestors(self)

    #  TODO: parametrize methods below
    @staticmethod
    def two_way_split(self):
        return 'early' if self.identifer < 30 else 'late'

    @staticmethod
    def three_way_split(self):
        if self.identifier < 5:
            return 'pip'
        elif self.identifier < 30:
            return 'early'
        else:
            return 'late'


