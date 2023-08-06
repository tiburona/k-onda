import os
import pickle
from matlab_interface import MatlabInterface
from math_functions import *
import numpy as np

SAMPLING_RATE = 30000
TONES_PER_PERIOD = 30  # The pip sounds 30 times in a tone period
TONE_PERIOD_DURATION = 30
INTER_TONE_INTERVAL = 30  # number of seconds between tone periods
PIP_DURATION = .05
LFP_SAMPLING_RATE = 2000
FREQUENCY_BANDS = dict(delta=(0, 4), theta_1=(4, 8), theta_2=(4, 12), gamma=(20, 55), hgamma=(70, 120))


class Period:
    """An interval in the experiment. Preprocesses data and initiates calls to Matlab to get the cross-spectrogram."""
    def __init__(self, raw_data, animal, period_type, data_opts, num):
        self.raw_data = raw_data
        self.animal = animal
        self.period_type = period_type
        self.data_opts = data_opts
        self.identifier = num
        self.processed_data = self.process_lfp(raw_data)
        self.spectrogram = self.calc_cross_spectrogram(self.processed_data)

    def process_lfp(self, raw):
        filtered = filter_60_hz(raw, 2000)
        return divide_by_rms(filtered)

    def calc_cross_spectrogram(self, data):
        frequency_args = dict(theta_1=[2048, 2000, 1000, 980, 2], theta_2=[2048, 2000, 1000, 980, 2])
        pickle_path = os.path.join(self.data_opts['data_path'], 'lfp', '_'.join(
                [self.animal.identifier, self.data_opts['brain_region']] +
                [str(arg) for arg in frequency_args[self.data_opts['fb']]] +
                [self.period_type, str(self.identifier)]) + '.pkl')
        if os.path.exists(pickle_path) and not self.data_opts.get('force_recalc'):
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        ml = MatlabInterface()
        result = ml.mtcsg(data, *frequency_args[self.data_opts['fb']])
        with open(pickle_path, 'wb') as f:
            pickle.dump(result, f)
        return result


class FrequencyPeriod:

    instances = []
    name = 'period'  # So named for later merging with spike data

    """A FrequencyPeriod is a Period with selected frequency range. This class slices a cross-spectrogram into its
    appropriate frequency range and its constituent trials, and calculates averages over those trials."""
    def __init__(self, period_type, period, animal, freq_range):
        self.instances.append(self)
        self.period_type = period_type
        self.period = period
        self.parent = animal
        self.freq_range = freq_range
        self.identifier = period.identifier
        self.power_in_freq_range = self.get_power_in_freq_range()
        self.average_power = self.get_average_over_trials()
        self.trials = []

    @property
    def data(self):
        return self.get_average_over_trials()  # TODO: Make this more general when I want to get other kinds of data

    @property
    def mean(self):
        return self.get_average_over_trials()

    def get_power_in_freq_range(self):
        indices = np.where(self.period.spectrogram[1] <= self.freq_range[0])
        ind1 = indices[0][-1] if indices[0].size > 0 else None   # last index that's <= start of the freq range
        ind2 = np.argmax(self.period.spectrogram[1] > self.freq_range[1])  # first index >= end of freq range
        return np.mean(self.period.spectrogram[0][ind1:ind2, :], axis=0)

    def get_trials(self):
        starts = np.arange(.75, 30.5, 1)

        time_bins = np.array(self.period.spectrogram[2])
        trials = []
        for start in starts:
            trial_times = np.linspace(start, start + .64, 65)

            mask = (np.abs(time_bins[:, None] - trial_times) <= 1e-6).any(axis=1)
            trials.append(np.array(self.get_power_in_freq_range())[mask])
        return trials

    def get_average_trials(self):
        return np.mean(self.get_trials(), axis=0)

    def get_average_over_trials(self):
        return np.mean(self.get_average_trials())


class LFP:
    """ An LFP is associated with an Animal. For a given brain region, divides raw data into tone and pretone
    Stages and Periods within those Stages. In its __init__ method, executes a method that sets the normalized_power
    property for access by other classes, such as Stats. """

    def __init__(self, animal, brain_region, data_opts):
        self.animal = animal
        self.brain_region = brain_region
        self.data_opts = data_opts
        self.periods = []
        self.prepare_periods()
        self.frequency_periods = self.prepare_frequency_periods()
        self.normalized_power = self.normalize_average_power()

    def prepare_periods(self):
        onsets = [int(onset * LFP_SAMPLING_RATE / SAMPLING_RATE) for onset in self.animal.tone_period_onsets]
        stage_intervals = {'tone': [(tpo - LFP_SAMPLING_RATE, tpo + (TONE_PERIOD_DURATION + 1) * LFP_SAMPLING_RATE)
                           for tpo in onsets],
                           'pretone': [(tpo - LFP_SAMPLING_RATE - 1 - ((TONE_PERIOD_DURATION + 2) * LFP_SAMPLING_RATE),
                                        tpo - LFP_SAMPLING_RATE - 1) for tpo in onsets]
                           }
        raw = self.animal.raw_lfp[self.brain_region]
        for key in stage_intervals:
            for i, period in enumerate(stage_intervals[key]):
                self.periods.append(Period(raw[slice(*period)], self.animal, key, self.data_opts, i))

    def prepare_frequency_periods(self):
        freq_periods = {}
        frequency_bands = [fb for fb in FREQUENCY_BANDS if fb in self.data_opts['fb']]
        for fb in frequency_bands:
            freq_periods[fb] = [FrequencyPeriod(period.period_type, period, self.animal, FREQUENCY_BANDS[fb])
                                for period in self.periods]
        return freq_periods

    def normalize_average_power(self):
        normalized_power = {}
        frequency_bands = [fb for fb in FREQUENCY_BANDS if fb in self.data_opts['fb']]
        for fb in frequency_bands:
            tone_power, pretone_power = [np.array([p.average_power for p in self.frequency_periods[fb]
                                                   if p.period_type == stage]) for stage in ('tone', 'pretone')]
            normalized_power[fb] = tone_power - pretone_power
        return normalized_power
