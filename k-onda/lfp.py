import os
import pickle
from scipy.signal import cwt, morlet
from neo.rawio import BlackrockRawIO

from spike import Animal, Unit
from proxy import Proxy
from data import Data
from context import data_type_context
from matlab_interface import MatlabInterface
from math_functions import *

SAMPLING_RATE = 30000
TONES_PER_PERIOD = 30  # The pip sounds 30 times in a tone period
TONE_PERIOD_DURATION = 30
INTER_TONE_INTERVAL = 30  # number of seconds between tone periods
PIP_DURATION = .05
LFP_SAMPLING_RATE = 2000
FREQUENCY_BANDS = dict(delta=(0, 4), theta_1=(4, 8), theta_2=(4, 12), delta_theta=(0, 12), gamma=(20, 55),
                       hgamma=(70, 120))
LO_FREQ_ARGS = (2048, 2000, 1000, 980, 2)
FREQUENCY_ARGS = {fb: LO_FREQ_ARGS for fb in ['delta', 'theta_1', 'theta_2', 'delta_theta']}


def initialize_lfp():
    return [LFPAnimal(animal) for animal in Animal.instances]


class LFP(Data):

    instances = []
    data_type_context = data_type_context

    @property
    def data(self):
        return getattr(self, f"get_{self.data_type}")()

    @property
    def brain_region(self):
        return self.data_opts.get('brain_region')

    @classmethod
    def update(cls, context):
        pass


class LFPAnimal(LFP, Proxy):
    """A wrapper around the spike_data.Animal class.  Collects raw LFP data for an animal, and divides it into Periods,
    as well as initializing FrequencyPeriods, subsets of the data for a Period for a given frequency range."""

    instances = []

    def __init__(self, animal):
        self.instances.append(self)
        Proxy.__init__(self, animal, Animal)
        self.raw_lfp = self.initialize_lfp_data()
        self.frequency_args = set(tuple(args) for fb, args in FREQUENCY_ARGS.items() if fb in self.data_opts['fb'])
        self.periods = self.prepare_periods()
        self.frequency_periods = self.prepare_frequency_periods()
        self.units = [LFPUnit(unit, self) for unit in self._target_instance]

    def initialize_lfp_data(self):
        file_path = os.path.join(self.data_opts['data_path'], 'single_cell_data', self.identifier, 'Safety')
        reader = BlackrockRawIO(filename=file_path, nsx_to_load=3)
        reader.parse_header()
        return {'hpc': reader.nsx_datas[3][0][:, 0], 'bla': reader.nsx_datas[3][0][:, 2],
                'pl': np.mean([reader.nsx_datas[3][0][:, 1], reader.nsx_datas[3][0][:, 3]], axis=0)}

    def prepare_periods(self):
        periods = []
        onsets = [int(onset * LFP_SAMPLING_RATE / SAMPLING_RATE) for onset in self.tone_period_onsets]
        stage_intervals = {'tone': [(tpo - LFP_SAMPLING_RATE, tpo + (TONE_PERIOD_DURATION + 1) * LFP_SAMPLING_RATE)
                           for tpo in onsets],
                           'pretone': [(tpo - LFP_SAMPLING_RATE - 1 - ((TONE_PERIOD_DURATION + 2) * LFP_SAMPLING_RATE),
                                        tpo - LFP_SAMPLING_RATE - 1) for tpo in onsets]
                           }
        raw = self.raw_lfp[self.brain_region]
        for arg_set in self.frequency_args:
            for key in stage_intervals:
                for i, period in enumerate(stage_intervals[key]):
                    periods.append(LFPPeriod(raw[slice(*period)], i, key, self, arg_set))
        return periods

    def prepare_frequency_periods(self):
        freq_periods = {}
        frequency_bands = [fb for fb in FREQUENCY_BANDS if fb in self.data_opts['fb']]
        for fb in frequency_bands:
            arg_set = FREQUENCY_ARGS[fb]
            freq_periods[fb] = [FrequencyPeriod(period, fb) for period in self.periods if period.arg_set == arg_set]
        return freq_periods

    def normalize_average_power(self):
        normalized_power = {}
        frequency_bands = [fb for fb in FREQUENCY_BANDS if fb in self.data_opts['fb']]
        for fb in frequency_bands:
            tone_power, pretone_power = [np.array([p.average_power for p in self.frequency_periods[fb]
                                                   if p.period_type == stage]) for stage in ('tone', 'pretone')]
            normalized_power[fb] = tone_power - pretone_power
        return normalized_power


class LFPPeriod(LFP):
    """An interval in the experiment. Preprocesses data and initiates calls to Matlab to get the cross-spectrogram."""
    def __init__(self, raw_data, index, period_type, animal, arg_set):
        self.raw_data = raw_data
        self.identifier = index
        self.animal = animal
        self.arg_set = arg_set
        self.processed_data = self.process_lfp(raw_data)
        self.mrl_data = self.processed_data[LFP_SAMPLING_RATE:-LFP_SAMPLING_RATE]
        self.period_type = period_type
        self.parent = animal

    @property
    def spectrogram(self):
        return self.calc_cross_spectrogram(self.processed_data)

    @staticmethod
    def process_lfp(raw):
        filtered = filter_60_hz(raw, 2000)
        return divide_by_rms(filtered)

    def calc_cross_spectrogram(self, data):
        pickle_path = os.path.join(self.data_opts['data_path'], 'lfp', '_'.join(
                [self.animal.identifier, self.data_opts['brain_region']] +
                [str(arg) for arg in self.arg_set] +
                [self.period_type, str(self.identifier)]) + '.pkl')
        if os.path.exists(pickle_path) and not self.data_opts.get('force_recalc'):
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        ml = MatlabInterface()
        result = ml.mtcsg(data, *FREQUENCY_ARGS[self.data_opts['fb']])
        with open(pickle_path, 'wb') as f:
            pickle.dump(result, f)
        return result


class LFPUnit(LFP, Proxy):
    """A wrapper around the spike_data.Unit class."""
    instances = []

    def __init__(self, unit, lfp_animal):
        Proxy.__init__(self, unit, Unit)
        super().__init__()
        self.parent = lfp_animal


class TrialCalculatorMixin:
    """Defines methods shared by FrequencyPeriod and FrequencyBin to subdivide themselves into trials and get
    averages. """
    def get_trials(self):
        starts = np.arange(.75, 30.5, 1)
        time_bins = np.array(self.period.spectrogram[2])
        trials = []
        for start in starts:
            trial_times = np.linspace(start, start + .64, 65)
            mask = (np.abs(time_bins[:, None] - trial_times) <= 1e-6).any(axis=1)
            trials.append(np.array(self.time_series)[mask])
        return trials

    def get_average_over_trials(self):
        return np.mean(self.get_trials(), axis=0)

    def get_average_over_time_bins(self):
        return np.mean(self.get_average_over_trials())


class FrequencyPeriod(LFP, Proxy, TrialCalculatorMixin):
    """A FrequencyPeriod is a Period with selected frequency range. This class slices a cross-spectrogram into its
      appropriate frequency range and its constituent trials, and calculates averages over those trials. It is the parent
      of FrequencyBin."""

    instances = []
    name = 'period'  # So named for later merging with spike data

    def __init__(self, period, fb):
        Proxy.__init__(self, period, LFPPeriod)
        self.instances.append(self)
        self.fb = fb
        self.freq_range = FREQUENCY_BANDS[fb]
        self.widths = None

    def slice_spectrogram(self):
        indices = np.where(self.spectrogram[1] <= self.freq_range[0])
        ind1 = indices[0][-1] if indices[0].size > 0 else None  # last index that's <= start of the freq range
        ind2 = np.argmax(self.spectrogram[1] > self.freq_range[1])  # first index >= end of freq range
        return self.spectrogram[0][ind1:ind2, :]

    def get_power(self):
        if self.data_opts['frequency'] == 'continuous':
            [FrequencyBin(i, data, self) for i, data in enumerate(self.slice_spectrogram())]
        return self.get_average_over_time_bins()  # TODO: Make this more general when I want to get other kinds of data

    @property
    def time_series(self):
        return np.mean(self.slice_spectrogram(), axis=0)

    def get_wavelets(self):
        self.widths = np.arange(1, 100)  # range of scales
        return cwt(self.processed_data, morlet, self.widths)

    def get_wavelet_phases(self, frequency):
        cwt_matrix = self.get_wavelets()
        if frequency == 0:
            frequency += 10 ** -6
        scale_idx = np.argmin(np.abs(self.widths - (LFP_SAMPLING_RATE / frequency)))
        return np.angle(cwt_matrix[scale_idx, :])

    def get_weights(self, unit):
        spike_period = unit.periods[self.period_type][self.identifier]
        spikes, start, end = spike_period.get_spikes(border=0)
        spikes = [int((spike - start) / (SAMPLING_RATE / LFP_SAMPLING_RATE)) for spike in spikes]
        weights_length = int((end - start) / (SAMPLING_RATE / LFP_SAMPLING_RATE))
        return np.array([1 if weight in spikes else float('nan') for weight in range(weights_length)])

    def get_mrl(self):
        unit_mrls = [self.mrl(unit.update_children()) for unit in self.animal]
        return np.mean(unit_mrls)

    def mrl(self, unit):
        fb = FREQUENCY_BANDS[self.fb]
        weights = self.get_weights(unit)
        if self.data_opts.get('phase') == 'wavelet':
            alpha = np.array([self.get_phases(frequency) for frequency in range(*fb)])
        else:
            low = fb[0] if fb[0] > 0 else .1
            high = fb[1]
            alpha = compute_phase(bandpass_filter(self.mrl_data, low, high, LFP_SAMPLING_RATE))
        mrl = circ_r2_unbiased(alpha, weights)
        if mrl.shape[0] > 1:
            for i, frequency in enumerate(range(*fb)):
                FrequencyUnit(unit, self, frequency, mrl[i])
        else:
            FrequencyUnit(unit, self, FREQUENCY_BANDS[self.fb], mrl[0])
        return mrl


class FrequencyBin(LFP, TrialCalculatorMixin):
    """A FrequencyBin contains a slice of cross-spectrogram at the smallest available frequency resolution."""

    name = 'frequency_bin'
    instances = []

    def __init__(self, index, data, frequency_period):
        super().__init__()
        self.parent = frequency_period
        self.period = self.parent.period
        self.period_type = self.parent.period_type
        self.identifier = self.period.spectrogram[1][index]
        self.fb = self.parent.fb
        self.children = None
        self.time_series = data

    @property
    def data(self):
        if self.data_opts['time'] == 'continuous':
            self.children = [TimeBin(i, power, self) for i, power in enumerate(self.get_average_over_trials())]
        return self.get_average_over_time_bins()


class FrequencyUnit(LFPUnit):
    """A FrequencyUnit contains a representation of the MRL calculation for a given Unit and a given frequency."""

    instances = []

    def __init__(self, lfp_unit, period, frequency, mrl):
        Proxy.__init__(self, lfp_unit, LFPUnit)
        self.instances.append(self)
        self.frequency = frequency  # integer (single frequency) or a range (average over frequencies)
        self.period = period
        self.parent = period
        self.mrl = mrl

    def get_mrl(self):
        return self.mrl


class TimeBin:

    name = 'time_bin'
    instances = []

    def __init__(self, i, power, parent):
        self.instances.append(self)
        self.parent = parent
        self.period_type = self.parent.period_type
        self.identifier = i
        self.data = power



