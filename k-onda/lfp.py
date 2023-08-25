import os
import pickle
from scipy.signal import cwt, morlet
from neo.rawio import BlackrockRawIO

from data import Data
from matlab_interface import MatlabInterface
from math_functions import *
from utils import get_ancestors

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


class LFPData(Data):
    
    @property
    def brain_region(self):
        return self.data_opts.get('brain_region')

    @property
    def frequency_bins(self):
        return [FrequencyBin(i, data_point, self) for i, data_point in enumerate(self.data)]

    @property
    def time_bins(self):
        tbs = []
        if len(self.data.shape) > 1:
            for i, data_point in enumerate(range(self.data.shape[1])):
                column = self.data[:, i]
                TimeBin(i, column, self)
                tbs.append(TimeBin)
            return tbs
        else:
            return [TimeBin(i, data_point, self) for i, data_point in enumerate(self.data)]

    @property
    def current_frequency_band(self):
        return self.data_opts.get('frequency_band')


class LFPExperiment(LFPData):

    name = 'experiment'

    def __init__(self, experiment):
        self.experiment = experiment
        self.all_animals = [LFPAnimal(animal) for animal in self.experiment.all_animals]
        self.all_periods = [period for animal in self.all_animals for stage in animal.periods
                            for period in animal.periods[stage]]

    @property
    def all_mrl_calculators(self):
        mrl_calculators = []
        for period in self.all_periods:
            for unit in self.experiment.all_units:
                mrl_calculators.append(MRLCalculator(unit, period))
        return mrl_calculators


class LFPAnimal(LFPData):
    """An animal in the experiment. Processes the raw LFP data and divides it into periods."""

    name = 'animal'

    def __init__(self, animal):
        self.animal = animal
        file_path = os.path.join(self.data_opts['data_path'], 'single_cell_data', self.animal.identifier, 'Safety')

        reader = BlackrockRawIO(filename=file_path, nsx_to_load=3)
        reader.parse_header()

        self.raw_lfp_unique_name = {
            'hpc': reader.nsx_datas[3][0][:, 0],
            'bla': reader.nsx_datas[3][0][:, 2],
            'pl': np.mean([reader.nsx_datas[3][0][:, 1], reader.nsx_datas[3][0][:, 3]], axis=0)
        }

        self.frequency_args = set(tuple(args) for fb, args in FREQUENCY_ARGS.items() if fb in self.data_opts['fb'])
        self.periods = self.prepare_periods()
        self.identifier = self.animal.identifier

    def prepare_periods(self):
        periods = {'tone': [], 'pretone': []}
        onsets = [int(onset * LFP_SAMPLING_RATE / SAMPLING_RATE) for onset in self.animal.tone_period_onsets]
        stage_intervals = {'tone': [(tpo - LFP_SAMPLING_RATE, tpo + (TONE_PERIOD_DURATION + 1) * LFP_SAMPLING_RATE)
                           for tpo in onsets],
                           'pretone': [(tpo - LFP_SAMPLING_RATE - 1 - ((TONE_PERIOD_DURATION + 2) * LFP_SAMPLING_RATE),
                                        tpo - LFP_SAMPLING_RATE - 1) for tpo in onsets]
                           }
        raw = self.raw_lfp_unique_name[self.brain_region]
        for arg_set in self.frequency_args:
            for key in stage_intervals:
                for i, period in enumerate(stage_intervals[key]):
                    periods[key].append(LFPPeriod(raw[slice(*period)], i, key, self, arg_set))
        return periods


class LFPDataSelector:

    """A class with methods shared by LFPPeriod and LFPTrial that are used to return portions of their data."""

    @property
    def freq_range(self):
        return FREQUENCY_BANDS[self.current_frequency_band]

    @property
    def frequency_bins(self):
        return [FrequencyBin(i, data_point, self) for i, data_point in enumerate(self.data)]

    @property
    def time_bins(self):
        return [TimeBin(i, data_column, self) for i, data_column in enumerate(self.data.T)]

    @property
    def mean_over_time_bins(self):
        return np.mean(self.data, axis=1)

    @property
    def mean_over_frequency(self):
        return np.mean(self.data, axis=0)

    @property
    def mean(self):
        return np.mean(self.data)

    def slice_spectrogram(self):
        indices = np.where(self.spectrogram[1] <= self.freq_range[0])
        ind1 = indices[0][-1] if indices[0].size > 0 else None  # last index that's <= start of the freq range
        ind2 = np.argmax(self.spectrogram[1] > self.freq_range[1])  # first index >= end of freq range
        return self.spectrogram[0][ind1:ind2, :]


class LFPPeriod(LFPData, LFPDataSelector):
    """A block in the experiment. Preprocesses data, initiates calls to Matlab to get the cross-spectrogram, and
    generates LFPTrials. Inherits from LFPSelector to be able to return portions of its data."""

    name = 'period'

    def __init__(self, raw_data, index, period_type, lfp_animal, arg_set):
        self.raw_data = raw_data
        self.identifier = index
        self.lfp_animal = lfp_animal
        self.arg_set = arg_set
        self.processed_data = self.process_lfp(raw_data)
        self.mrl_data = self.processed_data[LFP_SAMPLING_RATE:-LFP_SAMPLING_RATE]
        self.period_type = period_type
        self.parent = lfp_animal

    @property
    def trials(self):
        self.get_trials()

    @property
    def spectrogram(self):
        return self.calc_cross_spectrogram()

    def get_trials(self):
        starts = np.arange(.75, 30.5, 1)
        time_bins = np.array(self.spectrogram[2])
        trials = []
        for start in starts:
            trial_times = np.linspace(start, start + .64, 65)
            mask = (np.abs(time_bins[:, None] - trial_times) <= 1e-6).any(axis=1)
            trials.append(LFPTrial(trial_times, mask, self))
        return trials

    @staticmethod
    def process_lfp(raw):
        filtered = filter_60_hz(raw, 2000)
        return divide_by_rms(filtered)

    def calc_cross_spectrogram(self):
        pickle_path = os.path.join(self.data_opts['data_path'], 'lfp', '_'.join(
                [self.lfp_animal.animal.identifier, self.data_opts['brain_region']] +
                [str(arg) for arg in self.arg_set] +
                [self.period_type, str(self.identifier)]) + '.pkl')
        if os.path.exists(pickle_path) and not self.data_opts.get('force_recalc'):
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        ml = MatlabInterface()
        result = ml.mtcsg(self.processed_data, *FREQUENCY_ARGS[self.data_opts['frequency']])
        with open(pickle_path, 'wb') as f:
            pickle.dump(result, f)
        return result

    def get_power(self):
        power = np.mean([trial.data for trial in self.get_trials()], axis=0)
        if self.data_opts.get('adjustment') == 'normalize' and self.period_type == 'tone':
            power -= self.animal.periods['pretone'][self.identifier].get_power()
        return power


class LFPTrial(LFPData, LFPDataSelector):

    name = 'trial'

    def __init__(self, trial_times, mask, parent):
        self.trial_times = trial_times
        self.mask = mask
        self.parent = parent
        self.spectrogram = self.parent.spectrogram

    def get_power(self):
        return np.array(self.slice_spectrogram())[:, self.mask]


class FrequencyBin(LFPData):
    """A FrequencyBin contains a slice of cross-spectrogram or mrl calculation at the smallest available frequency
    resolution."""

    name = 'frequency_bin'

    def __init__(self, index, val, parent, unit=None):
        self.parent = parent
        self.val = val
        if isinstance(parent, LFPPeriod):
            self.identifier = self.period.spectrogram[1][index]
        else:  # parent is MRLCalculator  # TODO: this is really an approximation of what frequencies these actually were; make more exact
            self.identifier = list(range(*FREQUENCY_BANDS[self.current_frequency_band]))[index]
        self.period_type = self.parent.period_type
        self.unit = unit

    @property
    def data(self):
        return self.val


class TimeBin:

    name = 'time_bin'

    def __init__(self, i, data, parent):
        self.parent = parent
        self.period_type = self.parent.period_type
        self.identifier = i
        self.data = data


class MRLCalculator(LFPData):
    """Calculates the Mean Resultant Length of the vector that represents the phase of a frequency in the LFP data on
    the occasion the firing of a neuron. MRL """

    name = 'mrl_calculator'

    def __init__(self, unit, period):
        self.unit = unit
        self.period = period
        self.period_type = period.period_type
        self.parent = self.unit.parent
        self.identifier = f"{period.identifier}_{unit.identifier}"

    @property
    def ancestors(self):
        custom_ancestors = [self.unit] + [self.period]
        return [self] + custom_ancestors + self.parent.ancestors

    @property
    def mean_over_frequency(self):
        return np.mean(self.data, axis=0)

    @property
    def frequency_bins(self):
        return [FrequencyBin(i, data, self, unit=self.unit) for i, data in enumerate(self.data)]

    def get_weights(self):
        spike_period = self.unit.periods[self.period_type][self.period.identifier]
        spikes = [int((spike + i) * LFP_SAMPLING_RATE) for i, trial in enumerate(spike_period.trials) for spike in
                  trial.spikes]

        return np.array(
            [1 if weight in spikes else float('nan') for weight in range(TONES_PER_PERIOD * LFP_SAMPLING_RATE)])

    def get_wavelet_phases(self, frequency):
        """Get wavelet phases for a specific frequency."""
        scale = get_wavelet_scale(frequency, LFP_SAMPLING_RATE)

        # Compute the continuous wavelet transform
        cwt_matrix = cwt(self.period.mrl_data, morlet, [scale])

        if frequency == 0:
            frequency += 10 ** -6

        # Since we're computing the CWT for only one scale, the result is at index 0.
        return np.angle(cwt_matrix[0, :])

    def get_mrl(self):
        """Compute mean resultant length for the desired frequency band."""
        fb = FREQUENCY_BANDS[self.current_frequency_band]
        weights = self.get_weights()
        low = fb[0] + .1
        high = fb[1]
        if self.data_opts.get('phase') == 'wavelet':
            alpha = np.array([self.get_wavelet_phases(frequency)
                              for frequency in np.linspace(low, high, int(high - low) + 1)])
        else:
            alpha = compute_phase(bandpass_filter(self.period.mrl_data, low, high, LFP_SAMPLING_RATE))

        return circ_r2_unbiased(alpha, weights, dim=1)



