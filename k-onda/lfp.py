import os
import pickle
from scipy.signal import cwt, morlet
from neo.rawio import BlackrockRawIO

from data import Data, EvokedValueCalculator
from context import Subscriber, period_type_context as pt_context
from matlab_interface import MatlabInterface
from math_functions import *
from utils import cache_method

DATA_PATH = '/Users/katie/likhtik/data/single_cell_data'
SAMPLING_RATE = 30000
TONES_PER_PERIOD = 30  # The pip sounds 30 times in a tone period
TONE_PERIOD_DURATION = 30
INTER_TONE_INTERVAL = 30  # number of seconds between tone periods
PIP_DURATION = .05
LFP_SAMPLING_RATE = 2000
FREQUENCY_BANDS = dict(delta=(0, 4), theta_1=(4, 8), theta_2=(4, 12), delta_theta=(0, 12), gamma=(20, 55),
                       hgamma=(70, 120))
LO_FREQ_ARGS = (2048, 2000, 1000, 980, 2)
FREQUENCY_ARGS = {fb: LO_FREQ_ARGS for fb in ['delta', 'theta_1', 'theta_2', 'delta_theta', 'gamma', 'hgamma']}
# TODO: gamma and hgamma don't really belong there, find out what their args should be.

PL_ELECTRODES = {
    'IG154': (4, 6), 'IG155': (12, 14), 'IG156': (12, 14), 'IG158': (7, 14), 'IG160': (1, 8), 'IG161': (9, 11),
    'IG162': (13, 3), 'IG163': (14, 8), 'IG175': (15, 4), 'IG176': (11, 12), 'IG177': (15, 4), 'IG178': (6, 14),
    'IG179': (13, 15), 'IG180': (15, 4)
}


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

    @property
    def freq_range(self):
        if isinstance(self.current_frequency_band, type('str')):
            return FREQUENCY_BANDS[self.current_frequency_band]
        else:
            return self.current_frequency_band

    @property
    def selected_period_type(self):
        return self.period_type_context.val

    def get_mrl(self):
        axis = 0 if not self.data_opts.get('collapse_matrix') else None
        return self.get_average('get_mrl', stop_at='mrl_calculator', axis=axis)


class LFPExperiment(LFPData, Subscriber):
    name = 'experiment'

    def __init__(self, experiment):
        self.data_path = DATA_PATH
        self.experiment = experiment
        self.groups = [LFPGroup(group, self) for group in self.experiment.groups]
        self.all_groups = self.groups
        self.all_animals = [LFPAnimal(animal, self.data_path) for animal in self.experiment.all_animals]
        self.last_brain_region = None

    @property
    def all_periods(self):
        periods = []
        for animal in self.all_animals:
            [periods.extend(animal_periods) for animal_periods in animal.all_periods.values()]
        return periods

    @property
    def all_mrl_calculators(self):
        return [MRLCalculator(unit, period) for animal in self.all_animals
                for period in animal.all_periods['tone'] + animal.all_periods['pretone']
                for unit in animal.spike_target.children]

    def update(self, context):
        if context.name == 'data_type_context':
            if self.data_class == 'lfp' and self.last_brain_region != self.data_opts.get('brain_region'):
                [animal.update_children() for animal in self.all_animals]
                self.last_brain_region = self.data_opts.get('brain_region')


class LFPGroup(LFPData):
    name = 'group'

    def __init__(self, group, lfp_experiment):
        self.spike_target = group
        self.experiment = lfp_experiment
        self.identifier = self.spike_target.identifier

    @property
    def children(self):
        return [animal for animal in self.experiment.all_animals if animal.condition == self.identifier]

    @property
    def mrl_calculators(self):
        return [mrl_calc for animal in self.children for mrl_calc in animal.mrl_calculators if mrl_calc.is_valid]

    @property
    def grandchildren_scatter(self):
        if self.data_type != 'mrl':
            raise NotImplementedError("Grandchildren Scatter is currently only implemented for MRL")
        else:
            unit_points = []
            for animal in self.children:
                for unit in animal.spike_target.children:
                    unit_points.append(np.mean([mrl_calc.data for mrl_calc in animal.children
                                                if mrl_calc.unit.identifier == unit.identifier]))
            return unit_points

    @property
    def data_by_period(self):
        if self.data_type != 'mrl':
            raise NotImplementedError("Grandchildren Scatter is currently only implemented for MRL")
        data_by_period = []
        for i in range(5):
            data_by_period.append(
                np.mean(
                    [mrl_calc.data for animal in self.children for mrl_calc in animal.children
                     if mrl_calc.period.identifier == i], axis=0)
            )
        return np.array(data_by_period)

    def get_angle_counts(self):
        for calc in self.mrl_calculators:
            if any(np.isnan(calc.get_angle_counts())):
                print(f"{calc.parent.identifier} {calc.identifier}")
        counts = np.sum(np.array([calc.get_angle_counts() for calc in self.mrl_calculators]), axis=0)
        return counts

    def get_mrl(self):
        return self.get_average('get_mrl', stop_at='mrl_calculator', axis=None)


class LFPAnimal(LFPData):
    """An animal in the experiment. Processes the raw LFP data and divides it into periods."""

    name = 'animal'

    def __init__(self, animal, data_path):
        self.spike_target = animal
        self.data_path = data_path
        self.raw_lfp = self.get_raw_lfp()
        self.period_intervals = self.get_period_intervals()
        self.all_periods = None
        self.all_mrl_calculators = None

    def __getattr__(self, name):
        return getattr(self.spike_target, name)

    @property
    def frequency_args(self):
        return set(tuple(args) for fb, args in FREQUENCY_ARGS.items() if fb in self.data_opts['fb'])

    @property
    def mrl_calculators(self):
        if self.selected_period_type is None:
            calcs = [calc for calc in self.all_mrl_calculators['tone'] + self.all_mrl_calculators['pretone']]
        else:
            calcs = [calc for calc in self.all_mrl_calculators[self.selected_period_type]]
        return [calc for calc in calcs if calc.unit in self.spike_target.children and calc.is_valid]

    @property
    def periods(self):
        if self.all_periods is None:
            self.all_periods = self.prepare_periods()
        if self.selected_period_type:
            return self.all_periods[self.selected_period_type]
        else:
            return self.all_periods['tone'] + self.all_periods['pretone']

    @property
    def children(self):
        return self.mrl_calculators if self.data_type == 'mrl' else self.periods

    def update_children(self):
        self.all_periods = self.prepare_periods()
        if 'mrl' in self.data_type:
            self.all_mrl_calculators = self.prepare_mrl_calculators()

    def get_raw_lfp(self):
        file_path = os.path.join(self.data_path, self.identifier, 'Safety')
        reader = BlackrockRawIO(filename=file_path, nsx_to_load=3)
        reader.parse_header()
        pl1, pl2 = PL_ELECTRODES[self.identifier]
        return {
            'hpc': reader.nsx_datas[3][0][:, 0],
            'bla': reader.nsx_datas[3][0][:, 2],
            'pl': np.mean([reader.nsx_datas[3][0][:, pl1], reader.nsx_datas[3][0][:, pl2]], axis=0)
        }

    def get_period_intervals(self):
        onsets = [int(onset * LFP_SAMPLING_RATE / SAMPLING_RATE) for onset in self.tone_period_onsets]
        period_intervals = {'tone': [(tpo - LFP_SAMPLING_RATE, tpo + (TONE_PERIOD_DURATION + 1) * LFP_SAMPLING_RATE)
                                     for tpo in onsets],
                            'pretone': [(tpo - LFP_SAMPLING_RATE - 1 - ((TONE_PERIOD_DURATION + 2) * LFP_SAMPLING_RATE),
                                        tpo - LFP_SAMPLING_RATE - 1) for tpo in onsets]
                           }
        return period_intervals

    def prepare_periods(self):
        raw = self.raw_lfp[self.brain_region]
        return {stage: [LFPPeriod(raw[slice(*period)], i, stage, self) for i, period in enumerate(periods)]
                for stage, periods in self.period_intervals.items()}

    def prepare_mrl_calculators(self):
        return {stage: [MRLCalculator(unit, period) for period in periods for unit in self.spike_target.units['good']]
                for stage, periods in self.all_periods.items()}


class LFPDataSelector:
    """A class with methods shared by LFPPeriod and LFPTrial that are used to return portions of their data."""

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

    def __init__(self, raw_data, index, period_type, lfp_animal):
        self.raw_data = raw_data
        self.identifier = index
        self.animal = lfp_animal
        self.processed_data = self.process_lfp(raw_data)
        self.mrl_data = self.processed_data[LFP_SAMPLING_RATE:-LFP_SAMPLING_RATE]
        self.period_type = period_type
        self.parent = lfp_animal
        self.evoked_value_calculator = EvokedValueCalculator(self)

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
        arg_set = FREQUENCY_ARGS(self.current_frequency_band)
        pickle_path = os.path.join(self.data_opts['data_path'], 'lfp', '_'.join(
            [self.animal.identifier, self.data_opts['brain_region']] + [str(arg) for arg in arg_set] +
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
        if self.data_opts.get('evoked') and self.period_type == 'tone':
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
            self.identifier = list(range(*self.freq_range))[index]
        self.period_type = self.parent.period_type
        self.unit = unit
        self.is_valid = self.parent.is_valid if hasattr(self.parent, 'is_valid') else True

    @property
    def data(self):
        return self.val

    @property
    def mean_data(self):
        return np.mean(self.val)


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
        self.parent = self.period.parent
        self.identifier = f"{period.identifier}_{unit.identifier}"
        self.spike_period = self.unit.periods[self.period_type][self.period.identifier]
        self.spikes = [int((spike + i) * LFP_SAMPLING_RATE) for i, trial in enumerate(self.spike_period.trials)
                       for spike in trial.spikes]
        self.num_events = len(self.spikes)
        self.evoked_value_calculator = EvokedValueCalculator(self)

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

    @property
    def equivalent_calculator(self):
        other_stage = 'tone' if self.period_type == 'pretone' else 'pretone'
        return [calc for calc in self.parent.all_mrl_calculators[other_stage] if calc.identifier == self.identifier][0]

    @property
    def is_valid(self):
        if self.data_opts.get('evoked') == 'relative':
            return self.num_events > 4 and self.equivalent_calculator.num_events > 4
        else:
            return self.num_events > 4

    def get_weights(self):
        return np.array(
            [1 if weight in self.spikes else float('nan') for weight in range(TONES_PER_PERIOD * LFP_SAMPLING_RATE)])

    def get_wavelet_phases(self, scale):

        cwt_matrix = cwt(self.period.mrl_data, morlet, [scale])
        # Since we're computing the CWT for only one scale, the result is at index 0.
        return np.angle(cwt_matrix[0, :])

    def get_phases(self):
        low = self.freq_range[0] + .05
        high = self.freq_range[1]
        if self.data_opts.get('phase') == 'wavelet':
            if 'gamma' in self.current_frequency_band:
                frequencies = np.logspace(np.log10(low), np.log10(high), int((high - low) / 5))
            else:
                frequencies = np.linspace(low, high, int(high - low) + 1)
            scales = [get_wavelet_scale(f, LFP_SAMPLING_RATE) for f in frequencies]
            return np.array([self.get_wavelet_phases(s) for s in scales])
        else:
            if isinstance(self.current_frequency_band, type('str')):
                return compute_phase(bandpass_filter(self.period.mrl_data, low, high, LFP_SAMPLING_RATE))
            else:
                frequency_bands = [(f + .05, f + 1) for f in range(*self.freq_range)]
                return np.array([compute_phase(bandpass_filter(self.period.mrl_data, low, high, LFP_SAMPLING_RATE))
                                 for low, high in frequency_bands])

    @cache_method
    def get_angles(self):

        def adjust_angle(angle, weight):
            if np.isnan(weight):
                return np.nan
            return angle % (2 * np.pi)

        phases = self.get_phases().T
        weights = self.get_weights()

        # Apply the function to every element
        vfunc = np.vectorize(adjust_angle)

        if phases.ndim == 1:
            adjusted_phases = vfunc(phases, weights)
        else:
            # Expand the dimensions of weights to make it (60000, 1)
            weights_expanded = weights[:, np.newaxis]
            adjusted_phases = vfunc(phases, weights_expanded)

        # Filter out NaNs
        adjusted_phases = adjusted_phases[~np.isnan(adjusted_phases)]

        return adjusted_phases

    def get_angle_counts(self):
        n_bins = 36
        bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
        angles = self.get_angles()
        counts, _ = np.histogram(angles, bins=bin_edges)
        if self.data_opts.get('evoked'):
            counts = counts/len(angles)  # counts must be transformed to proportions for the subtraction to make sense
            if self.period_type == 'tone':
                counts -= self.equivalent_calculator.get_angle_counts()
        return counts

    @cache_method
    def get_mrl(self):
        w = self.get_weights()
        alpha = self.get_phases()
        dim = int(self.data_opts.get('phase') == 'wavelet' or not isinstance(self.current_frequency_band, type('str')))

        if w.ndim == 1 and alpha.ndim == 2:
            w = w[np.newaxis, :]

        # Handle NaN assignment based on the shape of alpha
        if alpha.ndim == 2:
            w[:, np.isnan(alpha).any(axis=0)] = np.nan
        else:
            w[np.isnan(alpha)] = np.nan

        if self.data_opts.get('mrl_func') == 'ppc':
            return circ_r2_unbiased(alpha, w, dim=dim)
        else:
            return compute_mrl(alpha, w, dim=dim)



