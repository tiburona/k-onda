import os
import pickle

from scipy.signal import cwt, morlet
from collections import defaultdict


from data import Data
from context import experiment_context
from block_constructor import BlockConstructor
from context import Subscriber
from matlab_interface import MatlabInterface
from math_functions import *
from utils import cache_method, get_ancestors


FREQUENCY_BANDS = dict(delta=(0, 4), theta_1=(4, 8), theta_2=(4, 12), delta_theta=(0, 12), gamma=(20, 55),
                       hgamma=(70, 120))
LO_FREQ_ARGS = (2048, 2000, 1000, 980, 2)
FREQUENCY_ARGS = {fb: LO_FREQ_ARGS for fb in ['delta', 'theta_1', 'theta_2', 'delta_theta', 'gamma', 'hgamma']}
# TODO: gamma and hgamma don't really belong there, find out what their args should be.


class LFPData(Data):

    @property
    def brain_region(self):
        return self.data_opts.get('brain_region')

    @property
    def frequency_bins(self):
        return [FrequencyBin(i, data_point, self) for i, data_point in enumerate(self.data)]

    @property
    def time_bins(self):
        return self.get_time_bins()

    @property
    def freq_range(self):
        if isinstance(self.current_frequency_band, type('str')):
            return FREQUENCY_BANDS[self.current_frequency_band]
        else:
            return self.current_frequency_band

    def get_mrl(self):
        axis = 0 if not self.data_opts.get('collapse_matrix') else None
        return self.get_average('get_mrl', stop_at='mrl_calculator', axis=axis)

    def get_time_bins(self):
        tbs = []
        if len(self.data.shape) > 1:
            for i, data_point in enumerate(range(self.data.shape[1])):
                column = self.data[:, i]
                tb = TimeBin(i, column, self)
                tbs.append(tb)
            return tbs
        else:
            return [TimeBin(i, data_point, self) for i, data_point in enumerate(self.data)]


class LFPExperiment(LFPData, Subscriber):
    name = 'experiment'

    def __init__(self, experiment, info, raw_lfp):
        self.experiment = experiment
        self.subscribe(experiment_context)
        self._sampling_rate = info['lfp_sampling_rate']
        self.all_animals = [LFPAnimal(animal, raw_lfp[animal.identifier], self._sampling_rate)
                            for animal in self.experiment.all_animals]
        self.groups = [LFPGroup(group, self) for group in self.experiment.groups]
        self.all_groups = self.groups
        self.last_brain_region = None
        self.last_neuron_type = 'uninitialized'
        self.last_block_type = 'uninitialized'
        self.selected_animals = None

    @property
    def all_blocks(self):
        blocks = []
        for animal in self.all_animals:
            [blocks.extend(animal_blocks) for animal_blocks in animal.blocks.values()]
        return blocks

    @property
    def all_mrl_calculators(self):
        mrl_calculators = []
        for animal in self.all_animals:
            [mrl_calculators.extend(animal_blocks) for animal_blocks in animal.mrl_calculators.values()]
        return mrl_calculators

    @property
    def all_events(self):
        if self.data_type == 'mrl':
            raise ValueError("You can't extract events from MRL data")
        events = [event for block in self.all_blocks for event in block.events]
        return events

    def update(self, name):
        if name == 'data':
            if self.data_class == 'lfp' and self.last_brain_region != self.data_opts.get('brain_region'):
                [animal.update_children() for animal in self.all_animals]
                self.last_brain_region = self.data_opts.get('brain_region')
            if self.data_opts.get('selected_animals') != self.selected_animals:
                [group.update_children() for group in self.groups]
                self.selected_animals = self.data_opts.get('selected_animals')
        if name == 'block_type':
            if self.selected_block_type != self.last_block_type:
                [animal.update_children() for animal in self.all_animals]
                self.last_block_type = self.selected_block_type
        if name == 'neuron_type':
            if self.selected_neuron_type != self.last_neuron_type:
                [animal.update_children() for animal in self.all_animals]
                self.last_neuron_type = self.selected_neuron_type


class LFPGroup(LFPData):
    name = 'group'

    def __init__(self, group, lfp_experiment):
        self.spike_target = group
        self.experiment = lfp_experiment
        self.identifier = self.spike_target.identifier
        self.parent = lfp_experiment
        self.children = [animal for animal in self.experiment.all_animals if animal.condition == self.identifier]
        for animal in self.children:
            animal.parent = self

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
                    unit_points.append(np.nanmean([mrl_calc.data for mrl_calc in animal.children
                                                   if mrl_calc.unit.identifier == unit.identifier]))
            return unit_points

    @property
    def data_by_block(self):
        if self.data_type != 'mrl':  # TODO: What? What does grandchildren scatter have to do with this?
            raise NotImplementedError("Grandchildren Scatter is currently only implemented for MRL")
        data_by_block = []
        for i in range(5):
            data_by_block.append(
                np.mean(
                    [mrl_calc.data for animal in self.children for mrl_calc in animal.children
                     if mrl_calc.block.identifier == i], axis=0)
            )
        return np.array(data_by_block)

    def update_children(self):
        self.children = [animal for animal in self.experiment.all_animals if animal.condition == self.identifier]
        if self.data_opts.get('selected_animals') is not None:
            self.children = [child for child in self.children if child.identifier in
                             self.data_opts.get('selected_animals')]

    def get_angle_counts(self):
        for calc in self.mrl_calculators:
            if any(np.isnan(calc.get_angle_counts())):
                print(f"{calc.parent.identifier} {calc.identifier}")
        counts = np.sum(np.array([calc.get_angle_counts() for calc in self.mrl_calculators]), axis=0)
        return counts

    def get_mrl(self):
        return self.get_average('get_mrl', stop_at='mrl_calculator', axis=None)


class LFPAnimal(LFPData, BlockConstructor):
    """An animal in the experiment. Processes the raw LFP data and divides it into blocks."""

    name = 'animal'

    def __init__(self, animal, raw_lfp, sampling_rate):
        self.spike_target = animal
        self.raw_lfp = raw_lfp
        self._sampling_rate = sampling_rate
        self.block_class = LFPBlock
        self.blocks = defaultdict(list)
        self.mrl_calculators = defaultdict(list)

    @property
    def children(self):
        children_type = self.mrl_calculators if self.data_type == 'mrl' else self.blocks
        if self.data_opts.get('spontaneous'):
            return children_type['spontaneous']
        elif self.selected_block_type is not None:
            return children_type[self.selected_block_type]
        else:
            return [child for key in children_type for child in children_type[key]]

    def __getattr__(self, name):
        # Check if 'name' is a property in the class
        prop = getattr(type(self), name, None)
        if isinstance(prop, property):
            # If it's a property, return its value using the property's fget
            return prop.fget(self)
        # Otherwise, delegate to spike_target
        return getattr(self.spike_target, name)

    @property
    def frequency_args(self):
        return set(tuple(args) for fb, args in FREQUENCY_ARGS.items() if fb in self.data_opts['fb'])

    def update_children(self):
        self.prepare_blocks()
        if 'mrl' in self.data_type:
            self.prepare_mrl_calculators()

    def prepare_mrl_calculators(self):
        if self.data_opts.get('spontaneous'):
            mrl_calculators = {'spontaneous': [SpontaneousMRLCalculator(unit, self)
                                               for unit in self.spike_target.units['good']]}
        else:
            mrl_calculators = {block_type: [BlockMRLCalculator(unit, block=block) for block in blocks
                                            for unit in self.spike_target.units['good']]
                               for block_type, blocks in self.all_blocks.items()}
        self.mrl_calculators = mrl_calculators

    @cache_method
    def get_power(self):
        return [block.get_power() for block in self.blocks]


class LFPDataSelector:
    """A class with methods shared by LFPBlock and LFPEvent that are used to return portions of their data."""

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

    @cache_method
    def slice_spectrogram(self):
        indices = np.where(self.spectrogram[1] <= self.freq_range[0])
        ind1 = indices[0][-1] if indices[0].size > 0 else None  # last index that's <= start of the freq range
        ind2 = np.argmax(self.spectrogram[1] > self.freq_range[1])  # first index >= end of freq range
        return self.spectrogram[0][ind1:ind2, :]

    @cache_method
    def trimmed_spectrogram(self):
        return self.sliced_spectrogram[:, 75:-75]  # TODO: 75 should be a function of block length and mtscg args

    @property
    def sliced_spectrogram(self):
        return self.slice_spectrogram()


class LFPBlock(LFPData, BlockConstructor, LFPDataSelector):
    """A block in the experiment. Preprocesses data, initiates calls to Matlab to get the cross-spectrogram, and
    generates LFPEvents. Inherits from LFPSelector to be able to return portions of its data."""

    name = 'block'

    def __init__(self, lfp_animal, i, block_type, block_info, onset, events=None, paired_block=None, is_reference=False):
        LFPDataSelector.__init__(self)
        self.animal = lfp_animal
        self.parent = lfp_animal
        self.identifier = i
        self.block_type = block_type
        self.event_starts = events if events is not None else []
        self.onset = int(onset * self.sampling_rate / self.animal.spike_target.sampling_rate)
        self.convolution_padding = block_info['lfp_padding']
        self.duration = block_info.get('duration')
        self.event_duration = block_info.get('event_duration')
        self.paired_block = paired_block
        self.is_reference = is_reference
        start = self.onset - (self.convolution_padding[0]) * self.sampling_rate
        stop = self.onset + (self.duration + self.convolution_padding[1]) * self.sampling_rate
        self.raw_data = self.animal.raw_lfp[self.brain_region][start:stop]
        self.processed_data = self.process_lfp(self.raw_data)
        self.mrl_data = self.processed_data[onset:onset + self.duration * self.sampling_rate]
        self.parent = lfp_animal
        self._spectrogram = None
        self.last_brain_region = None

    @property
    def events(self):
        return self.get_events()

    @property
    def spectrogram(self):
        if self._spectrogram is None or self.brain_region != self.last_brain_region:
            self._spectrogram = self.calc_cross_spectrogram()
            self.last_brain_region = self.brain_region
        return self._spectrogram

    def get_lost_signal(self):
        return self.block_info['lost_signal']

    def get_events(self):
        true_beginning = -self.convolution_padding + self.get_lost_signal()
        starts = np.arange(true_beginning, true_beginning + self.duration + .0001, int(self.event_duration))
        time_bins = np.array(self.spectrogram[2])
        events = []
        epsilon = 1e-6  # a small offset to avoid floating-point rounding issues
        for i, start in enumerate(starts):
            end = start + self.data_opts['post_stim']
            num_points = int(np.ceil((end - start) / .01 - epsilon))
            event_times = np.linspace(start, start + (num_points * .01), num_points, endpoint=False)
            event_times = event_times[event_times < end]
            mask = (np.abs(time_bins[:, None] - event_times) <= 1e-6).any(axis=1)
            events.append(LFPEvent(i, event_times, mask, self))
        return events

    @staticmethod
    def process_lfp(raw):
        filtered = filter_60_hz(raw, 2000)
        return divide_by_rms(filtered)

    def calc_cross_spectrogram(self):
        arg_set = FREQUENCY_ARGS[self.current_frequency_band]
        pickle_path = os.path.join(self.data_opts['data_path'], 'lfp', '_'.join(
            [self.animal.identifier, self.data_opts['brain_region']] + [str(arg) for arg in arg_set] +
            [self.block_type, str(self.identifier)]) + '.pkl')
        if os.path.exists(pickle_path) and not self.data_opts.get('force_recalc'):
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        ml = MatlabInterface()
        result = ml.mtcsg(self.processed_data, *FREQUENCY_ARGS[self.current_frequency_band])
        with open(pickle_path, 'wb') as f:
            pickle.dump(result, f)
        return result

    @cache_method
    def get_power(self):
        power = np.mean([event.data for event in self.get_events()], axis=0)
        if self.data_opts.get('evoked') and not self.reference:
            power -= self.paired_block.get_power()
        return power

    @property
    def power_deviations(self):
        return self.get_power_deviations()

    @cache_method
    def get_power_deviations(self, moving_window=.15):
        print(self.identifier, self.animal.identifier)
        parent_mean = self.parent.mean_data
        parent_sd = self.parent.sd_data
        moving_avgs = np.zeros(int(self.duration/.01))
        window_interval = int(moving_window/.01)  # TODO: bin size actually depends on args to mtcsg
        for i in range(self.trimmed_spectrogram().shape[1] - window_interval + 1):
            data = np.mean(self.trimmed_spectrogram()[:, i:i+window_interval])
            normalized_data = (data - parent_mean)/parent_sd
            for j in range(window_interval):
                if abs(normalized_data) > np.abs(moving_avgs[i + j]):
                    moving_avgs[i + j] = normalized_data
        return moving_avgs


class LFPEvent(LFPData, LFPDataSelector):
    name = 'event'

    def __init__(self, id, event_times, mask, parent):
        LFPDataSelector.__init__(self)
        self.identifier = id
        self.event_times = event_times
        self.mask = mask
        self.parent = parent
        self.block_type = self.parent.block_type
        self.spectrogram = self.parent.spectrogram

    @cache_method
    def get_power(self):
        return np.array(self.sliced_spectrogram)[:, self.mask]


class FrequencyBin(LFPData):
    """A FrequencyBin contains a slice of cross-spectrogram or mrl calculation at the smallest available frequency
    resolution."""

    name = 'frequency_bin'

    def __init__(self, index, val, parent, unit=None):
        self.parent = parent
        self.val = val
        if isinstance(parent, LFPBlock):
            self.identifier = self.block.spectrogram[1][index]
        else:  # parent is MRLCalculator  # TODO: this is really an approximation of what frequencies these actually were; make more exact
            self.identifier = list(range(*self.freq_range))[index]
        self.block_type = self.parent.block_type
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
        self.block_type = self.parent.block_type
        self.identifier = i
        self.data = data
        self.mean_data = np.mean(self.data)
        self.ancestors = get_ancestors(self)
        self.position = self.get_position_in_block_time_series()
        self.block = [ancestor for ancestor in self.ancestors if ancestor.name == 'block'][0]

    @property
    def power_deviation(self):
        return self.block.power_deviations[self.position]

    def get_position_in_block_time_series(self):
        if self.parent.name == 'event':
            self.parent.num_bins_per_event * self.parent.identifier + self.identifier
        else:
            return self.identifier


class MRLCalculator(LFPData):
    """Calculates the Mean Resultant Length of the vector that represents the phase of a frequency in the LFP data on
    the occasion the firing of a neuron. MRL """

    name = 'mrl_calculator'

    def __init__(self, unit):
        self.unit = unit

    @property
    def mean_over_frequency(self):
        return np.mean(self.data, axis=0)

    @property
    def frequency_bins(self):
        return [FrequencyBin(i, data, self, unit=self.unit) for i, data in enumerate(self.data)]

    def get_wavelet_phases(self, scale):
        cwt_matrix = cwt(self.block.mrl_data, morlet, [scale])
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
            scales = [get_wavelet_scale(f, self.sampling_rate) for f in frequencies]
            return np.array([self.get_wavelet_phases(s) for s in scales])
        else:
            if isinstance(self.current_frequency_band, type('str')):
                return compute_phase(bandpass_filter(self.mrl_data, low, high, self.sampling_rate))
            else:
                frequency_bands = [(f + .05, f + 1) for f in range(*self.freq_range)]
                return np.array([compute_phase(bandpass_filter(self.block.mrl_data, low, high, self.sampling_rate))
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
            if self.block_type == 'tone':
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


class BlockMRLCalculator(MRLCalculator):

    def __init__(self, unit, block):
        super().__init__(unit)
        self.block_type = block.block_type
        self.mrl_data = self.block.mrl_data
        self.identifier = f"{self.block.identifier}_{self.unit.identifier}"
        self.spike_block = self.unit.blocks[self.block_type][self.block.identifier]
        self.spikes = [int((spike + i) * self.sampling_rate) for i, event in enumerate(self.spike_block.events)
                       for spike in event.spikes]
        self.num_events = len(self.spikes)
        self.parent = self.block.parent

    @property
    def ancestors(self):
        return [self] + [self.unit] + [self.block] + self.parent.ancestors

    @property
    def equivalent_calculator(self):
        other_stage = self.spike_block.reference_block_type
        return [calc for calc in self.parent.mrl_calculators[other_stage] if calc.identifier == self.identifier][0]

    @property
    def is_valid(self):
        if self.data_opts.get('evoked') == 'relative':
            return self.num_events > 4 and self.equivalent_calculator.num_events > 4
        else:
            return self.num_events > 4

    def get_weights(self):
        return np.array(
            [1 if weight in self.spikes else float('nan') for weight in range(self.duration * self.sampling_rate)])


class SpontaneousMRLCalculator(MRLCalculator):
    def __init__(self, unit, animal):
        super().__init__(unit)
        self.animal = animal
        self.parent = self.animal
        self.block_type = self.identifier = 'spontaneous'
        self.spikes = self.unit.get_spontaneous_firing()
        self.num_events = len(self.spikes)
        self.raw = self.animal.raw_lfp[self.brain_region]
        if isinstance(self.data_opts['spontaneous'], tuple):
            self.start, self.end = np.array(self.data_opts['spontaneous']) * self.sampling_rate
        else:
            self.end = self.animal.earliest_block.onset
            self.start = self.end - self.data_opts['spontaneous'] * self.sampling_rate
        self.mrl_data = self.raw[self.start:self.end]

    def get_weights(self):
        return np.array([1 if weight in self.spikes else float('nan') for weight in range(self.start, self.end)])

    @property
    def is_valid(self):
        return self.num_events > 4
