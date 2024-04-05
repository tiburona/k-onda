import os
import json
import pickle
import numpy as np

from scipy.signal import cwt, morlet, coherence
from collections import defaultdict
from copy import deepcopy


from data import Data
from context import experiment_context
from period_constructor import PeriodConstructor
from context import Subscriber
from matlab_interface import MatlabInterface
from math_functions import *
from utils import cache_method, get_ancestors, find_ancestor_attribute



class LFPData(Data):

    @property
    def frequency_bins(self):
        return self.get_frequency_bins(self.data)

    @property
    def time_bins(self):
        return self.get_time_bins(self.data)

    @property
    def freq_range(self):
        frozen_freq_range = find_ancestor_attribute(self, 'any', 'frozen_freq_range')
        if frozen_freq_range is not None:
            return frozen_freq_range
        exp = self.find_experiment()
        if isinstance(self.current_frequency_band, type('str')):
            return exp.frequency_bands[self.current_frequency_band]
        else:
            return self.current_frequency_band
        
    @property
    def current_coherence_region_set(self):
        self.data_opts.get('coherence_region_set')

    @property
    def lfp_root(self):
        if hasattr(self, '_lfp_root'):
            return self._lfp_root
        elif hasattr(self, 'parent'):
            return self.parent.lfp_root
        else:
            return None

    @property
    def hierarchy(self):
        return {'experiment': 0, 'group': 1, 'animal': 2, 'period': 3, 'mrl_calculator': 3, 'event': 4, 
                'frequency_bin': 5, 'time_bin': 6}

    @cache_method
    def get_mrl(self):
        axis = 0 if not self.data_opts.get('collapse_matrix') else None
        return self.get_average('get_mrl', stop_at='mrl_calculator', axis=axis)


    @cache_method
    def get_power(self):
        return self.get_average('get_power', stop_at='event')
    
    @cache_method
    def get_mrl(self):
        return self.get_average('get_mrl', stop_at='mrl_calculator', axis=None)

    @cache_method
    def get_coherence(self):
        return self.get_average('get_coherence', stop_at='coherence_calculator')

    def get_time_bins(self, data):
        tbs = []
        if len(data.shape) > 1:
            for i, data_point in enumerate(range(data.shape[1])):
                column = data[:, i]
                tb = TimeBin(i, column, self)
                tbs.append(tb)
            return tbs
        else:
            return [TimeBin(i, data_point, self) for i, data_point in enumerate(data)]

    def get_frequency_bins(self, data):
        return [FrequencyBin(i, data_point, self) for i, data_point in enumerate(data)]

    def load(self, calc_name, other_identifiers):
        store = self.data_opts.get('store', 'pkl')
        lfp_dir = os.path.join(self.lfp_root, 'lfp')
        store_dir = os.path.join(lfp_dir, f"{calc_name}_{store}s")
        for p in [lfp_dir, store_dir]:
            if not os.path.exists(p):
                os.mkdir(p)
        store_path = os.path.join(store_dir, '_'.join(other_identifiers) + f".{store}")
        if os.path.exists(store_path) and not self.data_opts.get('force_recalc'):
            with open(store_path, 'rb') as f:
                if store == 'pkl':
                    return_val = pickle.load(f)
                else:
                    return_val = json.load(f)
                return True, return_val, store_path
        else:
            return False, None, store_path

    def save(self, result, store_path):
        store = self.data_opts.get('store', 'pkl')
        mode = 'wb' if store == 'pkl' else 'w'
        with open(store_path, mode) as f:
            if store == 'pkl':
                return pickle.dump(result, f)
            else:
                result_str = json.dumps([arr.tolist() for arr in result])
                f.write(result_str)


class LFPExperiment(LFPData, Subscriber):
    name = 'experiment'

    def __init__(self, experiment, info, raw_lfp):
        self.experiment = experiment
        self.subscribe(experiment_context)
        self._lfp_root = info['lfp_root']
        self._sampling_rate = info['lfp_sampling_rate']
        self.set_global_sampling_rate(self._sampling_rate)
        self.frequency_bands = info['frequency_bands']
        self.lost_signal = info['lost_signal']  # TODO: this is going to need to get more granular at some point but I haven't yet seen an analysis with multiple mtcsg args
        self.all_animals = [LFPAnimal(animal, raw_lfp[animal.identifier], self._sampling_rate)
                            for animal in self.experiment.all_animals]
        self.groups = [LFPGroup(group, self) for group in self.experiment.groups]
        self._children = self.groups
        self.all_groups = self.groups
        self.last_brain_region = None
        self.last_neuron_type = 'uninitialized'
        self.last_period_type = 'uninitialized'
        self.last_frequency_band = None
        self.selected_animals = None
        self.event_validation = {}

    @property
    def all_periods(self):
        periods = []
        for animal in [animal for animal in self.all_animals if self.in_selected_animals(animal)]:
            [periods.extend(animal_periods) for animal_periods in animal.periods.values()]
        return periods

    @property
    def all_mrl_calculators(self):
        mrl_calculators = []
        for animal in [animal for animal in self.all_animals if self.in_selected_animals(animal)]:
            [mrl_calculators.extend(animal_periods) for animal_periods in animal.mrl_calculators.values()]
        return mrl_calculators

    @property
    def all_events(self):
        if self.data_type == 'mrl':
            raise ValueError("You can't extract events from MRL data")
        return [event for period in self.all_periods for event in period.events]

    def update(self, _):
        if self.data_class == 'lfp': 
            [animal.update_if_necessary() for animal in self.all_animals]

    def validate_events(self, data_opts):  

        regions = data_opts['brain_regions']
        
        for region in regions:
            self.event_validation[region] = {}
            data_opts['brain_region'] = region
            self.data_opts = data_opts
            animals = [animal for animal in self.all_animals if animal.is_valid 
                       and region in animal.raw_lfp]
            for animal in animals:
                standards = {period_type: animal.get_median(
                    stop_at='event', extend_by=('frequency', 'time'), 
                    select_by=(('period', 'period_type', period_type),)) 
                    for period_type in animal.period_info}
               
                def validate_event(event):
                    frequency_bins = event.get_frequency_bins(event.get_original_data())
                    for frequency in frequency_bins: 
                        standard = standards[event.period_type]
                        for time_bin in frequency.time_bins:
                            if time_bin.data > data_opts.get('threshold', 20) * standard:
                                print(f"{region} {animal.identifier} {event.period_type} "
                                f"{event.period.identifier} {event.identifier} invalid!")
                                return False
                    return True
                   
                for event in animal.all_events:
                    event.validator = validate_event(event)
                
                self.event_validation[region][animal.identifier] = animal.event_validity()


class LFPGroup(LFPData):
    name = 'group'

    def __init__(self, group, lfp_experiment):
        self.spike_target = group
        self.experiment = lfp_experiment
        self.identifier = self.spike_target.identifier
        self.parent = lfp_experiment
        self.animals = [animal for animal in self.experiment.all_animals if animal.condition == self.identifier]
        self._children = self.animals
        for animal in self._children:
            animal.parent = self
            animal.group = self

    @property
    def mrl_calculators(self):
        return [mrl_calc for animal in self.children for mrl_calc in animal.mrl_calculators if mrl_calc.validator]

    @property
    def grandchildren_scatter(self):
        if self.data_type != 'mrl':
            raise NotImplementedError("Grandchildren Scatter is currently only implemented for MRL")
        
        unit_points = []
        for animal in self.children:
            # Initialize a dictionary to hold lists of data points for each unit identifier
            unit_data_map = {}
            for mrl_calc in animal.children:
                identifier = mrl_calc.unit.identifier
                if identifier not in unit_data_map:
                    unit_data_map[identifier] = []
                unit_data_map[identifier].append(mrl_calc.data)
            
            # For each unit in animal.spike_target.children, compute the mean of collected data points
            for unit in animal.spike_target.children:
                if unit.identifier in unit_data_map:
                    # Compute the mean of all data points for this unit
                    data_points = unit_data_map[unit.identifier]
                    mean_data_point = np.nanmean(data_points)
                    unit_points.append(mean_data_point)
        
        return unit_points


    @property
    def data_by_period(self):
        if self.data_type != 'mrl':
            raise NotImplementedError("Data by period is currently only implemented for MRL")
        data_by_period = []
        for i in range(5):
            data_by_period.append(
                np.mean(
                    [mrl_calc.data for animal in self.children for mrl_calc in animal.children
                     if mrl_calc.period.identifier == i], axis=0)
            )
        return np.array(data_by_period)

    def update_children(self):
        self._children = [animal for animal in self.experiment.all_animals if animal.condition == self.identifier]

    def get_angle_counts(self):
        for calc in self.mrl_calculators:
            if any(np.isnan(calc.get_angle_counts())):
                print(f"{calc.parent.identifier} {calc.identifier}")
        counts = np.sum(np.array([calc.get_angle_counts() for calc in self.mrl_calculators]), axis=0)
        return counts


class LFPAnimal(LFPData, PeriodConstructor):
    """An animal in the experiment. Processes the raw LFP data and divides it into periods."""

    name = 'animal'

    def __init__(self, animal, raw_lfp, sampling_rate, is_mirror=False):
        self.spike_target = animal
        self.raw_lfp = raw_lfp
        self._sampling_rate = sampling_rate
        self.period_class = LFPPeriod
        self.periods = defaultdict(list)
        self.mrl_calculators = defaultdict(list)
        self.coherence_calculators = defaultdict(list)
        self.parent = None
        self.group = None
        self._processed_lfp = {}
        self.last_brain_region = None
        self.last_neuron_type = 'uninitialized'
        self.last_period_type = 'uninitialized'
        self.last_frequency_band = None
        self.last_coherence_region_set = None
        self.is_mirror = is_mirror
        self._mirror = None
        self.frozen_freq_range = None
        self.frozen_periods = None

    def __getattr__(self, name):
        prop = getattr(type(self), name, None)
        if isinstance(prop, property):
            return prop.fget(self)
        return getattr(self.spike_target, name)

    @property
    def _children(self):
        if self.data_type == 'mrl':
            children = self.mrl_calculators
        elif self.data_type == 'power':
            children = self.periods
        elif self.data_type == 'coherence':
            children = self.coherence_calculators
        else:
            raise ValueError("Unknown data type")
        if self.data_opts.get('spontaneous'):
            children = children['spontaneous']
        else:
            children = self.filter_by_selected_periods(children)
        if self.data_type == 'mrl':
            children = [calc for calc in children if calc.unit in self.spike_target.children]
        return children

    def update_if_necessary(self):
        if self.data_type == 'coherence':
            regions = self.data_opts.get('coherence_region_set').split('_')
        else:
            regions = [self.data_opts.get('brain_region')]
        if any([region not in self.raw_lfp for region in regions]):
            return
        old_and_new = [(self.last_brain_region, self.current_brain_region),
                       (self.last_frequency_band, self.current_frequency_band),
                       (self.last_neuron_type, self.selected_neuron_type),
                       (self.last_period_type, self.selected_period_type),
                       (self.last_coherence_region_set, self.current_coherence_region_set)]
        if any([old != new for old, new in old_and_new]):
            self.update_children()
            self.last_brain_region = self.current_brain_region
            self.last_frequency_band = self.current_frequency_band
            self.last_neuron_type = self.selected_neuron_type
            self.last_period_type = self.selected_period_type
            self.last_coherence_region_set = self.current_coherence_region_set

    @property
    def processed_lfp(self):
        if self.current_brain_region not in self._processed_lfp:
            self.process_lfp()
        return self._processed_lfp
    
    @property
    def all_periods(self):
        return [p for period_list in self.periods.values() for p in period_list]

    @property
    def all_events(self):
        if self.data_type == 'mrl':
            raise ValueError("You can't extract events from MRL data")
        return [event for period in self.all_periods for event in period.events]

    def update_children(self):
        print(self.identifier, "started updating children")
        self.prepare_periods()
        if 'mrl' in self.data_type:
            self.prepare_mrl_calculators()
        if 'coherence' in self.data_type:
            self.prepare_coherence_calculators()
        print(self.identifier, "finished updating children")
       
    # def make_mirror(self):
    #     # This method permits calculating values to determine which data points might be noise with one set
    #     # frequencies when the actual calculation of interest uses another set. The mirror holds the memory of the first
    #     # calculation.
    #     mirror =  LFPAnimal(self.spike_target, self.raw_lfp, self.sampling_rate, is_mirror=True)
    #     mirror.parent = self.parent
    #     mirror.group = self.group
    #     mirror.update_children()
    #     return mirror
    
    def event_validity(self):
        return {period_type: [period.event_validity() for period in self.periods[period_type]] 
                for period_type in self.periods}

    def process_lfp(self):
        
        for brain_region in self.raw_lfp:
            data = self.raw_lfp[brain_region]/4
            filter = self.data_opts.get('filter', 'filtfilt')
            if filter == 'filtfilt':
                filtered = filter_60_hz(data, self.sampling_rate)
            elif filter == 'spectrum_estimation':
                ids = [self.identifier, brain_region]
                saved_calc_exists, filtered, pickle_path = self.load('filter', ids)
                if not saved_calc_exists:
                    ml = MatlabInterface(self.data_opts['matlab_configuration'])
                    filtered = ml.filter(data)
                    self.save(filtered, pickle_path)
                filtered = np.squeeze(np.array(filtered))
            else:
                raise ValueError("Unknown filter")
            normed = divide_by_rms(filtered)
            self._processed_lfp[brain_region] = normed

    def prepare_mrl_calculators(self):
        if self.data_opts.get('spontaneous'):
            mrl_calculators = {'spontaneous': [SpontaneousMRLCalculator(unit, self)
                                               for unit in self.spike_target.units['good']]}
        else:
            mrl_calculators = {period_type: [PeriodMRLCalculator(unit, period=period) 
                                             for unit in self.spike_target.units['good'] 
                                             for period in periods]
                               for period_type, periods in self.periods.items()}
        self.mrl_calculators = mrl_calculators

    def prepare_coherence_calculators(self):
        brain_region_1, brain_region_2 = self.data_opts.get('coherence_region_set').split('_')
        self.coherence_calculators = {
            period_type: [CoherenceCalculator(period, brain_region_1, brain_region_2) 
                          for period in self.periods[period_type]] 
                          for period_type in self.periods
                          }
        print("prepared coherence calculators")

class LFPDataSelector:
    """A class with methods shared by LFPPeriod and LFPEvent that are used to return portions of their data."""

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
        tolerance = .2  # todo: this might change with different mtcsg args
        indices = np.where(self.spectrogram[1] - tolerance <= self.freq_range[0])
        ind1 = indices[0][-1] if indices[0].size > 0 else None  # last index that's <= start of the freq range
        ind2 = np.argmax(self.spectrogram[1] > self.freq_range[1] + tolerance)  # first index > end of freq range
        val_to_return = self.spectrogram[0][ind1:ind2, :]
        np.array(val_to_return)
        return val_to_return
        
    @cache_method
    def trimmed_spectrogram(self):
        return self.sliced_spectrogram[:, 75:-75]  # TODO: 75 should be a function of period length and mtscg args

    @property
    def sliced_spectrogram(self):
        return self.slice_spectrogram()


class LFPPeriod(LFPData, PeriodConstructor, LFPDataSelector):
    """A period in the experiment. Preprocesses data, initiates calls to Matlab to get the cross-spectrogram, and
    generates LFPEvents. Inherits from LFPSelector to be able to return portions of its data."""

    name = 'period'

    def __init__(self, lfp_animal, i, period_type, period_info, onset, events=None, 
                 target_period=None, is_relative=False):
        LFPDataSelector.__init__(self)
        self.animal = lfp_animal
        self.parent = lfp_animal
        self.identifier = i
        self.period_type = period_type
        self.period_info = period_info
        self.onset = onset - 1
        self.target_period = target_period
        self._is_relative = is_relative
        if events is not None:
            self.event_starts = events - 1
        else:
            self.event_starts = np.array([])
        self.convolution_padding = period_info['lfp_padding']
        self.duration = period_info.get('duration')
        self.event_duration = period_info.get('event_duration')
        if self.event_duration is None:
            self.event_duration = target_period.event_duration
        self.reference_period_type = period_info.get('reference_period_type')
        start_pad_in_samples, end_pad_in_samples = np.array(self.convolution_padding) * self.sampling_rate
        self.duration_in_samples = int(self.duration * self.sampling_rate)
        self.start = int(self.onset)
        self.stop = self.start + self.duration_in_samples
        self.pad_start = self.start - start_pad_in_samples
        self.pad_stop = self.stop + end_pad_in_samples
        self._spectrogram = None
        self.last_brain_region = None
        self.experiment = self.animal.group.experiment
        self._events = None

    @property
    def raw_data(self):
        return self.get_data_from_animal_dict(self.raw_lfp, self.pad_start, self.pad_stop)

    @property
    def processed_data(self):
        return self.get_data_from_animal_dict(self.animal.processed_lfp, 
                                              self.pad_start, self.pad_stop)
        
    @property
    def unpadded_data(self):
        return self.get_data_from_animal_dict(self.animal.processed_lfp, self.start, self.stop)
        
    def get_data_from_animal_dict(self, data_source, start, stop):
        if self.current_brain_region:
            return data_source[self.current_brain_region][start:stop]
        else:
            return {brain_region: data_source[brain_region][start:stop] 
                    for brain_region in data_source}

    @property
    def _children(self):
        if self.data_opts.get('validate_events'):
            return [event for event in self.events if event.validator]
        else:
            return self.events

    @property
    def events(self):
        return self.get_events()

    @property
    def spectrogram(self):
        if self._spectrogram is None or self.current_brain_region != self.last_brain_region:
            self._spectrogram = self.calc_cross_spectrogram()
            self.last_brain_region = self.current_brain_region
        return self._spectrogram

    @property
    def extended_data(self):
        data = self.events[0].data
        for event in self:
            data = np.concatenate((data, event.data), axis=1)
        return data

    def lost_signal(self):
        return self.animal.parent.experiment.lost_signal

    def get_events(self):
        if self._events:
            return self._events
        true_beginning = self.convolution_padding[0] - self.lost_signal()/2
        pre_stim, post_stim = (self.data_opts['events'][self.period_type][opt] for opt in ['pre_stim', 'post_stim'])
        time_bins = np.array(self.spectrogram[2])
        events = []
        epsilon = 1e-6  # a small offset to avoid floating-point inting issues

        for i, event_start in enumerate(self.event_starts):
            # get normed data for the event in samples
            start = int(event_start - self.onset)
            stop = int(start + self.event_duration*self.sampling_rate)
            normed_data = self.unpadded_data[start:stop]

            # get time points where the event will fall in the spectrogram in seconds
            spect_start = start/self.sampling_rate + true_beginning - pre_stim
            spect_end = spect_start + pre_stim + post_stim
            num_points = int(np.ceil((spect_end - spect_start) / .01 - epsilon))  # TODO: the .01s in here depend on the mtcsg args
            event_times = np.linspace(spect_start, spect_start + (num_points * .01), num_points, endpoint=False)
            event_times = event_times[event_times < spect_end]
            # a binary mask that is True when a time bin in the spectrogram belongs to this event
            mask = (np.abs(time_bins[:, None] - event_times) <= epsilon).any(axis=1)
            events.append(LFPEvent(i, event_times, normed_data, mask, self))
        
        self._events = events
        return events

    def calc_cross_spectrogram(self):
        arg_set = self.data_opts['power_arg_set']
        pickle_args = [self.animal.identifier, self.data_opts['brain_region']] + [str(arg) for arg in arg_set] + \
                      [self.period_type, str(self.identifier)]
        saved_calc_exists, result, pickle_path = self.load('spectrogram', pickle_args)
        if not saved_calc_exists:
            ml = MatlabInterface(self.data_opts['matlab_configuration'])
            result = ml.mtcsg(self.processed_data, *arg_set)
            self.save(result, pickle_path)
        return [np.array(arr) for arr in result]

    @property
    def power_deviations(self):
        return self.get_power_deviations()

    @cache_method
    def get_power_deviations(self, moving_window=.15):
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
    
        
    def equivalent_period(self, animal):
        return animal.periods[self.period_type][self.identifier]
    
    def event_validity(self):
        return [event.validator for event in self.get_events()]


class LFPEvent(LFPData, LFPDataSelector):
    name = 'event'

    def __init__(self, id, event_times, normed_data, mask, period):
        LFPDataSelector.__init__(self)
        self.identifier = id
        self.event_times = event_times
        self.mask = mask
        if sum(mask) == 300:
            raise ValueError
        self.normed_data = normed_data
        self.period = period
        self.parent = period
        self.period_type = self.parent.period_type
        self.spectrogram = self.parent.spectrogram
        self._children = None
        self.base_node = True

    @cache_method
    def _get_power(self):
        if sum(self.mask) == 300:
            raise ValueError
        return np.array(self.sliced_spectrogram)[:, self.mask]

    @cache_method
    def get_power(self):
        return self.refer(self._get_power())

    def get_original_data(self):
        return self._get_power()
    
    def equivalent_event(self, animal):
        return animal.periods[self.period_type][self.period.identifier][self.identifier]


class FrequencyBin(LFPData):
    """A FrequencyBin contains a slice of cross-spectrogram or mrl calculation at the smallest available frequency
    resolution."""

    name = 'frequency_bin'

    def __init__(self, index, val, parent, unit=None):
        self.parent = parent
        self.val = val
        self.identifier = index
        if self.parent.name == 'period':
            self.frequency = self.parent.spectrogram[1][index]
        elif hasattr(self.parent, 'period'):
            self.frequency = self.parent.period.spectrogram[1][index]
        self.unit = unit
        self.validator = self.parent.validator if hasattr(self.parent, 'validator') else True

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
        self.identifier = i
        self.data = data
        self.mean_data = np.mean(self.data)
        self.ancestors = get_ancestors(self)
        self.position = self.get_position_in_period_time_series()
        period_ancestor = [ancestor for ancestor in self.ancestors if ancestor.name == 'period'] # TODO if there isn't yet a util to deal with this more elegantly write one
        if len(period_ancestor):
            self.period = period_ancestor[0]
        else:
            self.period = None
        self.period_type = self.period.period_type

    @property
    def power_deviation(self):
        return self.period.power_deviations[self.position]

    def get_position_in_period_time_series(self):
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
        self._children = None
        self.base_node = True
        self.neuron_quality = unit.quality

    @property
    def mean_over_frequency(self):
        return np.mean(self.data, axis=0)

    @property
    def frequency_bins(self):
        return [FrequencyBin(i, data, self, unit=self.unit) for i, data in enumerate(self.data)]

    def get_wavelet_phases(self, scale):
        cwt_matrix = cwt(self.period.unpadded_data, morlet, [scale])
        # Since we're computing the CWT for only one scale, the result is at index 0.
        return np.angle(cwt_matrix[0, :])

    def get_phases(self):
        low = self.freq_range[0] + .05
        high = self.freq_range[1]
        if self.data_opts.get('phase') == 'wavelet':
            if 'gamma' in self.current_frequency_band: # TODO need to fix this since frequency band can currently be a range of numbers
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
                return np.array([compute_phase(bandpass_filter(self.period.unpadded_data, low, high, self.sampling_rate))
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

    def get_angle_counts(self): # TODO: give this a data type and make it use data
        n_bins = 36
        bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
        angles = self.get_angles()
        counts, _ = np.histogram(angles, bins=bin_edges)
        if self.data_opts.get('evoked'):
            counts = counts/len(angles)  # counts must be transformed to proportions for the subtraction to make sense
            if self.period_type == 'tone':  # TODO: generalize this
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
            return self.refer(circ_r2_unbiased(alpha, w, dim=dim))
        else:
            return self.refer(compute_mrl(alpha, w, dim=dim))


class PeriodMRLCalculator(MRLCalculator):

    def __init__(self, unit, period):
        super().__init__(unit)
        self.period = period
        self.period_type = period.period_type
        self.mrl_data = period.unpadded_data
        self.duration = period.duration
        self.identifier = f"{self.period.identifier}_{self.unit.identifier}"
        self.parent = self.period.parent
        self.spike_period = self.unit.periods[self.period_type][self.period.identifier]
        self.spikes = [int((spike + i) * self.sampling_rate) for i, event in enumerate(self.spike_period.events)
                       for spike in event.spikes]
        #self.
        self.num_events = len(self.spikes)

    @property
    def ancestors(self):
        return [self] + [self.unit] + [self.period] + self.parent.ancestors

    @property
    def equivalent_calculator(self):
        other_stage = self.spike_period.reference_period_type
        return [calc for calc in self.parent.mrl_calculators[other_stage] if calc.identifier == self.identifier][0]

    @property
    def validator(self):
        if self.data_opts.get('evoked'):
            return self.num_events > 4 and self.equivalent_calculator.num_events > 4
        else:
            return self.num_events > 4

    def translate_spikes_to_lfp_events(self, spikes):
        pre_stim = self.data_opts['events'][self.period.period_type]['pre_stim'] * self.sampling_rate
        events = np.array(self.period.event_starts) - self.period.event_starts[0] - pre_stim
        indices = {}
        for spike in spikes:
            # Find the index of the event the spike belongs to
            index = np.argmax(events > spike)
            if events[index] > spike:
                indices[spike] = index - 1
            else:
                indices[spike] = len(events) - 1
        return indices
    
    def get_weights(self):
        weight_range = range(self.duration * self.sampling_rate)
        if not self.data_opts.get('validate_events'):
            weights = [1 if weight in self.spikes else float('nan') for weight in weight_range]
        else:
            indices = self.translate_spikes_to_lfp_events(self.spikes)
            self.update_data_opts([(['data_type'], 'power')])
            weight_validity = {spike: self.period.event_validity[event] 
                               for spike, event in indices.items()}
            self.update_data_opts([(['data_type'], 'mrl')])
            weights = np.array([1 if weight_validity.get(w) else float('nan') for w in weight_range])
        return np.array(weights)
            

class SpontaneousMRLCalculator(MRLCalculator):
    def __init__(self, unit, animal):
        super().__init__(unit)
        self.animal = animal
        self.parent = self.animal
        self.period_type = self.identifier = 'spontaneous'
        self.spikes = self.unit.get_spontaneous_firing()
        self.num_events = len(self.spikes)
        self.raw = self.animal.raw_lfp[self.brain_region]
        if isinstance(self.data_opts['spontaneous'], tuple):
            self.start, self.end = np.array(self.data_opts['spontaneous']) * self.sampling_rate
        else:
            self.end = self.animal.earliest_period.onset
            self.start = self.end - self.data_opts['spontaneous'] * self.sampling_rate
        self.mrl_data = self.raw[self.start:self.end]

    def get_weights(self):
        return np.array([1 if weight in self.spikes else float('nan') for weight in range(self.start, self.end)])

    @property
    def validator(self):
        return self.num_events > 4
    

class CoherenceCalculator(LFPData):

    name = 'coherence_calculator'

    def __init__(self, period, brain_region_1, brain_region_2):
        self.period = period
        self.period_type = period.period_type
        self.region_1 = brain_region_1
        self.region_2 = brain_region_2
        self.identifier = f"{brain_region_1}_{brain_region_2}_{self.period.identifier}"
        self.region_1_data = self.period.animal.processed_lfp[self.region_1][
            self.period.start:self.period.stop]
        self.region_2_data = self.period.animal.processed_lfp[self.region_2][
            self.period.start:self.period.stop]
        self.base_node = True
        self._children = None
        self.parent = self.period.parent

    def joint_event_validity(self):
        ev1, ev2 = [self.get_event_validity(region) for region in (self.region_1, self.region_2)]
        return {i: ev1[i] and ev2[i] for i in ev1}

    def get_event_validity(self, region):
        validation = self.period.animal.group.experiment.event_validation
        animal_validation_dict = validation[region][self.period.animal.identifier]
        return {i: is_valid for i, is_valid in 
                enumerate(animal_validation_dict[self.period_type][self.period.identifier])}

    def get_coherence(self):
        if not self.data_opts.get('validate_events'):
            return self.calc_coherence(self.region_1_data, self.region_2_data)
        else:
            valid_sets_1 = self.divide_data_into_valid_sets(self.region_1_data)
            valid_sets_2 = self.divide_data_into_valid_sets(self.region_2_data)
            valid_sets = zip(valid_sets_1, valid_sets_2)
            data_len = sum([len(vs) for vs in valid_sets_1])
            weighted_coherences = [self.calc_coherence(data_1, data_2)*len(data_1)/data_len 
                                   for data_1, data_2 in valid_sets]
            return sum(weighted_coherences)
        
    def calc_coherence(self, data_1, data_2):

        nperseg = 2000  
        noverlap = int(nperseg/2)
        window = 'hann'  # Window type
        f, Cxy = coherence(data_1, data_2, fs=self.sampling_rate, 
                           window=window, nperseg=nperseg, noverlap=noverlap)
        low, high = self.freq_range
        mask = (f >= low) & (f <= high)
        Cxy_band = Cxy[mask]
        return Cxy_band
    
    def divide_data_into_valid_sets(self, region_data):
        valid_sets = []
        current_set = []
        event_duration = self.sampling_rate * self.period.event_duration
    
        for i in range(0, len(region_data), event_duration):
            if self.joint_event_validity()[i // event_duration]:  
                start = i
                current_set.extend(region_data[start:start+event_duration])
            else:
                if current_set:  
                    valid_sets.append(current_set)
                    current_set = []
    
        if current_set:  
            valid_sets.append(current_set)
    
        return valid_sets
        
   



