from period_event import Period, Event
from lfp_methods import LFPMethods
import numpy as np
from matlab_interface import MatlabInterface


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

    def slice_spectrogram(self):
        tolerance = .2  # TODO: this might change with different mtcsg args
        indices = np.where(self.spectrogram[1] - tolerance <= self.freq_range[0])
        ind1 = indices[0][-1] if indices[0].size > 0 else None  # last index that's <= start of the freq range
        ind2 = np.argmax(self.spectrogram[1] > self.freq_range[1] + tolerance)  # first index > end of freq range
        val_to_return = self.spectrogram[0][ind1:ind2, :]
        return val_to_return

    @property
    def sliced_spectrogram(self):
        return self.slice_spectrogram()
    
    
class EventValidator:
    
    def get_event_validity(self, region):
        period = self if self.name == 'period' else self.period
        ev = period.animal.group.experiment.event_validation
        validity = ev[region][period.animal.identifier]
        return {i: valid for i, valid in enumerate(validity[self.period_type][period.identifier])}


class LFPPeriod(Period, LFPMethods, LFPDataSelector, EventValidator):

    def __init__(self, animal, index, period_type, period_info, onset, events=None, 
                 target_period=None, is_relative=False, experiment=None):
        super().__init__(index, period_type, period_info, onset, experiment=experiment, 
                         target_period=target_period, is_relative=is_relative, events=events)
        self.animal = animal
        self.parent = animal
        padding = self.calc_opts['lfp_padding']
        start_pad, end_pad = np.round(np.array(padding) * self.lfp_sampling_rate).astype(int)
        self.duration_in_samples = round(self.duration * self.lfp_sampling_rate)
        self.start = round(self.onset)
        self.stop = self.start + self.duration_in_samples
        self.pad_start = self.start - start_pad
        self.pad_stop = self.stop + end_pad
        self._spectrogram = None
        
    @property
    def padded_data(self):
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
    def spectrogram(self):
        if self._spectrogram is None:
            self._spectrogram = self.calc_cross_spectrogram()
        last_frequency = self.freq_range[1]
        index_of_last_frequency = np.where(self._spectrogram[1] > last_frequency)[0][0]
        self._spectrogram[0] = self._spectrogram[0][0:index_of_last_frequency, :]
        return self._spectrogram

    def get_events(self):
        padding, lost_signal, bin_size = self.fetch_opts(['lfp_padding', 'lost_signal', 'bin_size'])

        true_beginning = padding[0] - lost_signal[0]
        time_bins = np.array(self.spectrogram[2])
        events = []
        epsilon = 1e-6  # a small offset to avoid floating-point issues

        for i, event_start in enumerate(self.event_starts):

            # get time points where the event will fall in the spectrogram in seconds
            spect_start = round(
                (event_start - self.onset)/self.lfp_sampling_rate + true_beginning - self.pre_stim
                , 2)
            spect_end = round(spect_start + self.pre_stim + self.post_stim, 2)
            num_points = round(np.ceil((spect_end - spect_start) / bin_size - epsilon))  
            event_times = np.linspace(spect_start, spect_start + (num_points * bin_size), 
                                      num_points, endpoint=False)
            event_times = event_times[event_times < spect_end]

            # a binary mask that is True when a time bin in the spectrogram belongs to this event
            mask = (np.abs(time_bins[:, None] - event_times) <= epsilon).any(axis=1)

            events.append(LFPEvent(i, event_times, mask, self))
        
        self._events = events
        return events
    
    @property
    def extended_data(self):
        data = self.events[0].data
        for event in self.events[1:]:
            data = np.concatenate((data, event.data), axis=1)
        return data
    
    def calc_cross_spectrogram(self):
        power_arg_set = self.calc_opts['power_arg_set']
        arg_set = [[self.animal.identifier, self.calc_opts['brain_region']], 
                       [str(arg) for arg in power_arg_set], 
                       [self.period_type, str(self.identifier)], 
                       ['padding'], [str(pad) for pad in [self.calc_opts['lfp_padding']]]]
        pickle_args = [item for sublist in arg_set for item in sublist]
        saved_calc_exists, result, pickle_path = self.load('spectrogram', pickle_args)
        if not saved_calc_exists:
            ml = MatlabInterface(self.calc_opts['matlab_configuration'])
            result = ml.mtcsg(self.padded_data, *power_arg_set)
            self.save(result, pickle_path)
        return [np.array(arr) for arr in result]
    

class LFPEvent(Event, LFPMethods, LFPDataSelector):

    def __init__(self, identifier, event_times, mask, period):
        super().__init__(period, identifier)
        self.event_times = event_times
        self.mask = mask
        self.animal = period.animal
        self.period_type = self.parent.period_type
        self.spectrogram = self.parent.spectrogram

    @property
    def is_valid(self):        
        return self.animal.lfp_event_validity[self.current_brain_region][self.period_type][
            self.period.identifier][self.identifier]
    
    def get_power(self):
        return self.refer(np.array(self.sliced_spectrogram)[:, self.mask])
    
