from neo.rawio import BlackrockRawIO
import numpy as np

from k_onda.data.period_event import Period, Event
from k_onda.interfaces import MatlabInterface
from k_onda.utils import filter_60_hz, divide_by_rms, downsample, PrepMethods


class LFPPrepMethods(PrepMethods):

    def select_lfp_children(self):
        return self.select_children('lfp_periods')

    def lfp_prep(self):
        self.prep_data()
        self.prepare_periods()
    
    def prep_data(self):
        data_label = f"{self.current_brain_region}_lfp"
        if data_label in self.initialized:
            return
        else:
            self.process_lfp()
            self.initialized.append(data_label)

    def get_raw_lfp(self):
        file_path = self.construct_path('lfp')
        try:
            reader = BlackrockRawIO(filename=file_path, nsx_to_load=3)
        except OSError:
            return {}
        reader.parse_header()
        if all(k not in self.animal_info for k in ('lfp_electrodes', 'lfp_from_stereotrodes')):
            return {}
        data_to_return = {region: reader.nsx_datas[3][0][:, val]
                          for region, val in self.animal_info['lfp_electrodes'].items()}
        if self.animal_info.get('lfp_from_stereotrodes') is not None:
            data_to_return = self.get_lfp_from_stereotrodes(self, data_to_return, file_path)
        return data_to_return

    def get_lfp_from_stereotrodes(self, animal, data_to_return, file_path):
        lfp_from_stereotrodes_info = animal.animal_info['lfp_from_stereotrodes']
        nsx_num = lfp_from_stereotrodes_info['nsx_num']
        reader = self.load_blackrock_file(file_path, nsx_to_load=nsx_num)
        for region, region_data in lfp_from_stereotrodes_info['electrodes'].items():
            electrodes = region_data if isinstance(region_data, list) else region_data[animal.identifier]
            data = np.mean([reader.nsx_datas[nsx_num][0][:, electrode] for electrode in electrodes], axis=0)
            downsampled_data = downsample(data, self.experiment.exp_info['sampling_rate'], 
                                          self.experiment.exp_info['lfp_sampling_rate'])
            data_to_return[region] = downsampled_data
        return data_to_return

    @property
    def processed_lfp(self):
        if self.current_brain_region not in self._processed_lfp:
            self.process_lfp()
        return self._processed_lfp
    
    def process_lfp(self):
        
        raw_lfp = self.get_raw_lfp()

        for brain_region in raw_lfp:
            data = raw_lfp[brain_region]/4
            filter = self.calc_opts.get('remove_noise', 'filtfilt')
            if filter == 'filtfilt':
                filtered = filter_60_hz(data, self.lfp_sampling_rate)
            elif filter == 'spectrum_estimation':
                ids = [self.identifier, brain_region]
                saved_calc_exists, filtered, pickle_path = self.load('filter', ids)
                if not saved_calc_exists:
                    ml = MatlabInterface(self.calc_opts['matlab_configuration'])
                    filtered = ml.filter(data)
                    self.save(filtered, pickle_path)
                filtered = np.squeeze(np.array(filtered))
            else:
                raise ValueError("Unknown filter")
            normed = divide_by_rms(filtered)
            self._processed_lfp[brain_region] = normed

    def validate_events(self):
        if not self.include():
            return 
        region = self.current_brain_region

        saved_calc_exists, validity, pickle_path = self.load('validity', [region, self.identifier])
        if saved_calc_exists:
            self.lfp_event_validity[region] = validity
            return
        
        def validate_event(event, standard):
            for frequency in event.frequency_bins: 
                for time_bin in frequency.time_bins:
                    if time_bin.calc > self.calc_opts.get('threshold', 20) * standard:
                        print(f"{region} {self.identifier} {event.period_type} "
                        f"{event.period.identifier} {event.identifier} invalid!")
                        return False
            return True
            
        for period_type in self.period_info:
            self.selected_period_type = period_type
            standard = self.get_median(extend_by=('frequency', 'time'))
            self.lfp_event_validity[region][period_type] = [
                [validate_event(event, standard) for event in period.children]
                for period in self.children 
               ]

        self.save(self.lfp_event_validity[region], pickle_path)


class LFPMethods:
 
    def get_power(self):
        return self.get_average('get_power', stop_at=self.calc_opts.get('base', 'event'))
    

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
        ev = period.animal.lfp_event_validity[region]
        return {i: valid for i, valid in enumerate(ev[self.period_type][period.identifier])}


class LFPPeriod(Period, LFPMethods, LFPDataSelector, EventValidator):

    def __init__(self, animal, index, period_type, period_info, onset, events=None, 
                 target_period=None, is_relative=False, experiment=None):
        super().__init__(index, period_type, period_info, onset, experiment=experiment, 
                         target_period=target_period, is_relative=is_relative, events=events)
        self.animal = animal
        self.parent = animal
        padding = self.calc_opts['lfp_padding']
        start_pad, end_pad = np.round(np.array(padding) * self.lfp_sampling_rate).astype(int)
        self.duration_in_lfp_samples = round(self.duration * self.lfp_sampling_rate)
        conversion_factor = self.lfp_sampling_rate/self.sampling_rate 
        # For LFP, you need a subtraction for 0 indexing. 
        self.onset_in_lfp_samples = int(self.onset * conversion_factor) - 1
        period_events = (np.array(period_events) * conversion_factor).astype(int) - 1
        self.start_in_lfp_samples = self.onset_in_lfp_samples
        self.stop_in_lfp_samples = self.start_in_lfp_samples + self.duration_in_lfp_samples
        self.pad_start = self.start_in_lfp_samples - start_pad
        self.pad_stop = self.stop_in_lfp_samples + end_pad
        self._spectrogram = None
        self.brain_region = self.current_brain_region
        self.frequency_band = self.current_frequency_band
        
    @property
    def padded_data(self):
        return self.get_data_from_animal_dict(self.animal.processed_lfp, 
                                              self.pad_start, self.pad_stop)
        
    @property
    def unpadded_data(self):
        return self.get_data_from_animal_dict(
            self.animal.processed_lfp, self.start_in_lfp_samples, self.stop_in_lfp_samples)
    
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
                (event_start - self.lfp_onset)/self.lfp_sampling_rate + true_beginning - self.pre_event
                , 2)
            spect_end = round(spect_start + self.pre_event + self.post_event, 2)
            num_points = round(np.ceil((spect_end - spect_start) / bin_size - epsilon))  
            event_times = np.linspace(spect_start, spect_start + (num_points * bin_size), 
                                      num_points, endpoint=False)
            event_times = event_times[event_times < spect_end]

            # a binary mask that is True when a time bin in the spectrogram belongs to this event
            mask = (np.abs(time_bins[:, None] - event_times) <= epsilon).any(axis=1)

            events.append(LFPEvent(i, event_times, event_start, mask, self))
        
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

    def __init__(self, identifier, event_times, onset, mask, period):
        super().__init__(period, onset, identifier)
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
        return np.array(self.sliced_spectrogram)[:, self.mask]
    

