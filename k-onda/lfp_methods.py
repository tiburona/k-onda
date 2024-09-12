from matlab_interface import MatlabInterface
from math_functions import filter_60_hz, divide_by_rms, downsample
import numpy as np
from copy import deepcopy
from neo.rawio import BlackrockRawIO
import os


class LFPPrepMethods:

    def select_lfp_children(self):
        if self.calc_type == 'power':
            attr = 'lfp_periods'
        else:
            attr = f"{self.calc_type}_calculators"
        if self.selected_period_type:
            children = getattr(self, attr)[self.selected_period_type]
            if self.selected_period_type in self.calc_opts.get('periods', {}):
                return [child for i, child in enumerate(children)
                        if i in self.calc_opts['periods'][self.selected_period_type]]
            else:
                return children
        else:
            return self.get_all(attr)

    def lfp_prep(self):
        self.prepare_periods()
        if self.calc_type != 'power':
            getattr(self, f"prepare_{self.calc_type}_calculators")()

    def get_raw_lfp(self):
        path_constructor = deepcopy(self.experiment.info['lfp_path_constructor'])
        if path_constructor[-1] == 'identifier':
            path_constructor[-1] = self.identifier
        file_path = os.path.join(self.experiment.info['lfp_root'], self.identifier, 
                                 *path_constructor)
        try:
            reader = BlackrockRawIO(filename=file_path, nsx_to_load=3)
        except OSError:
            return {}
        reader.parse_header()
        if all([k not in self.animal_info for k in ['lfp_electrodes', 'lfp_from_stereotrodes']]):
            return {}
        data_to_return = {region: reader.nsx_datas[3][0][:, val]
                          for region, val in self.animal_info['lfp_electrodes'].items()}
        if self.animal_info.get('lfp_from_stereotrodes') is not None:
            data_to_return = self.get_lfp_from_stereotrodes(self, data_to_return, file_path)
        return data_to_return

    def get_lfp_from_stereotrodes(self, animal, data_to_return, file_path):
        lfp_from_stereotrodes_info = animal.animal_info['lfp_from_stereotrodes']
        nsx_num = lfp_from_stereotrodes_info['nsx_num']
        reader = BlackrockRawIO(filename=file_path, nsx_to_load=nsx_num)
        reader.parse_header()
        for region, region_data in lfp_from_stereotrodes_info['electrodes'].items():
            electrodes = region_data if isinstance(region_data, list) else region_data[animal.identifier]
            data = np.mean([reader.nsx_datas[nsx_num][0][:, electrode] for electrode in electrodes], axis=0)
            downsampled_data = downsample(data, self.experiment.info['sampling_rate'], 
                                          self.experiment.info['lfp_sampling_rate'])
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
                    if time_bin.data > self.calc_opts.get('threshold', 20) * standard:
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
    
    