from neo.rawio import BlackrockRawIO
from neo.io import NixIO
import numpy as np
import h5py
import os

from k_onda.utils import filter_60_hz, divide_by_rms, downsample, PrepMethods
from k_onda.interfaces import MatlabInterface
from .lfp_methods_and_data_structures import AmpCrossCorrCalculator, GrangerCalculator, CoherenceCalculator


class LFPPrepMethods(PrepMethods):

    def select_lfp_children(self):
        if self.calc_type in ['coherence', 'granger']: # TODO: add the others
            return self.select_children(f"{self.calc_type}_calculators")
        else:
            return self.select_children('lfp_periods')

    def lfp_prep(self):
        self.prep_data()
        self.prepare_periods()
        if self.calc_type in ['phase_relationship', 'granger', 'amp_crosscorr', 'coherence']:
            getattr(self, f"prepare_{self.calc_type}_calculators")()
    
    def prep_data(self):
        data_label = f"{self.selected_brain_region}_lfp"
        if data_label not in self.initialized:
            self.process_lfp()
            self.initialized.append(data_label)

    def load_blackrock_file(self, nsx_to_load=3):
        file_path = self.construct_path('lfp')
        try:
            reader = BlackrockRawIO(filename=file_path, nsx_to_load=nsx_to_load)
            reader.parse_header()
            data = reader.nsx_datas[nsx_to_load][0]
        except OSError:
            return {}
        except KeyError as e:
            print(f"KeyError: {e}.  We'll try to handle this exception by opening the file in Matlab.") 
            file_dir = os.path.dirname(file_path)
            h5_path = os.path.join(file_dir, f'output_data_ns{nsx_to_load}.h5')
            ml = MatlabInterface(self.env_config['matlab_config'])
            if not os.path.exists(h5_path):
                ml.open_nsx(file_path, nsx_to_load)
            with h5py.File(h5_path, 'r') as f:
                key = f'/NS{str(nsx_to_load)}_Data'
                data = f[key][:]
        return data
    
    def load_neo_file(self):
        """Load LFP data from a Neo-supported file (e.g., .nix, .h5)."""
        file_path = self.construct_path('lfp')
        ext = file_path.split('.')[-1].lower()
        if ext == 'nix':
            io = NixIO(file_path, mode='ro')
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        with io:
            block = io.read_block()
            return block.segments[0].analogsignals[0].magnitude

        
    def get_raw_lfp(self):
        if all(k not in self.animal_info for k in ('lfp_electrodes', 'lfp_from_stereotrodes')):
            return {}
        if self.io_opts.get('read_opts', {}).get('lfp_file_load') == 'neo':
            data = self.load_neo_file()
        else:
            data = self.load_blackrock_file(nsx_to_load=3)         
        data_to_return = {region: data[:, val]
                          for region, val in self.animal_info['lfp_electrodes'].items()}
        if self.animal_info.get('lfp_from_stereotrodes') is not None:
            data_to_return = self.get_lfp_from_stereotrodes(self, data_to_return)
        return data_to_return

    def get_lfp_from_stereotrodes(self, animal, data_to_return):
        lfp_from_stereotrodes_info = animal.animal_info['lfp_from_stereotrodes']
        data = self.load_blackrock_file(lfp_from_stereotrodes_info['nsx_num'])
        for region, region_data in lfp_from_stereotrodes_info['electrodes'].items():
            electrodes = region_data if isinstance(region_data, list) else region_data[animal.identifier]
            data = np.mean([data[:, electrode] for electrode in electrodes], axis=0)
            downsampled_data = downsample(data, self.experiment.exp_info['sampling_rate'], 
                                          self.experiment.exp_info['lfp_sampling_rate'])
            data_to_return[region] = downsampled_data
        return data_to_return

    @property
    def processed_lfp(self):
        if self.selected_brain_region not in self._processed_lfp:
            self.process_lfp()
        return self._processed_lfp
    
    def delete_lfp_data(self, region):
        if region in self.processed_lfp:
            self.processed_lfp.pop(region)
    
    def process_lfp(self):
        
        raw_lfp = self.get_raw_lfp()

        for brain_region in raw_lfp:
            data = raw_lfp[brain_region]/4
            filter = self.calc_opts.get('remove_noise', 'filtfilt')
            if filter == 'filtfilt':
                filtered = filter_60_hz(data, self.lfp_sampling_rate)
            elif filter == 'spectrum_estimation':
                ids = [self.identifier, brain_region]
                saved_calc_exists, filtered, pickle_path = self.load('lfp_output', 'filter', ids)
                if not saved_calc_exists:
                    ml = MatlabInterface(self.env_config['matlab_config'])
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
        region = self.selected_brain_region

        saved_calc_exists, validity, pickle_path = self.load('lfp_output', 'validity', [region, self.identifier])
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


    def prepare_coherence_calculators(self):
        cls = CoherenceCalculator
        self.coherence_calculators = self.prepare_region_relationship_calculators(cls)

    def prepare_correlation_calculators(self):
        cls = CorrelationCalculator
        self.correlation_calculators = self.prepare_region_relationship_calculators(cls)

    def prepare_granger_calculators(self):
        cls = GrangerCalculator
        self.granger_calculators = self.prepare_region_relationship_calculators(cls)

    def prepare_phase_relationship_calculators(self):
        cls = PhaseRelationshipCalculator
        self.phase_relationship_calculators = self.prepare_region_relationship_calculators(cls)

    def prepare_region_relationship_calculators(self, calc_class):
        brain_regions = self.calc_opts.get('region_set').split('_')
        return ({period_type: [calc_class(period, brain_regions) 
                               for period in self.lfp_periods[period_type]] 
                               for period_type in self.lfp_periods})