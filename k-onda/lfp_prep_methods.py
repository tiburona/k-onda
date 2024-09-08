from matlab_interface import MatlabInterface
from math_functions import filter_60_hz, divide_by_rms
from numpy import np


class LFPPrepMethods:

    @property
    def processed_lfp(self):
        if self.current_brain_region not in self._processed_lfp:
            self.process_lfp()
        return self._processed_lfp
    
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