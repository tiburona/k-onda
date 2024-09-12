from copy import copy
import numpy as np

from base_data import Data
    

class Bin(Data):
    
    def __init__(self, index, val, parent, parent_data):
        self.identifier = index
        self.val = val
        self.parent = parent
        self.parent_data = parent_data
        self.period_type = copy(self.selected_period_type)

    @property
    def calc(self):
        return self.val


class TimeBin(Bin):

    _name = 'time_bin'

    def __init__(self, index, val, parent, parent_data):
        super().__init__(index, val, parent, parent_data) 

        if self.calc_type == 'correlation':
            ts = np.arange(-self.calc_opts['lags'], self.calc_opts['lags'] + 1) / self.sampling_rate
        else:
            pre_stim, post_stim = [self.calc_opts['events'][self.period_type][opt] 
                                for opt in ['pre_stim', 'post_stim']]
            bin_size = self.calc_opts.get('bin_size')
            if bin_size is None:
                try:
                    bin_size = parent.spectrogram_bin_size
                except AttributeError:
                    bin_size = parent.parent.spectrogram_bin_size
            ts = np.arange(-pre_stim, post_stim, bin_size)
        
        # Round the timestamps to the nearest 10th of a microsecond
        ts = np.round(ts, decimals=7)

        self.time = ts[self.identifier] 


class TimeBinMethods:
         
    def get_time_bins(self, data):
        return [TimeBin(i, data_point, self, data) for i, data_point in enumerate(data)]

    @property
    def time_bins(self):
        return self.get_time_bins(self.calc)
    

class FrequencyBin(Bin, TimeBinMethods):

    _name = 'frequency_bin'

    def __init__(self, index, val, parent, parent_data):
        super().__init__(index, val, parent, parent_data) 

        # All of the below is determining the actual frequency this data represented
         
        if parent.name == 'period':
            representative = parent
        if 'calculator' in parent.name or parent.name == 'event':
            representative = parent.period
        elif parent.name == 'animal':
            representative = parent.children[0]
        elif parent.name == 'group':
            representative = parent.children[0].children[0]
        else:
            raise ValueError("Unexpected Object Type")
        
        if self.calc_type == 'power': 
            self.frequency = representative.spectrogram[1][index]
        elif 'phase' in self.calc_type:
            self.frequency = representative.frequency_bands[index][0]
        else:
            freq_range = list(range(self.freq_range[0], self.freq_range[1] + 1)) 
            if isinstance(parent_data, dict):
                shape = list(parent_data.values())[0].shape
            else:
                shape = parent_data.shape
            if shape[0] > len(freq_range):
                freq_range = np.linspace(freq_range[0], freq_range[-1], shape[0])
            self.frequency = freq_range[index]

    @property
    def mean_data(self):
        return np.mean(self.val)
    
    
class FrequencyBinMethods:

    def get_frequency_bins(self, data):
        return [FrequencyBin(i, data_point, self, data) for i, data_point in enumerate(data)]

    @property
    def frequency_bins(self):
        return self.get_frequency_bins(self.calc)
    

class BinMethods(TimeBinMethods, FrequencyBinMethods):
    pass




            
        
           
           

            




