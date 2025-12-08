from copy import copy

import numpy as np

from ..data import Data
    

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
        self.period_type = self.parent.period_type
        self.time = parent_data.isel(time_bin=index)


class TimeBinMethods:
         
    def get_time_bins(self, data):
        if data.ndim > 1 and all(dim > 1 for dim in data.shape):
            return [TimeBin(i, slice_, self, data) for i, slice_ in enumerate(data[..., -1])]
        return [TimeBin(i, data_point, self, data) for i, data_point in enumerate(data)]

    @property
    def time_bins(self):
        return self.get_time_bins(self.calc)
    

class FrequencyBin(Bin, TimeBinMethods):

    _name = 'frequency_bin'

    def __init__(self, index, val, parent, parent_data):
        super().__init__(index, val, parent, parent_data)  
        self.frequency = parent_data.coords['frequency'][index] 

    @property
    def mean_data(self):
        # todo: this isn't the right way to take a mean anymore now that everything is data arrays 
        # with attrs that need to be conserved 
        return np.mean(self.val)
    
    
class FrequencyBinMethods:

    # TODO: consider whether in the future if the possible dimensionality of data grows 
    # you are really going to need explicit trackers of what dim is what.

    def get_frequency_bins(self, data):
        return [FrequencyBin(i, data_point, self, data) for i, data_point in enumerate(data)]

    @property
    def frequency_bins(self):
        return self.get_frequency_bins(self.calc)
    

class BinMethods(TimeBinMethods, FrequencyBinMethods):

    @property
    def num_bins_per(self):
        if not hasattr(self, 'start') and hasattr(self, 'stop'):
            return None
        num_bins = self.to_int((self.stop-self.start)/self.bin_size)
        return num_bins




            
        
           
           

            




