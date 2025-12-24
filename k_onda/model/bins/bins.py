from copy import copy

import numpy as np

from ..data import Data
    

class Bin(Data):
    
    def __init__(self, index, parent, parent_data):
        self.identifier = index
        self.parent = parent
        self.parent_data = parent_data
        self.period_type = copy(self.selected_period_type)


class TimeBin(Bin):

    _name = 'time_bin'

    def __init__(self, index, parent, parent_data):
        super().__init__(index, parent, parent_data) 
        self.period_type = self.parent.period_type
        self.parent = parent
        self.parent_data = parent_data

    @property
    def calc(self):
        # Index only when you actually need the data
        return self.parent_data.isel(time_bin=self.identifier)


class TimeBinMethods:
         
    def get_time_bins(self, data):
        # Assume time dimension is named "time_bin"
        n_bins = data.sizes["time_bin"]
        return [TimeBin(i, self, data) for i in range(n_bins)]

    @property
    def time_bins(self):
        return self.get_time_bins(self.calc)
    

class FrequencyBin(Bin, TimeBinMethods):

    _name = 'frequency_bin'

    def __init__(self, index, parent, parent_data):
        super().__init__(index, parent, parent_data)  
        self.frequency = parent_data.coords['frequency'][index] 

    @property
    def calc(self):
        # Index only when you actually need the data
        return self.parent_data.isel(freq_bin=self.identifier)

    @property
    def mean_data(self):
        # todo: this isn't the right way to take a mean anymore now that everything is data arrays 
        # with attrs that need to be conserved 
        return np.mean(self.val)
    
    
class FrequencyBinMethods:

    def get_frequency_bins(self, data):
        n_bins = data.sizes["freq_bin"]
        return [FrequencyBin(i, self, data) for i in range(n_bins)]

    @property
    def frequency_bins(self):
        return self.get_frequency_bins(self.calc)
    

class BinMethods(TimeBinMethods, FrequencyBinMethods):

    @property
    def num_bins_per(self):
        if not hasattr(self, 'start') and hasattr(self, 'stop'):
            return None
        num_bins = self.to_int(
            (self.stop - self.start).pint.to('second') / self.bin_size.pint.to('second'))
        return num_bins




            
        
           
           

            



