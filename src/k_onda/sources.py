from neo.rawio import BlackrockRawIO
import numpy as np
import xarray as xr

from .signal import Signal, TimeSeriesSignal
from .calculator_mixins import SignalCalculatorMixin
from .dataarray_factories import make_time_series, get_time_coords
from .select_mixin import SelectMixin


class DataSource:
    """A file or resource containing experimental data."""
    
    def __init__(self, session, data_loader_config):
        self.session = session
        self.data_loader_config = data_loader_config
        self.file_path = self.data_loader_config["file_path"]
        self.file_ext = self.data_loader_config["file_ext"]
        self._raw_data = None 
    

class LFPRecording(DataSource):

    def __init__(self, session, data_loader_config, sampling_rate=None):
        # In some cases you can get the sampling rate from the recording, probably
        super().__init__(session, data_loader_config)
        self.sampling_rate = sampling_rate

    @property
    def raw_data(self):
        if self._raw_data is None:
            self._raw_data = self._load_all_channels()
        return self._raw_data
    
    def get_channel(self, idx):
        return self.raw_data[:, idx]  # View, not copy
    
    def _load_all_channels(self):

        if self.file_ext[0:2] == 'ns':
            return self.load_blackrock_file()
        else:
            raise ValueError("Unknown file type")

    def load_blackrock_file(self):
        filename = self.data_loader_config["file_path"]
        nsx_to_load = int(self.data_loader_config["file_ext"][2])
        reader = BlackrockRawIO(filename=filename, nsx_to_load=nsx_to_load)
        reader.parse_header()
        data = reader.nsx_datas[nsx_to_load][0]
        return data
    

class DataComponent:

    output_signal_class = Signal
    
    def __init__(self, source):
        self.data_source = source
        self._data = None
        
    @property
    def data(self):
        if self._data is None:
            self._data = self.data_loader()
        return self._data
  
    
class LFPChannel(DataComponent, SignalCalculatorMixin, SelectMixin):

    output_signal_class = TimeSeriesSignal

    def __init__(self, data_source, channel_idx):
        super().__init__(data_source)
        self.channel_idx = channel_idx
        self.sampling_rate = self.data_source.sampling_rate

    def data_loader(self):
        data = self.data_source.get_channel(self.channel_idx)
        da = make_time_series(data, self.sampling_rate) 
        return da  
