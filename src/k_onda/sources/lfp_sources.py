import pint

from k_onda.central import make_time_series, Schema
from  k_onda.signals import TimeSeriesSignal
from .core import DataComponent, DataIdentity, GenericSource
from k_onda.central import Schema, AxisInfo, AxisKind, CoordInfo
from k_onda.utils import is_unitful


class LFPRecording(GenericSource):
    def __init__(self, session, data_loader_config, sampling_rate=None):
        # In some cases you can get the sampling rate from the recording, probably
        super().__init__(session, data_loader_config)
        self.sampling_rate = sampling_rate or data_loader_config.get('sampling_rate')
        if not is_unitful(self.sampling_rate):
            ureg = pint.get_application_registry()
            self.sampling_rate = self.sampling_rate * ureg.hertz

    @property
    def raw_data(self):
        if self._raw_data is None:
            self._raw_data = self.load_data()
        return self._raw_data
    
    @property
    def components(self):
        if not len(self._components):
            for channel_idx in range(self.raw_data.shape[1]):
                lfp_channel = LFPChannel(self, channel_idx)
                self._components.append(lfp_channel)
        return self._components

    def get_channel(self, idx):
        return self.raw_data[:, idx]  # View, not copy

    def load_data(self):
        if self.file_ext.startswith("ns"):
            return self.load_blackrock_file()
        data = super().load_data()
        if data is None:
            raise ValueError("Unknown file type")
        return data

    def load_blackrock_file(self):
        from neo.rawio import BlackrockRawIO

        filename = self.data_loader_config["file_path"]
        nsx_to_load = int(self.data_loader_config["file_ext"][2])
        reader = BlackrockRawIO(filename=filename, nsx_to_load=nsx_to_load)
        reader.parse_header()
        data = reader.nsx_datas[nsx_to_load][0]
        return data


class LFPChannel(DataComponent):
    output_class = TimeSeriesSignal

    def __init__(self, data_source, channel_idx):
        super().__init__(data_source)
        self.channel_idx = channel_idx
        self.sampling_rate = self.data_source.sampling_rate

    def data_loader(self):
        data = self.data_source.get_channel(self.channel_idx)
        da = make_time_series(data, self.sampling_rate)
        return da

    def to_signal(self):
        signal = super().to_signal()
        signal.sampling_rate = self.sampling_rate
        return signal

    @property
    def data_schema(self):
        return Schema(
            axes=[AxisInfo("time", kind=AxisKind.AXIS, metadim="time")], 
            value_metadim='V'
        )
    
    @property
    def region(self):
        regions = self.data_source.data_loader_config.get("row_to_region", {})
        return regions.get(self.channel_idx)


    @property
    def identifiers(self):
        ids = [self.channel_idx]
        row_to_region = self.data_source.data_loader_config.get("row_to_region")
        if row_to_region:
            ids.append(row_to_region[self.channel_idx])
        return ids


class LFPBrainRegion(DataIdentity):
    name = "lfp_brain_region"

    def __init__(self, data_components, config, subject=None):
        super().__init__(data_components, subject=subject)
        self._label = data_components[0].region

    @property
    def label(self):
        return self._label
    
    @property
    def region(self):
        return self.label
