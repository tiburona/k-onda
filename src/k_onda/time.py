from collections import defaultdict
import h5py
from datetime import datetime
from dataclasses import dataclass
from copy import deepcopy

from .central import ureg
from .utils import group_to_dict

class Epoch:
    
    def __init__(self, session, onset, duration, index=None, epoch_type=None, 
                 mode='pushdown'):
        self.session = session
        self.onset = onset
        self.duration = duration
        self.index = index
        self.epoch_type = epoch_type
        self.mode = mode
        self.t0 = self.onset
        self.t1 = self.onset + self.duration
        



@dataclass(frozen=True)
class TimeBase:
    fs_hz: float
    start_sample: int = 0
    start_datetime: datetime | None = None
    duration_sample: int | None = None
    duration_sec: float | None = None


@dataclass
class Marker:
    onset_sample: int | None = None
    onset_sec: float | None = None
    duration_sec: float | None = None
    code: int | str | None = None
    label: str | None = None


class NEVMixin:
    
    # todo: for now we're loading NEV file from converted matfile; need to also be
    # able to load directly of course
  
    def load_nev(self, nev_path, mat=True):
        if mat:
            with h5py.File(nev_path, 'r') as mat_file:
                data = group_to_dict(mat_file['NEV'])
                return data
      
    
    def get_start_time(self, data):
        raw_time = data['MetaTags']['DateTimeRaw']
        raw_time = [int(entry) for sublist in raw_time for entry in sublist]
        year, month, _dow, day, hour, minute, second, millisecond = raw_time
        dt = datetime(
            year, month, day, hour, minute, second,
            microsecond=millisecond * 1000
        )
        return dt
    
    def get_time_base(self, data):
        fs_hz = data['MetaTags']['TimeRes']
        start_datetime = self.get_start_time(data)
        duration_sample = data['MetaTags']['DataDuration']
        duration_sec = data['MetaTags']['DataDurationSec']
        return TimeBase(
            fs_hz=fs_hz, 
            start_datetime=start_datetime,
            duration_sample=duration_sample,
            duration_sec=duration_sec
            )
          
    def get_markers(self, data):

        markers = defaultdict(list)

        for i, code in enumerate(data['Data']['SerialDigitalIO']['UnparsedData'][0]):
            onset_sample = int(data['Data']['SerialDigitalIO']['TimeStamp'][i][0])
            onset_sec = data['Data']['SerialDigitalIO']['TimeStampSec'][i][0]
            marker = Marker(
                onset_sample=onset_sample,
                onset_sec=onset_sec,
                code=code
            )
            markers[code].append(marker)

        return markers




class Session(NEVMixin):
    
    def __init__(self, experiment, subject, config, ureg):
        self.experiment = experiment
        self.subject = subject
        self.config = config
        self.ureg = ureg
        self.subject.sessions.append(self)
        self._time_base = None
        self._markers = None
        self._epochs = defaultdict(list)
        self._start = None
        self._duration = None
      
    @property
    def time_base(self):
        if self._time_base is None:
            self.data_loader()
        return self._time_base
    
    @property
    def markers(self):
        if self._markers is None:
            self.data_loader()
        return self._markers

    @property
    def epochs(self):
        if not self._epochs:
            self.create_epochs()
        return self._epochs
    
    @property
    def start(self):
        if self._start is None:
            self.get_start_and_duration()
        return self._start
    
    @property
    def duration(self):
        if self._duration is None:
            self.get_start_and_duration()
        return self._duration

    def data_loader(self):
        if self.config.get('nev'):
            nev_data = self.load_nev(self.config['nev']['path'])
            self._time_base = self.get_time_base(nev_data)
            self._markers = self.get_markers(nev_data)
        # other kinds of data will follow
        
    def create_epochs(self):

        if 'nev' in self.config and 'epochs' in self.config['nev']:
            self.get_epochs(self.config['nev']['epochs'])
        # other ways to create epochs will follow

    def get_epochs(self, epoch_config):

        for epoch_type, epoch_info in epoch_config.items():
            code = epoch_info['code']
            markers = self.markers.get(code)
            if not markers:
                raise ValueError("Specified codes werent' found in the NEV file")
            
            unitful_markers = deepcopy(markers)
            
            if markers[0].onset_sample is not None:
                for marker in unitful_markers:
                    marker.onset_sample *= self.ureg.raw_sample 
                        
            if markers[0].onset_sec is not None:
                for marker in unitful_markers:
                    marker.onset_sec *= self.ureg.s
            else:
                for marker in unitful_markers:
                    marker.onset_sec = marker.onset_sample.to('s')
                    
            duration = epoch_info['duration'] * self.ureg('second')
            self._epochs[epoch_type].extend(
                [Epoch(self, marker.onset_sec, duration) for marker in unitful_markers])
            
    def get_start_and_duration(self):
        start_sample = self.time_base.start_sample
        start_sec = (start_sample * self.ureg.raw_sample).to('s') 
        self._start = start_sec

        duration_sec = self.time_base.duration_sec
        if duration_sec is not None:
            self._duration = duration_sec
        else:
            duration_sample = self.time_base.duration_sample
            if duration_sample:
                duration_sample *= self.ureg.raw_sample
                self._duration = duration_sample.to('s')
                
                
  