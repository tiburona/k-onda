from collections import defaultdict
import h5py
from datetime import datetime
from dataclasses import dataclass
from copy import deepcopy
import uuid
import re

from k_onda.mixins import ConfigSetter
from k_onda.utils import group_to_dict
from k_onda.loci import Epoch, EpochSet, Event, EventSet


from k_onda.sources import LFPRecording, PhyOutput, initialize_neurons_from_phy

data_source_registry = {
    'lfp': {'class': LFPRecording},
    'phy': {'class': PhyOutput, 'followup': initialize_neurons_from_phy}
}

        

@dataclass(frozen=True)
class TimeBase:
    fs_hz: float
    start_sample: int = 0
    start_datetime: datetime | None = None
    duration_sample: int | None = None
    duration_sec: float | None = None


@dataclass
class Onset:
    in_samples: int | None = None
    in_secs: float | None = None
    duration_sec: float | None = None
    code: int | str | None = None
    label: str | None = None


class NEVMixin:
    
    # todo: for now we're loading NEV file from converted matfile; need to also be
    # able to load directly of course
  
    def load_nev(self, nev_path, mat=True):

        nev_path = self.fill_fields(
            nev_path, 
            experiment=self.experiment, 
            subject=self.subject, 
            session=self
            )

        if mat:
            with h5py.File(nev_path, 'r') as mat_file:
                data = group_to_dict(mat_file['NEV'])
                return data
      
    
    def get_nev_start_time(self, data):
        raw_time = data['MetaTags']['DateTimeRaw']
        raw_time = [int(entry) for sublist in raw_time for entry in sublist]
        year, month, _dow, day, hour, minute, second, millisecond = raw_time
        dt = datetime(
            year, month, day, hour, minute, second,
            microsecond=millisecond * 1000
        )
        return dt
    
    def get_nev_time_base(self, data):
        fs_hz = data['MetaTags']['TimeRes']
        start_datetime = self.get_nev_start_time(data)
        duration_sample = data['MetaTags']['DataDuration']
        duration_sec = data['MetaTags']['DataDurationSec']
        return TimeBase(
            fs_hz=fs_hz, 
            start_datetime=start_datetime,
            duration_sample=duration_sample,
            duration_sec=duration_sec
            )
          
    def get_nev_onsets(self, data):

        markers = defaultdict(list)

        for i, code in enumerate(data['Data']['SerialDigitalIO']['UnparsedData'][0]):
            in_samples = int(data['Data']['SerialDigitalIO']['TimeStamp'][i][0])
            in_secs = data['Data']['SerialDigitalIO']['TimeStampSec'][i][0]
            marker = Onset(
                in_samples=in_samples,
                in_secs=in_secs,
                code=code
            )
            markers[code].append(marker)

        return markers
    

class Session(NEVMixin, ConfigSetter):
    
    def __init__(self, experiment, subject, config, label=None, conditions=None):
        self.uid = uuid.uuid4()
        self.experiment = experiment
        self.subject = subject
        self.config = config
        self.label = label or self.config.get('label')
        self.conditions = conditions or self.config.get('conditions')
        self.data_sources = []
        self.ureg = self.experiment.ureg
        self._time_base = None
        self._onsets = None
        self._epochs = defaultdict(EpochSet(self, epochs=[]))
        self._events = defaultdict(EventSet(self, events=[]))
        self._start = None
        self._duration = None
        self.initialize_data_sources()
        
    @property
    def display_id(self):
        parts = [self.experiment.id, self.subject.id]
        if self.label:
            parts.append(self.label)
        if self.time_base.start_datetime:
            parts.append(str(self.time_base.start_datetime))
        else:
            parts.append(str(self.uid)[:8])
        return ':'.join(parts)

    @property
    def time_base(self):
        if self._time_base is None:
            self.metadata_loader()
        return self._time_base
    
    @property
    def onsets(self):
        if self._onsets is None:
            self.metadata_loader()
        return self._onsets

    @property
    def epochs(self):
        if not self._epochs:
            self.create_epochs()
        return EpochSet(
            self, 
            [epoch for epoch_list in self._epochs.values() for epoch in epoch_list], 
            conditions=self.conditions)
    
    @property
    def events(self):
        if not self._events:
            self.create_events()
        return EventSet(
            self, 
            [event for event_list in self._events.values() for event in event_list], 
            conditions=self.conditions)

    
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
    
    def initialize_data_sources(self):
        for data_source in self.config.get('data_sources', []):
            data_source_config = self.resolve_config(data_source, self.experiment.data_sources_config)
            data_source_class = data_source_registry[data_source_config['registry_key']]['class']
            self.data_sources.append(data_source_class(self, data_source_config))

    def metadata_loader(self):

        if self.config.get('nev_path'):
            nev_data = self.load_nev(self.config['nev_path'])
            self._time_base = self.get_nev_time_base(nev_data)
            self._onsets = self.get_nev_onsets(nev_data)

        else:
            raise NotImplementedError("Need to add more metadata types")
        # other kinds of metadata will follow
        
    def create_epochs(self):

        for epoch_type in self.config.get('epochs', {}):
            epoch_config = self.experiment.epochs_config['epoch_type']
            conditions = epoch_config.get('conditions', {})
            if 'from_nev' in epoch_config:
                self.nev_epoch_config(epoch_type, epoch_config, conditions)
            elif 'relative_to' in epoch_config:
                self.relative_epoch_config(epoch_type, epoch_config, conditions)
                
    def nev_epoch_config(self, epoch_type, epoch_config, conditions):
        code = epoch_config['code']
        onsets = self.onsets.get(code)
        if not onsets:
            raise ValueError("Specified codes werent' found in the NEV file")
        
        unitful_onsets = deepcopy(onsets)
        
        if onsets[0].in_samples is not None:
            for onset in unitful_onsets:
                onset.in_samples *= self.ureg.raw_sample 
                    
        if onsets[0].in_secs is not None:
            for onset in unitful_onsets:
                onset.in_secs *= self.ureg.s
        else:
            for onset in unitful_onsets:
                onset.in_secs = onset.in_samples.to('s')
                
        duration = epoch_config['duration'] * self.ureg('second')
        self._epochs[epoch_type].extend([
            Epoch(self, onset.in_secs, duration, conditions=conditions, 
                  config=epoch_config) 
            for onset in unitful_onsets
            ])

    def relative_epoch_config(self, epoch_type, epoch_config, conditions):
        relative_to = epoch_config['relative_to']
        baseline_ind = None
        bracket_matches = re.findall(r'\[([^\]]*)\]', relative_to)
        if len(bracket_matches):
            baseline_ind = int(bracket_matches[0])
            relative_to = relative_to[:relative_to.index('[')]
        
        target_epochs = self.epochs[relative_to]
        shift = epoch_config['shift'] * self.experiment.ureg('s')
        duration = epoch_config['duration'] * self.experiment.ureg('s')
        if baseline_ind is not None:
            target_epochs = [target_epochs[baseline_ind]]
        self._epochs[epoch_type].extend(
            [Epoch(self, epoch.onset - shift, duration, conditions=conditions, 
                   config=epoch_config) 
            for epoch in target_epochs])
            
    def get_start_and_duration(self):
        start_sample = self.time_base.start_sample
        start_sec = (start_sample * self.ureg.raw_sample).to('s') 
        self._start = start_sec

        duration_sec = self.time_base.duration_sec
        if duration_sec is not None:
            self._duration = duration_sec * self.ureg.s
        else:
            duration_sample = self.time_base.duration_sample
            if duration_sample:
                duration_sample *= self.ureg.raw_sample
                self._duration = duration_sample.to('s')

    def create_events(self):
        if self.config.get('events'):
            # this is where events that had some definition independent of epochs would live
            pass
        else:
            for epoch_type, epoch in self._epochs:
                self._events[epoch_type].extend([epoch.events])

                
                
  