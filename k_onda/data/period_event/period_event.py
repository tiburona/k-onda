import numpy as np

from ..data import Data
from ..bins import BinMethods


class TimeLineMethods:
      
    def get_universal_res_onset(self):
        return self.onset/self.sampling_rate/self.finest_res
    
    @property
    def universal_res_start(self):
        return int(self.start/self.finest_res)
    
    @property
    def universal_res_stop(self):  # TODO make sure this is handles the right edge properly
        return np.ceil(self.stop/self.finest_res)
       

class Period(Data, BinMethods, TimeLineMethods):

    _name = 'period'

    def __init__(self, index, period_type, period_info, onset, target_period=None, 
                 is_relative=False, experiment=None, events=None):
        super().__init__()
        self.identifier = index
        self.period_type = period_type
        self.period_info = period_info
        self.onset = onset
        self.target_period = target_period
        self.is_relative = is_relative
        self.experiment = experiment
        self.event_starts = events if events is not None else []
        self.onset_in_seconds = self.onset/self.sampling_rate
        self._events = []
        self.conditions = period_info.get('conditions')
        self.shift = period_info.get('shift')
        self.duration = period_info.get('duration')
        self.reference_period_type = period_info.get('reference')
        self.event_duration = period_info.get('event_duration')
        if target_period and hasattr(target_period, 'event_duration'):
            self.event_duration = target_period.event_duration
        self._start = self.onset/self.sampling_rate 
        self._stop = self._start + self.duration
        self.universal_res_onset = self.get_universal_res_onset()
        self.duration_in_universal_res = self.duration/self.finest_res 

    def __repr__(self):
        return (f"Period {self.animal.identifier} {self.period_type} "
                f"{self.identifier}")

    @property
    def children(self):
        return self.events
    
    @property
    def start(self):
        return self._start - self.pre_period
    
    @property
    def stop(self):
        return self._stop + self.post_period
       
    @property
    def events(self):
        if not self._events:
            self.get_events()
        return self._events
    
    @property
    def reference_override(self):
        override = self.calc_opts.get('reference_override')
        return override and override[0] == self.name
        
    @property
    def reference(self):
        if (self.is_relative or not self.reference_period_type) and (
            not self.reference_override):
            return None
        if self.reference_override:
            return getattr(self, self.calc_opts['reference_override'][1])
        else:
            period_attr = f"{self.kind_of_data}_periods"
            if hasattr(self, period_attr):
                periods = getattr(self, period_attr)
            else:
                periods = getattr(self.parent, period_attr)
            return periods[self.reference_period_type][self.identifier]
        

class Event(Data, BinMethods, TimeLineMethods):

    _name = 'event'

    def __init__(self, period, onset, index):
        super().__init__()
        self.period = period
        self.onset = onset
        self.identifier = index
        self.experiment = self.period.experiment
        self._start = onset/self.sampling_rate
        self.parent = period
        self.period_type = self.period.period_type
        self.duration = self.pre_event + self.post_event
        self.universal_res_onset = self.get_universal_res_onset()


    @property
    def start(self):
        return self._start - self.pre_event
    
    @property
    def stop(self):
        return self._start + self.post_event

    @property
    def reference(self):
        if self.period.is_relative:
            return None
        reference_period_type = self.period.reference_period_type
        if not reference_period_type:
            return None
        else:
            period = self if self.name == 'period' else self.parent
            reference_periods = getattr(period.parent, f"{self.kind_of_data}_periods")
            return reference_periods[reference_period_type][period.identifier]
        
   
    
    
  

