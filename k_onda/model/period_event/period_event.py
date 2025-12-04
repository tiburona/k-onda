from k_onda.model import Data
from ..bins import BinMethods  

       
class Period(Data, BinMethods):

    _name = 'period'

    def __init__(self, index, period_type, period_info, onset, target_period=None, 
                 is_relative=False, duration=None, shift=None, experiment=None, events=None):
        super().__init__()
        self.identifier = index
        self.period_type = period_type
        self.period_info = period_info
        self.onset = onset
        self.target_period = target_period
        self.is_relative = is_relative
        self.experiment = experiment
        self.event_starts = events if events is not None else []
        self._events = []
        self.conditions = period_info.get('conditions')
        self.shift = shift
        self.duration = duration
        self.reference_period_type = period_info.get('reference')
        self.reference = None
        self.event_duration = period_info.get('event_duration')
        if target_period and hasattr(target_period, 'event_duration'):
            self.event_duration = target_period.event_duration
        self._stop = self._start + self.duration

    def __repr__(self):
        return (f"Period {self.animal.identifier} {self.period_type} "
                f"{self.identifier}")
    
    @property
    def unique_id(self):
        return '_'.join(
            [str(tag) for tag in 
             [self.parent.unique_id, self.period_type, self.identifier]]
             )

    @property
    def children(self):
        return self.events
    
    @property
    def pre(self):
        return self.pre_period
    
    @property
    def post(self):
        return self.post_period
    
    @property
    def start(self):
        return self._start - self.pre_period
    
    @property
    def stop(self):
        return self._start + self.duration + self.post_period
    
    @property
    def events(self):
        if not self._events:
            self.get_events()
        return self._events
        

class Event(Data, BinMethods):

    _name = 'event'

    def __init__(self, period, onset, index):
        super().__init__()
        self.period = period
        self.onset = onset
        self.identifier = index
        self.experiment = self.period.experiment
        self.parent = period
        self.period_type = self.period.period_type
        self.duration = self.pre_event + self.post_event
        self.reference = self.parent.reference

    @property
    def start(self):
        return self._start - self.pre_event
    
    @property
    def stop(self):
        return self._start + self.post_event

        
   
    
    
  

