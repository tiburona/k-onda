from base_data import Data
from bins import BinMethods



class Period(Data, BinMethods):

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
        self._events = []
        self.shift = period_info.get('shift')
        self.duration = period_info.get('duration')
        self.reference_period_type = period_info.get('reference_period_type')
        self.event_duration = period_info.get('event_duration')
        if target_period and hasattr(target_period, 'event_duration'):
            self.event_duration = target_period.event_duration
        self.pre_period = self.calc_opts.get('pre_period')
        self.zero_point = self.pre_period
        if 'events' in self.calc_opts:
            self.events_settings = self.calc_opts['events'].get(
                self.period_type, {'pre_stim': 0, 'post_stim': 1})
            self._pre_stim, self._post_stim = (self.events_settings[opt] 
                                         for opt in ['pre_stim', 'post_stim'])
        else:
            self._pre_stim, self._post_stim, self.events_settings = (None, None, None)
        
    @property
    def pre_stim(self):
        return self._pre_stim
    
    @property
    def post_stim(self):
        return self._post_stim

    @property
    def children(self):
        return self.events
       
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
            return self.periods[self.reference_period_type][self.identifier]
        

class Event(Data, BinMethods):

    _name = 'event'

    def __init__(self, period, index):
        super().__init__()
        self.period = period
        self.identifier = index
        self.parent = period
        self.parent = period
        self.period_type = self.period.period_type
        events_settings = self.calc_opts['events'].get(self.period_type, 
                                                       {'pre_stim': 0, 'post_stim': 1})
        self._pre_stim, self._post_stim = (events_settings[opt] 
                                           for opt in ['pre_stim', 'post_stim'])
        self.duration = self.pre_stim + self.post_stim
        self.experiment = self.period.experiment

    @property
    def pre_stim(self):
        return self._pre_stim
    
    @property
    def post_stim(self):
        return self._post_stim

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
        
   
    
    
  

