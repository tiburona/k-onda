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
        if self.event_duration is None:
            self.event_duration = target_period.event_duration
        self.events_settings = self.calc_opts['events'].get(
            self.period_type, {'pre_stim': 0, 'post_stim': 1})
        self.pre_stim, self.post_stim = (self.events_settings[opt] 
                                         for opt in ['pre_stim', 'post_stim'])

    @property
    def children(self):
        return self.events
       
    @property
    def events(self):
        if not self._events:
            self.get_events()
        return self._events
        
    @property
    def reference(self):
        if self.is_relative:
            return None
        if not self.reference_period_type:
            return None
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
        self.pre_stim, self.post_stim = (events_settings[opt] for opt in ['pre_stim', 'post_stim'])
        self.duration = self.pre_stim + self.post_stim
        self.experiment = self.period.experiment

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
        
    @property
    def num_bins_per_event(self):
        bin_size = self.calc_opts.get('bin_size')
        pre_stim, post_stim = (self.calc_opts['events'][self.period_type].get(opt) 
                               for opt in ['pre_stim', 'post_stim'])
        return round((pre_stim + post_stim) / bin_size)
    
    
  

